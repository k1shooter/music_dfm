"""Training and sampling runners for FSNTG-v2."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from music_graph_dfm.constants import GRAPH_KERNEL_APPROX_COORDS, SPAN_CHANNELS
from music_graph_dfm.data import FSNTGV2JSONDataset, collate_states, infer_vocab_sizes
from music_graph_dfm.diffusion.ctmc import ctmc_sample
from music_graph_dfm.diffusion.edit_flow import (
    derive_oracle_edit_move,
    editflow_rate_loss,
    random_edit_augmentation_step,
    sample_multistep_supervision_segment,
    sample_forward_edit_ctmc_source,
    sample_edit_ctmc_step,
)
from music_graph_dfm.diffusion.losses import auxiliary_denoising_loss, music_structure_loss, rate_matching_loss
from music_graph_dfm.diffusion.masking import coordinate_masks, enforce_state_constraints
from music_graph_dfm.diffusion.schedules import StructureFirstSchedule
from music_graph_dfm.diffusion.state_ops import (
    PriorConfig,
    batch_to_coords,
    coords_to_batch,
    sample_forward_path,
    sample_prior,
)
from music_graph_dfm.models import FSNTGV2HeteroTransformer, ModelConfig, SimpleFactorizedBaseline
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import FSNTGV2State
from music_graph_dfm.whole_song.generation import build_long_context_template, generate_whole_song

LOGGER = logging.getLogger(__name__)


def _load_codecs(data_root: Path) -> tuple[RhythmTemplateVocab, PitchTokenCodec]:
    import json

    rhythm = RhythmTemplateVocab.from_dict(json.loads((data_root / "rhythm_templates.json").read_text(encoding="utf-8")))
    pitch = PitchTokenCodec.from_dict(json.loads((data_root / "pitch_codec.json").read_text(encoding="utf-8")))
    return rhythm, pitch


def _move_to_device(obj, device):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def _template_spec(rhythm_vocab: RhythmTemplateVocab, max_vocab: int) -> dict:
    onset_bin = [0 for _ in range(max_vocab)]
    duration_class = [0 for _ in range(max_vocab)]
    tie_flag = [0 for _ in range(max_vocab)]
    extension_class = [0 for _ in range(max_vocab)]

    for idx in range(min(max_vocab, rhythm_vocab.vocab_size)):
        tpl = rhythm_vocab.decode(idx)
        onset_bin[idx] = int(tpl.onset_bin)
        duration_class[idx] = int(tpl.duration_class)
        tie_flag[idx] = int(tpl.tie_flag)
        extension_class[idx] = int(tpl.extension_class)

    return {
        "onset_bin": onset_bin,
        "duration_class": duration_class,
        "tie_flag": tie_flag,
        "extension_class": extension_class,
        "duration_ticks": list(rhythm_vocab.duration_ticks),
        "onset_bins": int(rhythm_vocab.onset_bins),
        "tie_extension_fraction": float(rhythm_vocab.tie_extension_fraction),
    }


def build_model(vocab_sizes: dict, model_cfg: dict, rhythm_vocab: RhythmTemplateVocab):
    cfg = ModelConfig(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    kind = str(model_cfg.get("kind", "full"))
    spec = _template_spec(rhythm_vocab, max(2, int(vocab_sizes["note.template"])))

    if kind in {"baseline", "posterior", "progress_like"}:
        return SimpleFactorizedBaseline(vocab_sizes=vocab_sizes, cfg=cfg)
    return FSNTGV2HeteroTransformer(
        vocab_sizes=vocab_sizes,
        cfg=cfg,
        template_spec=spec,
    )


def save_checkpoint(path: str | Path, model, optimizer=None, extra: dict | None = None):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("save_checkpoint requires torch") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model, optimizer=None):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("load_checkpoint requires torch") from exc

    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def read_checkpoint_extra(path: str | Path) -> dict:
    try:
        import torch
    except Exception:
        return {}

    payload = torch.load(Path(path), map_location="cpu")
    return dict(payload.get("extra", {}))


def _coords_to_states(coords: Dict[str, "torch.Tensor"], base_batch: dict) -> List[FSNTGV2State]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_coords_to_states requires torch") from exc

    span_mask = base_batch["span_mask"]
    note_mask = base_batch["note_mask"]
    out = []
    for b in range(span_mask.shape[0]):
        s = int(span_mask[b].sum().item())
        n = int(note_mask[b].sum().item())
        span_attrs = {channel: coords[f"span.{channel}"][b, :s].detach().cpu().tolist() for channel in SPAN_CHANNELS}
        note_attrs = {
            channel: coords[f"note.{channel}"][b, :n].detach().cpu().tolist()
            for channel in ["active", "pitch_token", "velocity", "role"]
        }
        host = coords["note.host"][b, :n].detach().cpu().tolist()
        template = coords["note.template"][b, :n].detach().cpu().tolist()
        e_ss = coords["e_ss.relation"][b, :s, :s].detach().cpu().tolist()
        tps = int(base_batch["ticks_per_span"][b].item())
        span_starts = [j * tps for j in range(s)]
        meta = dict(base_batch.get("meta", [{}])[b]) if b < len(base_batch.get("meta", [])) else {}

        out.append(
            FSNTGV2State(
                span_attrs=span_attrs,
                note_attrs=note_attrs,
                host=host,
                template=template,
                e_ss=e_ss,
                span_starts=span_starts,
                ticks_per_span=tps,
                metadata=meta,
            )
        )
    return out


def _make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda items: items,
    )


def _graph_kernels(vocab_sizes: dict, enabled: bool):
    if not enabled:
        return {}
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    kernels = {}
    for coord, radius in [("span.harm_root", 1), ("note.pitch_token", 2)]:
        v = int(vocab_sizes[coord])
        mat = torch.zeros((v, v), dtype=torch.float32)
        for i in range(v):
            neigh = list(range(max(0, i - radius), min(v, i + radius + 1)))
            p = 1.0 / max(1, len(neigh))
            for j in neigh:
                mat[i, j] = p
        kernels[coord] = mat
    return kernels


def _load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _checkpoint_extra(
    cfg: dict,
    vocab_sizes: dict,
    mode: str,
    data_root: Path,
    graph_kernel_enabled: bool,
) -> dict:
    path_type = str(cfg.get("diffusion", {}).get("path_type", "mixture"))
    kernel_coords = list(GRAPH_KERNEL_APPROX_COORDS) if graph_kernel_enabled else []
    graph_kernel_meta = {
        "enabled": bool(graph_kernel_enabled),
        "path_type": path_type,
        "coordinates": kernel_coords,
        "approximate": bool(path_type == "graph_kernel" and graph_kernel_enabled),
        "target_distribution": "q_t(x^c)=(1-kappa)delta_{x0^c}+kappa*K[x1^c,:] for graph-kernel coords",
        "target_rate_approximation": "off-diagonal Poisson matching with eta_c*K[x1^c,v] for v!=x_t^c",
    }
    return {
        "cfg": cfg,
        "mode": mode,
        "vocab_sizes": vocab_sizes,
        "model_cfg": dict(cfg.get("model", {})),
        "train_cfg": dict(cfg.get("train", {})),
        "diffusion_cfg": dict(cfg.get("diffusion", {})),
        "data_meta": {
            "data_root": str(data_root),
            "stats": _load_optional_json(data_root / "stats.json"),
            "preprocessing_config": _load_optional_json(data_root / "preprocessing_config.json"),
            "rhythm_template_vocab": _load_optional_json(data_root / "rhythm_templates.json"),
            "pitch_codec": _load_optional_json(data_root / "pitch_codec.json"),
        },
        "graph_kernel_is_approximate": bool(graph_kernel_meta["approximate"]),
        "graph_kernel_target_rate_mode": graph_kernel_meta["target_rate_approximation"],
        "graph_kernel": graph_kernel_meta,
    }


def _init_training_context(cfg: dict):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for training") from exc

    data_root = Path(cfg["data_root"]).expanduser().resolve()
    train_set = FSNTGV2JSONDataset(data_root / "train.jsonl")
    valid_set = FSNTGV2JSONDataset(data_root / "valid.jsonl")
    if len(train_set) == 0:
        raise RuntimeError(f"No training data found in {data_root}")

    rhythm_vocab, pitch_codec = _load_codecs(data_root)
    vocab_sizes = infer_vocab_sizes(train_set.states)

    model = build_model(vocab_sizes=vocab_sizes, model_cfg=cfg["model"], rhythm_vocab=rhythm_vocab)
    device_name = cfg.get("device", "cpu")
    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("learning_rate", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    schedule = StructureFirstSchedule(**cfg["diffusion"].get("schedule", {}))
    prior_cfg = PriorConfig(**cfg["diffusion"].get("prior", {}))
    path_type = str(cfg["diffusion"].get("path_type", "mixture"))
    graph_kernel_enabled = bool(cfg["diffusion"].get("graph_kernel", {}).get("enabled", False))
    if path_type == "graph_kernel" and graph_kernel_enabled:
        LOGGER.warning("=" * 88)
        LOGGER.warning(
            "EXPERIMENTAL GRAPH-KERNEL MODE ENABLED for span.harm_root/note.pitch_token. "
            "Target rates use an approximate off-diagonal matching surrogate."
        )
        LOGGER.warning("This run should be treated as experimental and reported as approximate.")
        LOGGER.warning("=" * 88)
    if path_type == "graph_kernel" and not graph_kernel_enabled:
        LOGGER.warning(
            "path_type=graph_kernel but diffusion.graph_kernel.enabled=false; falling back to mixture kernels map."
        )
        path_type = "mixture"

    kernels = _graph_kernels(vocab_sizes, enabled=graph_kernel_enabled)
    kernels = {k: v.to(device) for k, v in kernels.items()}

    train_loader = _make_loader(
        train_set,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    valid_loader = _make_loader(
        valid_set,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    epochs = int(cfg["train"].get("epochs", 20))
    rng = random.Random(int(cfg.get("seed", 7)))
    compat_table = torch.tensor(pitch_codec.compatibility_table(), dtype=torch.float32, device=device)

    return {
        "data_root": data_root,
        "train_set": train_set,
        "valid_set": valid_set,
        "rhythm_vocab": rhythm_vocab,
        "pitch_codec": pitch_codec,
        "compat_table": compat_table,
        "vocab_sizes": vocab_sizes,
        "model": model,
        "optimizer": optimizer,
        "schedule": schedule,
        "prior_cfg": prior_cfg,
        "path_type": path_type,
        "graph_kernel_enabled": graph_kernel_enabled,
        "kernels": kernels,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "epochs": epochs,
        "rng": rng,
        "device": device,
    }


def run_training_dfm(cfg: dict) -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for training") from exc

    ctx = _init_training_context(cfg)
    model = ctx["model"]
    optimizer = ctx["optimizer"]
    train_loader = ctx["train_loader"]
    valid_loader = ctx["valid_loader"]
    schedule = ctx["schedule"]
    prior_cfg = ctx["prior_cfg"]
    path_type = ctx["path_type"]
    kernels = ctx["kernels"]
    vocab_sizes = ctx["vocab_sizes"]
    rhythm_vocab = ctx["rhythm_vocab"]
    pitch_codec = ctx["pitch_codec"]
    compat_table = ctx["compat_table"]
    rng = ctx["rng"]
    device = ctx["device"]

    beta_aux = float(cfg["train"].get("beta_aux", 0.2))
    beta_structure = float(cfg["train"].get("beta_structure", 0.1))
    structure_loss_every_k_steps = max(1, int(cfg["train"].get("structure_loss_every_k_steps", 1)))
    structure_loss_subsample_notes = max(0, int(cfg["train"].get("structure_loss_subsample_notes", 0)))
    structure_loss_subsample_pairs = max(0, int(cfg["train"].get("structure_loss_subsample_pairs", 0)))
    fast_music_loss_only = bool(cfg["train"].get("fast_music_loss_only", False))
    epochs = int(ctx["epochs"])
    history = []
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        full_structure_steps = 0

        for states in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = _move_to_device(collate_states(states), device)
            x1 = batch_to_coords(batch)
            x0 = sample_prior(batch, vocab_sizes=vocab_sizes, cfg=prior_cfg)

            t_float = rng.uniform(0.01, 0.99)
            xt, xt_is_x0, eta, path_meta = sample_forward_path(
                x0,
                x1,
                t=t_float,
                schedule=schedule,
                path_type=path_type,
                graph_kernels=kernels,
            )
            xt = enforce_state_constraints(xt, batch)
            batch_xt = coords_to_batch(batch, xt)
            outputs = model(batch_xt, torch.tensor(t_float, device=device))
            masks = coordinate_masks(batch_xt)

            loss_rate = rate_matching_loss(outputs, xt, x1, xt_is_x0, eta, masks, path_meta)
            loss_aux = auxiliary_denoising_loss(outputs, x1, masks)
            run_full_structure = (global_step % structure_loss_every_k_steps == 0) and (not fast_music_loss_only)
            loss_structure_dict = music_structure_loss(
                outputs=outputs,
                x_t=xt,
                batch=batch_xt,
                masks=masks,
                rhythm_vocab=rhythm_vocab,
                pitch_codec=pitch_codec,
                compat_table=compat_table,
                fast_music_loss_only=not run_full_structure,
                structure_loss_subsample_notes=structure_loss_subsample_notes,
                structure_loss_subsample_pairs=structure_loss_subsample_pairs,
            )
            loss_structure = loss_structure_dict["total"]
            if run_full_structure:
                full_structure_steps += 1
                LOGGER.info("Decoded structure loss executed at global_step=%d", global_step)
            loss = loss_rate + beta_aux * loss_aux + beta_structure * loss_structure

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            batches += 1
            global_step += 1

        train_loss /= max(1, batches)

        model.eval()
        valid_loss = 0.0
        vb = 0
        with torch.no_grad():
            for states in valid_loader:
                batch = _move_to_device(collate_states(states), device)
                x1 = batch_to_coords(batch)
                x0 = sample_prior(batch, vocab_sizes=vocab_sizes, cfg=prior_cfg)
                xt, xt_is_x0, eta, path_meta = sample_forward_path(
                    x0,
                    x1,
                    t=0.5,
                    schedule=schedule,
                    path_type=path_type,
                    graph_kernels=kernels,
                )
                xt = enforce_state_constraints(xt, batch)
                batch_xt = coords_to_batch(batch, xt)
                outputs = model(batch_xt, torch.tensor(0.5, device=device))
                masks = coordinate_masks(batch_xt)

                loss_rate = rate_matching_loss(outputs, xt, x1, xt_is_x0, eta, masks, path_meta)
                loss_aux = auxiliary_denoising_loss(outputs, x1, masks)
                loss_structure_dict = music_structure_loss(
                    outputs=outputs,
                    x_t=xt,
                    batch=batch_xt,
                    masks=masks,
                    rhythm_vocab=rhythm_vocab,
                    pitch_codec=pitch_codec,
                    compat_table=compat_table,
                    fast_music_loss_only=fast_music_loss_only,
                    structure_loss_subsample_notes=structure_loss_subsample_notes,
                    structure_loss_subsample_pairs=structure_loss_subsample_pairs,
                )
                loss_structure = loss_structure_dict["total"]
                loss = loss_rate + beta_aux * loss_aux + beta_structure * loss_structure

                valid_loss += float(loss.item())
                vb += 1
        valid_loss /= max(1, vb)

        summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "mode": "dfm",
            "structure_loss_full_steps": full_structure_steps,
            "structure_loss_every_k_steps": structure_loss_every_k_steps,
            "fast_music_loss_only": fast_music_loss_only,
        }
        history.append(summary)
        print(summary)

        if (epoch + 1) % int(cfg["train"].get("save_every", 1)) == 0:
            save_checkpoint(
                Path(cfg["train"].get("checkpoint_dir", "artifacts/checkpoints")) / f"epoch_{epoch + 1}.pt",
                model,
                optimizer=optimizer,
                extra=_checkpoint_extra(
                    cfg=cfg,
                    vocab_sizes=vocab_sizes,
                    mode="dfm",
                    data_root=ctx["data_root"],
                    graph_kernel_enabled=bool(ctx["graph_kernel_enabled"]),
                ),
            )

    return {"history": history, "final": history[-1] if history else {}}


def run_training_editflow(cfg: dict) -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for training") from exc

    ctx = _init_training_context(cfg)
    model = ctx["model"]
    optimizer = ctx["optimizer"]
    train_loader = ctx["train_loader"]
    valid_loader = ctx["valid_loader"]
    vocab_sizes = ctx["vocab_sizes"]
    rng = ctx["rng"]
    device = ctx["device"]
    epochs = int(ctx["epochs"])
    history = []
    editflow_mode = str(cfg.get("train", {}).get("editflow_mode", "one_step_oracle"))
    source_steps = int(cfg.get("train", {}).get("editflow_source_steps", 1))
    use_random_aug = bool(cfg.get("train", {}).get("editflow_random_augmentation", False))
    allow_multistep_oracle = bool(cfg.get("train", {}).get("allow_multistep_oracle", False))
    if editflow_mode not in {"one_step_oracle", "multistep_segment"}:
        raise ValueError("train.editflow_mode must be one of: one_step_oracle, multistep_segment")
    if editflow_mode == "one_step_oracle" and source_steps != 1 and not allow_multistep_oracle:
        raise ValueError(
            "one-step oracle mode supports source_steps=1 by default. "
            "Set train.editflow_source_steps=1 or enable allow_multistep_oracle for experiments."
        )
    if editflow_mode == "multistep_segment" and source_steps < 2:
        raise ValueError("multistep_segment mode requires train.editflow_source_steps >= 2")
    if editflow_mode == "multistep_segment" and use_random_aug:
        raise ValueError("editflow_random_augmentation cannot be used with multistep_segment mode")

    if editflow_mode == "multistep_segment":
        LOGGER.warning(
            "EditFlow multistep_segment mode is experimental. "
            "Supervision uses sampled trajectory segments, not exact full marginalization."
        )

    def _build_edit_supervision_batch(target_states):
        source_states: list[FSNTGV2State] = []
        oracle = []
        t_values: list[float] = []

        if editflow_mode == "multistep_segment":
            for st in target_states:
                source_state, prev_state, move, t_value = sample_multistep_supervision_segment(
                    target_state=st,
                    vocab_sizes=vocab_sizes,
                    rng=rng,
                    num_steps=max(2, source_steps),
                    h=1.0 / max(1, source_steps),
                )
                source_states.append(source_state)
                oracle.append(move if move is not None else derive_oracle_edit_move(source_state, prev_state))
                t_values.append(float(t_value))
            return source_states, oracle, t_values

        # one_step_oracle (stable baseline)
        if use_random_aug:
            source_states = [random_edit_augmentation_step(st, vocab_sizes=vocab_sizes, rng=rng) for st in target_states]
        else:
            source_states = [
                sample_forward_edit_ctmc_source(
                    target_state=st,
                    vocab_sizes=vocab_sizes,
                    rng=rng,
                    num_steps=max(1, source_steps),
                    h=1.0 / max(1, source_steps),
                )
                for st in target_states
            ]
        oracle = [derive_oracle_edit_move(src, tgt) for src, tgt in zip(source_states, target_states)]
        t_values = [rng.uniform(0.01, 0.99) for _ in source_states]
        return source_states, oracle, t_values

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0

        for states in train_loader:
            optimizer.zero_grad(set_to_none=True)
            source_states, oracle, t_values = _build_edit_supervision_batch(states)
            batch = _move_to_device(collate_states(source_states), device)
            t = torch.tensor(t_values, device=device)
            edit_outputs = model.forward_edit(batch, t)
            loss = editflow_rate_loss(edit_outputs, oracle)

            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            batches += 1

        train_loss /= max(1, batches)

        model.eval()
        valid_loss = 0.0
        vb = 0
        with torch.no_grad():
            for states in valid_loader:
                source_states, oracle, t_values = _build_edit_supervision_batch(states)
                batch = _move_to_device(collate_states(source_states), device)
                edit_outputs = model.forward_edit(batch, torch.tensor(t_values, device=device))
                loss = editflow_rate_loss(edit_outputs, oracle)
                valid_loss += float(loss.item())
                vb += 1
        valid_loss /= max(1, vb)

        summary = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "mode": "editflow",
            "editflow_mode": editflow_mode,
            "source_steps": source_steps,
            "source_process": "random_augmentation" if use_random_aug else "forward_edit_ctmc",
            "experimental": bool(editflow_mode != "one_step_oracle"),
        }
        history.append(summary)
        print(summary)

        if (epoch + 1) % int(cfg["train"].get("save_every", 1)) == 0:
            save_checkpoint(
                Path(cfg["train"].get("checkpoint_dir", "artifacts/checkpoints")) / f"epoch_{epoch + 1}.pt",
                model,
                optimizer=optimizer,
                extra=_checkpoint_extra(
                    cfg=cfg,
                    vocab_sizes=vocab_sizes,
                    mode="editflow",
                    data_root=ctx["data_root"],
                    graph_kernel_enabled=bool(ctx["graph_kernel_enabled"]),
                )
                | {
                    "editflow_mode": editflow_mode,
                    "editflow_source_steps": source_steps,
                    "editflow_source_process": "random_augmentation" if use_random_aug else "forward_edit_ctmc",
                    "editflow_is_experimental": bool(editflow_mode != "one_step_oracle"),
                    "editflow_training_objective": (
                        "trajectory_segment_supervision"
                        if editflow_mode == "multistep_segment"
                        else "one_step_oracle"
                    ),
                },
            )

    return {"history": history, "final": history[-1] if history else {}}


def run_training(cfg: dict) -> dict:
    mode = str(cfg.get("train", {}).get("mode", "dfm"))
    if mode == "editflow":
        return run_training_editflow(cfg)
    return run_training_dfm(cfg)


def _load_model_for_sampling(checkpoint: Path, data_root: Path, split: str, device: str):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    ds = FSNTGV2JSONDataset(data_root / f"{split}.jsonl")
    if len(ds) == 0:
        raise RuntimeError(f"Split {split} is empty")

    rhythm_vocab, pitch_codec = _load_codecs(data_root)

    payload = torch.load(checkpoint, map_location="cpu")
    extra = payload.get("extra", {})
    vocab_sizes = extra.get("vocab_sizes") or infer_vocab_sizes(ds.states)
    model_cfg = extra.get("model_cfg") or {"kind": "full", "hidden_dim": 256, "num_layers": 6, "num_heads": 8, "dropout": 0.1}

    model = build_model(vocab_sizes=vocab_sizes, model_cfg=model_cfg, rhythm_vocab=rhythm_vocab)
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    model.load_state_dict(payload["model"])
    model.eval()
    return ds, model, vocab_sizes, rhythm_vocab, pitch_codec, dev, extra


def _sample_state(model, vocab_sizes: dict, ref_state: FSNTGV2State, num_steps: int, device):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    batch = _move_to_device(collate_states([ref_state]), device)
    x0 = sample_prior(batch, vocab_sizes=vocab_sizes, cfg=PriorConfig())
    with torch.no_grad():
        coords = ctmc_sample(
            model=model,
            init_coords=x0,
            base_batch=batch,
            num_steps=num_steps,
            t_start=1e-3,
            t_end=0.999,
        )
    coords = enforce_state_constraints(coords, batch)
    return _coords_to_states(coords, batch)[0]


def _sample_state_edit_one_step(model, ref_state: FSNTGV2State, num_steps: int, device):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    state = ref_state.copy()
    for step in range(max(1, int(num_steps))):
        t = torch.tensor((step + 1) / max(1, num_steps), device=device)
        batch = _move_to_device(collate_states([state]), device)
        with torch.no_grad():
            edit_outputs = model.forward_edit(batch, t)
        state = sample_edit_ctmc_step(state=state, edit_outputs_single=edit_outputs, h=1.0 / max(1, num_steps))
    return state


def _sample_state_edit_multistep(model, ref_state: FSNTGV2State, num_steps: int, device):
    """Experimental multistep edit sampler with micro-steps per time slice."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required") from exc

    state = ref_state.copy()
    steps = max(1, int(num_steps))
    for step in range(steps):
        for micro in range(2):
            t_val = (step + (micro + 1) / 2.0) / steps
            t = torch.tensor(t_val, device=device)
            batch = _move_to_device(collate_states([state]), device)
            with torch.no_grad():
                edit_outputs = model.forward_edit(batch, t)
            state = sample_edit_ctmc_step(
                state=state,
                edit_outputs_single=edit_outputs,
                h=0.5 / steps,
            )
    return state


def generate_samples_from_checkpoint(
    checkpoint: Path,
    data_root: Path,
    split: str = "test",
    num_samples: int = 16,
    num_steps: int = 96,
    device: str = "cpu",
    sampler_mode: str = "dfm",
    whole_song_mode: str | None = None,
    whole_song_segments: int = 4,
) -> List[FSNTGV2State]:
    ds, model, vocab_sizes, _rhythm_vocab, _pitch_codec, dev, ckpt_extra = _load_model_for_sampling(
        checkpoint=checkpoint,
        data_root=data_root,
        split=split,
        device=device,
    )

    out = []
    if sampler_mode == "dfm":
        editflow_mode = "n/a"
        sample_fn = _sample_state
    else:
        editflow_mode = str(ckpt_extra.get("editflow_mode", "one_step_oracle"))
        if editflow_mode == "multistep_segment":
            sample_fn = _sample_state_edit_multistep
        else:
            sample_fn = _sample_state_edit_one_step

    if sampler_mode == "editflow":
        LOGGER.info("Sampling editflow with mode=%s", editflow_mode)
    for i in range(num_samples):
        if whole_song_mode is None:
            ref = ds[i % len(ds)]
            if sampler_mode == "dfm":
                sampled = sample_fn(model, vocab_sizes, ref_state=ref, num_steps=num_steps, device=dev)
            else:
                sampled = sample_fn(model, ref_state=ref, num_steps=num_steps, device=dev)
            out.append(sampled)
            continue

        if whole_song_mode == "long_context":
            refs = [ds[(i * whole_song_segments + k) % len(ds)] for k in range(whole_song_segments)]
            long_template = build_long_context_template(refs)
            if sampler_mode == "dfm":
                sampled_long = sample_fn(
                    model,
                    vocab_sizes,
                    ref_state=long_template,
                    num_steps=num_steps,
                    device=dev,
                )
            else:
                sampled_long = sample_fn(model, ref_state=long_template, num_steps=num_steps, device=dev)
            out.append(generate_whole_song([sampled_long], mode="long_context"))
            continue

        if whole_song_mode == "stitching_baseline":
            refs = [ds[(i * whole_song_segments + k) % len(ds)] for k in range(whole_song_segments)]
            if sampler_mode == "dfm":
                segments = [
                    sample_fn(model, vocab_sizes, ref_state=ref, num_steps=num_steps, device=dev)
                    for ref in refs
                ]
            else:
                segments = [sample_fn(model, ref_state=ref, num_steps=num_steps, device=dev) for ref in refs]
            out.append(generate_whole_song(segments, mode="stitching_baseline"))
            continue

        raise ValueError("whole_song_mode must be None, 'long_context', or 'stitching_baseline'")

    return out
