"""Training and sampling runners for FSNTG-v2."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List

from music_graph_dfm.datasets import FSNTGV2JSONDataset, collate_states, infer_vocab_sizes
from music_graph_dfm.diffusion.ctmc import ctmc_sample
from music_graph_dfm.diffusion.edit_flow import derive_oracle_edit_move, editflow_rate_loss, perturb_state_for_editflow
from music_graph_dfm.diffusion.losses import auxiliary_denoising_loss, host_uniqueness_penalty, rate_matching_loss
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


def _template_tables(rhythm_vocab: RhythmTemplateVocab, max_vocab: int) -> tuple[list[int], list[int]]:
    tps = 480
    onset = [0 for _ in range(max_vocab)]
    duration = [1 for _ in range(max_vocab)]
    for idx in range(min(max_vocab, rhythm_vocab.vocab_size)):
        onset[idx] = rhythm_vocab.onset_ticks(idx, tps)
        duration[idx] = rhythm_vocab.duration_ticks_with_semantics(idx, tps)
    return onset, duration


def build_model(vocab_sizes: dict, model_cfg: dict, rhythm_vocab: RhythmTemplateVocab):
    cfg = ModelConfig(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    kind = str(model_cfg.get("kind", "full"))
    onset, duration = _template_tables(rhythm_vocab, max(2, int(vocab_sizes["note.template"])))

    if kind == "baseline":
        return SimpleFactorizedBaseline(vocab_sizes=vocab_sizes, cfg=cfg)
    return FSNTGV2HeteroTransformer(
        vocab_sizes=vocab_sizes,
        cfg=cfg,
        template_onset_ticks=onset,
        template_duration_ticks=duration,
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
        span_attrs = {
            channel: coords[f"span.{channel}"][b, :s].detach().cpu().tolist()
            for channel in ["key", "harm", "meter", "section", "reg_center"]
        }
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
    for coord, radius in [("span.harm", 1), ("note.pitch_token", 2)]:
        v = int(vocab_sizes[coord])
        mat = torch.zeros((v, v), dtype=torch.float32)
        for i in range(v):
            neigh = list(range(max(0, i - radius), min(v, i + radius + 1)))
            p = 1.0 / max(1, len(neigh))
            for j in neigh:
                mat[i, j] = p
        kernels[coord] = mat
    return kernels


def run_training(cfg: dict) -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for training") from exc

    data_root = Path(cfg["data_root"]).expanduser().resolve()
    train_set = FSNTGV2JSONDataset(data_root / "train.jsonl")
    valid_set = FSNTGV2JSONDataset(data_root / "valid.jsonl")
    if len(train_set) == 0:
        raise RuntimeError(f"No training data found in {data_root}")

    rhythm_vocab, _pitch_codec = _load_codecs(data_root)
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

    kernels = _graph_kernels(vocab_sizes, enabled=bool(cfg["diffusion"].get("graph_kernel", {}).get("enabled", False)))
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

    mode = str(cfg["train"].get("mode", "dfm"))
    epochs = int(cfg["train"].get("epochs", 20))
    beta_aux = float(cfg["train"].get("beta_aux", 0.2))
    beta_host = float(cfg["train"].get("beta_host", 0.1))

    rng = random.Random(int(cfg.get("seed", 7)))
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches = 0

        for states in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if mode == "editflow":
                target_states = states
                source_states = [perturb_state_for_editflow(st, vocab_sizes=vocab_sizes, rng=rng) for st in target_states]
                oracle = [derive_oracle_edit_move(src, tgt) for src, tgt in zip(source_states, target_states)]

                batch = _move_to_device(collate_states(source_states), device)
                t = torch.tensor(rng.uniform(0.01, 0.99), device=device)
                edit_outputs = model.forward_edit(batch, t)
                loss = editflow_rate_loss(edit_outputs, oracle)
            else:
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
                loss_host = host_uniqueness_penalty(xt, masks)
                loss = loss_rate + beta_aux * loss_aux + beta_host * loss_host

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
                if mode == "editflow":
                    target_states = states
                    source_states = [perturb_state_for_editflow(st, vocab_sizes=vocab_sizes, rng=rng) for st in target_states]
                    oracle = [derive_oracle_edit_move(src, tgt) for src, tgt in zip(source_states, target_states)]
                    batch = _move_to_device(collate_states(source_states), device)
                    edit_outputs = model.forward_edit(batch, torch.tensor(0.5, device=device))
                    loss = editflow_rate_loss(edit_outputs, oracle)
                else:
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
                    outputs = model(coords_to_batch(batch, xt), torch.tensor(0.5, device=device))
                    masks = coordinate_masks(batch)
                    loss = rate_matching_loss(outputs, xt, x1, xt_is_x0, eta, masks, path_meta)
                valid_loss += float(loss.item())
                vb += 1
        valid_loss /= max(1, vb)

        summary = {"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss, "mode": mode}
        history.append(summary)
        print(summary)

        if (epoch + 1) % int(cfg["train"].get("save_every", 1)) == 0:
            save_checkpoint(
                Path(cfg["train"].get("checkpoint_dir", "artifacts/checkpoints")) / f"epoch_{epoch + 1}.pt",
                model,
                optimizer=optimizer,
                extra={
                    "cfg": cfg,
                    "vocab_sizes": vocab_sizes,
                    "model_cfg": cfg["model"],
                    "mode": mode,
                },
            )

    return {"history": history, "final": history[-1] if history else {}}


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
    return ds, model, vocab_sizes, rhythm_vocab, pitch_codec, dev


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


def generate_samples_from_checkpoint(
    checkpoint: Path,
    data_root: Path,
    split: str = "test",
    num_samples: int = 16,
    num_steps: int = 96,
    device: str = "cpu",
    whole_song_mode: str | None = None,
    whole_song_segments: int = 4,
) -> List[FSNTGV2State]:
    ds, model, vocab_sizes, _rhythm_vocab, _pitch_codec, dev = _load_model_for_sampling(
        checkpoint=checkpoint,
        data_root=data_root,
        split=split,
        device=device,
    )

    out = []
    for i in range(num_samples):
        if whole_song_mode is None:
            ref = ds[i % len(ds)]
            sampled = _sample_state(model, vocab_sizes, ref_state=ref, num_steps=num_steps, device=dev)
            out.append(sampled)
            continue

        if whole_song_mode == "long_context":
            refs = [ds[(i * whole_song_segments + k) % len(ds)] for k in range(whole_song_segments)]
            long_template = build_long_context_template(refs)
            sampled_long = _sample_state(model, vocab_sizes, ref_state=long_template, num_steps=num_steps, device=dev)
            out.append(generate_whole_song([sampled_long], mode="long_context"))
            continue

        if whole_song_mode == "stitching_baseline":
            refs = [ds[(i * whole_song_segments + k) % len(ds)] for k in range(whole_song_segments)]
            segments = [_sample_state(model, vocab_sizes, ref_state=ref, num_steps=num_steps, device=dev) for ref in refs]
            out.append(generate_whole_song(segments, mode="stitching_baseline"))
            continue

        raise ValueError("whole_song_mode must be None, 'long_context', or 'stitching_baseline'")

    return out
