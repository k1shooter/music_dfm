"""Shared training runner for FSNTG DFM and EditFlow modes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

from music_graph_dfm.data.dataset import FSNTGJSONDataset, collate_fsntg, infer_vocab_sizes
from music_graph_dfm.diffusion.edit_ops import random_edit_step
from music_graph_dfm.diffusion.losses import (
    auxiliary_denoising_loss,
    music_structure_loss,
    rate_matching_loss,
)
from music_graph_dfm.diffusion.schedules import StructureFirstSchedule
from music_graph_dfm.diffusion.state_ops import (
    PriorConfig,
    batch_to_coords,
    coords_to_batch,
    sample_factorized_prior,
    sample_xt_mixture,
)
from music_graph_dfm.utils.training import build_model, load_codecs, save_checkpoint


def _move_to_device(obj, device):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required") from exc

    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def _make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required") from exc

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x,
    )


def _make_local_transition_kernel(vocab_size: int, radius: int = 1):
    import torch

    k = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    for i in range(vocab_size):
        neigh = [j for j in range(max(0, i - radius), min(vocab_size, i + radius + 1))]
        p = 1.0 / max(1, len(neigh))
        for j in neigh:
            k[i, j] = p
    return k


def run_training(cfg: dict, use_edit_ops: bool = False) -> Dict[str, float]:
    try:
        import torch
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        raise RuntimeError("torch (+ tensorboard) is required for training") from exc

    data_root = Path(cfg["data"]["cache_root"]) 
    train_set = FSNTGJSONDataset(data_root / "train.jsonl")
    valid_set = FSNTGJSONDataset(data_root / "valid.jsonl")
    if len(train_set) == 0:
        raise RuntimeError(f"No training data found at {data_root / 'train.jsonl'}")

    vocab_sizes = infer_vocab_sizes(train_set.states)

    model = build_model(vocab_sizes=vocab_sizes, cfg=cfg["model"])
    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    schedule = StructureFirstSchedule(**cfg["diffusion"]["schedule"])
    prior_cfg = PriorConfig(**cfg["diffusion"]["prior"])
    path_type = str(cfg["diffusion"].get("path_type", "mixture"))

    train_loader = _make_dataloader(
        train_set,
        batch_size=int(cfg["data"]["loader"]["batch_size"]),
        shuffle=bool(cfg["data"]["loader"]["shuffle"]),
        num_workers=int(cfg.get("num_workers", 0)),
    )
    valid_loader = _make_dataloader(
        valid_set,
        batch_size=int(cfg["data"]["loader"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    _, pitch_codec = load_codecs(data_root)
    compat_table = torch.tensor(pitch_codec.compatibility_table(), dtype=torch.float32, device=device)
    graph_kernels = None
    if path_type == "graph_kernel" and cfg["diffusion"].get("use_graph_kernel_for_harm_pitch", False):
        graph_kernels = {
            "span.harm": _make_local_transition_kernel(vocab_sizes["span.harm"], radius=1).to(device),
            "note.pitch_token": _make_local_transition_kernel(vocab_sizes["note.pitch_token"], radius=2).to(
                device
            ),
        }

    writer = None
    if cfg["train"].get("tensorboard", {}).get("enabled", False):
        writer = SummaryWriter(log_dir=cfg["train"]["tensorboard"].get("log_dir", "tb"))

    global_step = 0
    metrics = {}
    max_epochs = int(cfg["train"]["max_epochs"])
    beta = float(cfg["model"]["loss"]["beta_aux"])
    gamma = float(cfg["model"]["loss"]["gamma_music"])

    for epoch in range(max_epochs):
        model.train()
        train_acc = {"loss": 0.0, "rate": 0.0, "aux": 0.0, "music": 0.0}
        num_batches = 0

        for states in train_loader:
            if use_edit_ops:
                p_insert = float(cfg["train"].get("edit", {}).get("p_insert", 0.2))
                p_delete = float(cfg["train"].get("edit", {}).get("p_delete", 0.2))
                states = [random_edit_step(st, vocab_sizes, p_insert=p_insert, p_delete=p_delete) for st in states]

            batch = _move_to_device(collate_fsntg(states), device)
            x1 = batch_to_coords(batch)

            t = random.uniform(float(cfg["diffusion"]["train_time"]["t_min"]), float(cfg["diffusion"]["train_time"]["t_max"]))
            x0 = sample_factorized_prior(batch, vocab_sizes=vocab_sizes, prior_cfg=prior_cfg)
            xt, xt_is_x0, eta = sample_xt_mixture(
                x0,
                x1,
                t=t,
                schedule=schedule,
                path_type=path_type,
                graph_kernels=graph_kernels,
            )
            batch_xt = coords_to_batch(batch, xt)

            outputs = model(batch_xt, torch.tensor(t, device=device))

            loss_rate = rate_matching_loss(outputs, xt, x1, xt_is_x0, eta)
            loss_aux = auxiliary_denoising_loss(outputs, x1)
            mreg = music_structure_loss(outputs, xt, compat_table if cfg["model"]["loss"].get("compat_loss", True) else None)
            loss_music = mreg["total"]
            loss = loss_rate + beta * loss_aux + gamma * loss_music

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_acc["loss"] += float(loss.item())
            train_acc["rate"] += float(loss_rate.item())
            train_acc["aux"] += float(loss_aux.item())
            train_acc["music"] += float(loss_music.item())
            num_batches += 1
            global_step += 1

            if writer and global_step % int(cfg["train"]["log_every"]) == 0:
                writer.add_scalar("train/loss", float(loss.item()), global_step)
                writer.add_scalar("train/loss_rate", float(loss_rate.item()), global_step)
                writer.add_scalar("train/loss_aux", float(loss_aux.item()), global_step)
                writer.add_scalar("train/loss_music", float(loss_music.item()), global_step)

        for k in train_acc:
            train_acc[k] /= max(1, num_batches)

        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for states in valid_loader:
                batch = _move_to_device(collate_fsntg(states), device)
                x1 = batch_to_coords(batch)
                t = 0.5
                x0 = sample_factorized_prior(batch, vocab_sizes=vocab_sizes, prior_cfg=prior_cfg)
                xt, xt_is_x0, eta = sample_xt_mixture(
                    x0,
                    x1,
                    t=t,
                    schedule=schedule,
                    path_type=path_type,
                    graph_kernels=graph_kernels,
                )
                batch_xt = coords_to_batch(batch, xt)
                outputs = model(batch_xt, torch.tensor(t, device=device))
                loss_rate = rate_matching_loss(outputs, xt, x1, xt_is_x0, eta)
                loss_aux = auxiliary_denoising_loss(outputs, x1)
                mreg = music_structure_loss(outputs, xt, compat_table)
                loss = loss_rate + beta * loss_aux + gamma * mreg["total"]
                valid_loss += float(loss.item())
                valid_batches += 1
        valid_loss /= max(1, valid_batches)

        if writer:
            writer.add_scalar("valid/loss", valid_loss, epoch)

        metrics = {
            "epoch": epoch,
            "train_loss": train_acc["loss"],
            "valid_loss": valid_loss,
            "train_rate": train_acc["rate"],
            "train_aux": train_acc["aux"],
            "train_music": train_acc["music"],
        }

        if (epoch + 1) % int(cfg["train"]["save_every"]) == 0:
            save_checkpoint(
                Path("checkpoints") / f"epoch_{epoch + 1}.pt",
                model,
                optimizer=optimizer,
                step=global_step,
                extra={"metrics": metrics, "vocab_sizes": vocab_sizes, "cfg": cfg},
            )

        print(json.dumps(metrics))

    if writer:
        writer.close()

    return metrics
