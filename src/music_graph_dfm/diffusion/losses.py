"""Masked losses for coordinate-wise DFM objectives."""

from __future__ import annotations

from typing import Dict

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.diffusion.paths import graph_kernel_target_distribution


def _masked_mean(values, mask):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_masked_mean requires torch") from exc

    num = (values * mask).sum()
    den = mask.sum().clamp(min=1.0)
    return num / den


def _target_distribution(coord: str, x1, path_meta: dict, vocab_size: int):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_target_distribution requires torch") from exc

    path_type = path_meta.get("path_type", "mixture")
    kernels = path_meta.get("graph_kernels", {})
    if path_type == "graph_kernel" and coord in kernels and coord in {"span.harm", "note.pitch_token"}:
        return graph_kernel_target_distribution(x1, kernels[coord])
    x1 = x1.clamp(min=0, max=vocab_size - 1)
    return torch.nn.functional.one_hot(x1, num_classes=vocab_size).to(torch.float32)


def rate_matching_loss(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    x_1: Dict[str, "torch.Tensor"],
    xt_is_x0: Dict[str, "torch.Tensor"],
    eta: Dict[str, float],
    masks: Dict[str, "torch.Tensor"],
    path_meta: dict,
    eps: float = 1e-8,
):
    """Poisson rate matching on off-diagonal rates only."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("rate_matching_loss requires torch") from exc

    total = torch.tensor(0.0, device=next(iter(x_t.values())).device)
    for coord in COORD_ORDER:
        out = outputs[coord]
        lam = torch.nn.functional.softplus(out["lambda"]).squeeze(-1)
        logits = out["logits"]
        pi = torch.softmax(logits, dim=-1)

        xt = x_t[coord]
        x1 = x_1[coord]
        mask = masks[coord].to(torch.float32)
        indicator = xt_is_x0[coord].to(torch.float32)

        vocab = pi.shape[-1]
        current = torch.nn.functional.one_hot(xt.clamp(min=0, max=vocab - 1), num_classes=vocab).to(torch.float32)
        pred_offdiag = lam.unsqueeze(-1) * pi * (1.0 - current)

        target_dist = _target_distribution(coord, x1, path_meta, vocab)
        target_rates = indicator.unsqueeze(-1) * float(eta[coord]) * target_dist * (1.0 - current)

        m = mask.unsqueeze(-1)
        nll = pred_offdiag - target_rates * torch.log(pred_offdiag + eps)
        total = total + _masked_mean(nll.sum(dim=-1), mask)
    return total


def auxiliary_denoising_loss(
    outputs: Dict[str, dict],
    x_1: Dict[str, "torch.Tensor"],
    masks: Dict[str, "torch.Tensor"],
):
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("auxiliary_denoising_loss requires torch") from exc

    losses = []
    for coord in COORD_ORDER:
        logits = outputs[coord]["logits"]
        target = x_1[coord]
        vocab = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1), reduction="none").reshape(target.shape)
        losses.append(_masked_mean(ce, masks[coord].to(torch.float32)))
    return torch.stack(losses).mean()


def host_uniqueness_penalty(coords: Dict[str, "torch.Tensor"], masks: Dict[str, "torch.Tensor"]):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("host_uniqueness_penalty requires torch") from exc

    active = coords["note.active"].to(torch.float32)
    has_host = (coords["note.host"] > 0).to(torch.float32)
    diff = (active - has_host).abs()
    return _masked_mean(diff, masks["note.active"].to(torch.float32))
