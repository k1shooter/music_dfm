"""Coordinate extraction, priors, and forward path sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from music_graph_dfm.constants import COORD_ORDER, NOTE_CHANNELS, SPAN_CHANNELS
from music_graph_dfm.diffusion.masking import enforce_state_constraints
from music_graph_dfm.diffusion.paths import graph_kernel_sample_tensor, mixture_sample_tensor


@dataclass
class PriorConfig:
    active_on_prob: float = 0.18
    template_on_prob: float = 0.22
    e_ss_non_none_prob: float = 0.05


def batch_to_coords(batch: dict) -> Dict[str, "torch.Tensor"]:
    return {
        **{f"span.{channel}": batch["span"][channel] for channel in SPAN_CHANNELS},
        **{f"note.{channel}": batch["note"][channel] for channel in NOTE_CHANNELS},
        "note.host": batch["host"],
        "note.template": batch["template"],
        "e_ss.relation": batch["e_ss"],
    }


def coords_to_batch(base_batch: dict, coords: Dict[str, "torch.Tensor"]) -> dict:
    out = {
        "span": {channel: coords[f"span.{channel}"] for channel in SPAN_CHANNELS},
        "note": {channel: coords[f"note.{channel}"] for channel in NOTE_CHANNELS},
        "host": coords["note.host"],
        "template": coords["note.template"],
        "e_ss": coords["e_ss.relation"],
        "span_mask": base_batch["span_mask"],
        "note_mask": base_batch["note_mask"],
        "ticks_per_span": base_batch["ticks_per_span"],
        "meta": base_batch.get("meta", []),
    }
    return out


def sample_prior(batch: dict, vocab_sizes: Dict[str, int], cfg: PriorConfig) -> Dict[str, "torch.Tensor"]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sample_prior requires torch") from exc

    span_mask = batch["span_mask"]
    note_mask = batch["note_mask"]
    bsz, max_spans = span_mask.shape
    _, max_notes = note_mask.shape

    coords: Dict[str, torch.Tensor] = {}

    for channel in SPAN_CHANNELS:
        coord = f"span.{channel}"
        vocab = max(1, int(vocab_sizes[coord]))
        coords[coord] = torch.randint(0, vocab, size=(bsz, max_spans), device=span_mask.device)

    vocab_rel = max(1, int(vocab_sizes["e_ss.relation"]))
    non_none = torch.bernoulli(
        torch.full((bsz, max_spans, max_spans), fill_value=float(cfg.e_ss_non_none_prob), device=span_mask.device)
    ).to(torch.long)
    relation = torch.randint(1, max(2, vocab_rel), size=(bsz, max_spans, max_spans), device=span_mask.device)
    coords["e_ss.relation"] = non_none * relation

    active = torch.bernoulli(
        torch.full((bsz, max_notes), fill_value=float(cfg.active_on_prob), device=note_mask.device)
    ).to(torch.long)
    coords["note.active"] = active

    for channel in ["pitch_token", "velocity", "role"]:
        coord = f"note.{channel}"
        vocab = max(1, int(vocab_sizes[coord]))
        coords[coord] = torch.randint(0, vocab, size=(bsz, max_notes), device=note_mask.device)

    template_vocab = max(1, int(vocab_sizes["note.template"]))
    host_vocab = max(1, int(vocab_sizes["note.host"]))

    host = torch.zeros((bsz, max_notes), dtype=torch.long, device=note_mask.device)
    template = torch.zeros((bsz, max_notes), dtype=torch.long, device=note_mask.device)
    for b in range(bsz):
        span_count = int(span_mask[b].sum().item())
        if span_count <= 0:
            continue
        for n in range(max_notes):
            if int(active[b, n].item()) == 0:
                continue
            if int(note_mask[b, n].item()) == 0:
                continue
            host[b, n] = int(torch.randint(1, min(span_count + 1, host_vocab), size=(1,), device=host.device).item())
            is_template_on = bool(torch.bernoulli(torch.tensor(cfg.template_on_prob, device=host.device)).item())
            if is_template_on:
                template[b, n] = int(torch.randint(1, max(2, template_vocab), size=(1,), device=host.device).item())
            else:
                active[b, n] = 0

    coords["note.host"] = host
    coords["note.template"] = template

    coords = enforce_state_constraints(coords, batch)
    return coords


def sample_forward_path(
    x0: Dict[str, "torch.Tensor"],
    x1: Dict[str, "torch.Tensor"],
    t: float,
    schedule,
    path_type: str = "mixture",
    graph_kernels: Dict[str, "torch.Tensor"] | None = None,
):
    xt: Dict[str, "torch.Tensor"] = {}
    xt_is_x0: Dict[str, "torch.Tensor"] = {}
    eta: Dict[str, float] = {}

    for coord in COORD_ORDER:
        kappa = schedule.kappa(coord, t)
        use_graph_kernel = (
            path_type == "graph_kernel"
            and graph_kernels is not None
            and coord in graph_kernels
            and coord in {"span.harm", "note.pitch_token"}
        )
        if use_graph_kernel:
            xt[coord], xt_is_x0[coord] = graph_kernel_sample_tensor(x0[coord], x1[coord], kappa, graph_kernels[coord])
        else:
            xt[coord], xt_is_x0[coord] = mixture_sample_tensor(x0[coord], x1[coord], kappa)
        eta[coord] = schedule.eta(coord, t)

    meta = {
        "path_type": path_type,
        "graph_kernels": graph_kernels or {},
        "graph_kernel_is_approximate": path_type == "graph_kernel",
    }
    return xt, xt_is_x0, eta, meta
