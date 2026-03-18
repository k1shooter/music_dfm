"""State tensor transforms, factorized priors, and forward path sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from music_graph_dfm.diffusion.paths import graph_kernel_sample_tensor, mixture_sample_tensor
from music_graph_dfm.diffusion.schedules import StructureFirstSchedule
from music_graph_dfm.utils.constants import NOTE_CHANNELS, SPAN_CHANNELS


COORD_ORDER = [
    *(f"span.{c}" for c in SPAN_CHANNELS),
    "e_ss.relation",
    "e_ns.template",
    *(f"note.{c}" for c in NOTE_CHANNELS),
]


@dataclass
class PriorConfig:
    active_on_prob: float = 0.15
    e_ns_non_none_prob: float = 0.04
    e_ss_non_none_prob: float = 0.03


def batch_to_coords(batch: dict) -> Dict[str, "torch.Tensor"]:
    return {
        **{f"span.{c}": batch["span"][c] for c in SPAN_CHANNELS},
        **{f"note.{c}": batch["note"][c] for c in NOTE_CHANNELS},
        "e_ns.template": batch["e_ns"],
        "e_ss.relation": batch["e_ss"],
    }


def coords_to_batch(base_batch: dict, coords: Dict[str, "torch.Tensor"]) -> dict:
    out = {
        "span": {c: coords[f"span.{c}"] for c in SPAN_CHANNELS},
        "note": {c: coords[f"note.{c}"] for c in NOTE_CHANNELS},
        "e_ns": coords["e_ns.template"],
        "e_ss": coords["e_ss.relation"],
        "span_mask": base_batch["span_mask"],
        "note_mask": base_batch["note_mask"],
        "meta": base_batch.get("meta", []),
    }
    return out


def sample_factorized_prior(
    batch: dict,
    vocab_sizes: Dict[str, int],
    prior_cfg: PriorConfig,
) -> Dict[str, "torch.Tensor"]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("sample_factorized_prior requires torch") from exc

    coords_x1 = batch_to_coords(batch)
    x0: Dict[str, torch.Tensor] = {}

    for coord, x1 in coords_x1.items():
        vocab = vocab_sizes[coord]
        if coord == "note.active":
            probs = torch.full_like(x1, fill_value=prior_cfg.active_on_prob, dtype=torch.float32)
            x0[coord] = torch.bernoulli(probs).to(torch.long)
        elif coord == "e_ns.template":
            non_none = torch.bernoulli(
                torch.full_like(x1, fill_value=prior_cfg.e_ns_non_none_prob, dtype=torch.float32)
            ).to(torch.long)
            rand_tpl = torch.randint(1, max(2, vocab), size=x1.shape, device=x1.device)
            x0[coord] = non_none * rand_tpl
        elif coord == "e_ss.relation":
            non_none = torch.bernoulli(
                torch.full_like(x1, fill_value=prior_cfg.e_ss_non_none_prob, dtype=torch.float32)
            ).to(torch.long)
            rand_rel = torch.randint(1, max(2, vocab), size=x1.shape, device=x1.device)
            x0[coord] = non_none * rand_rel
        elif coord == "note.pitch_token":
            x0[coord] = torch.randint(0, max(1, vocab), size=x1.shape, device=x1.device)
        else:
            x0[coord] = torch.randint(0, max(1, vocab), size=x1.shape, device=x1.device)

    # Ensure inactive notes have no note-span edges.
    inactive = x0["note.active"] == 0
    x0["e_ns.template"][inactive.unsqueeze(-1).expand_as(x0["e_ns.template"])] = 0
    return x0


def sample_xt_mixture(
    x0: Dict[str, "torch.Tensor"],
    x1: Dict[str, "torch.Tensor"],
    t: float,
    schedule: StructureFirstSchedule,
    path_type: str = "mixture",
    graph_kernels: Dict[str, "torch.Tensor"] | None = None,
) -> Tuple[Dict[str, "torch.Tensor"], Dict[str, "torch.Tensor"], Dict[str, float]]:
    xt = {}
    xt_is_x0 = {}
    eta = {}
    for coord in COORD_ORDER:
        kappa = schedule.kappa(coord, t)
        if (
            path_type == "graph_kernel"
            and graph_kernels is not None
            and coord in graph_kernels
            and coord in {"span.harm", "note.pitch_token"}
        ):
            xt[coord], xt_is_x0[coord] = graph_kernel_sample_tensor(
                x0[coord], x1[coord], kappa, graph_kernels[coord]
            )
        else:
            xt[coord], xt_is_x0[coord] = mixture_sample_tensor(x0[coord], x1[coord], kappa)
        eta[coord] = schedule.eta(coord, t)
    return xt, xt_is_x0, eta
