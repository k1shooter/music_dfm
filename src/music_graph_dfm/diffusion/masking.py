"""Mask construction and hard constraints for padded/inactive coordinates."""

from __future__ import annotations

from typing import Dict

from music_graph_dfm.constants import COORD_ORDER


def pair_masks(batch: dict) -> dict:
    span_mask = batch["span_mask"]
    note_mask = batch["note_mask"]
    span_pair = span_mask.unsqueeze(-1) & span_mask.unsqueeze(-2)
    note_pair = note_mask.unsqueeze(-1) & note_mask.unsqueeze(-2)
    note_span = note_mask.unsqueeze(-1) & span_mask.unsqueeze(-2)
    return {
        "span_pair": span_pair,
        "note_pair": note_pair,
        "note_span": note_span,
    }


def coordinate_masks(batch: dict, coords: Dict[str, "torch.Tensor"] | None = None) -> Dict[str, "torch.Tensor"]:
    del coords
    masks = pair_masks(batch)
    span_mask = batch["span_mask"]
    note_mask = batch["note_mask"]
    out: Dict[str, "torch.Tensor"] = {}
    for coord in COORD_ORDER:
        if coord.startswith("span."):
            out[coord] = span_mask
        elif coord == "e_ss.relation":
            out[coord] = masks["span_pair"]
        else:
            out[coord] = note_mask
    return out


def enforce_state_constraints(coords: Dict[str, "torch.Tensor"], batch: dict) -> Dict[str, "torch.Tensor"]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("enforce_state_constraints requires torch") from exc

    span_mask = batch["span_mask"]
    note_mask = batch["note_mask"]
    masks = coordinate_masks(batch)

    for coord, tensor in coords.items():
        mask = masks[coord]
        coords[coord] = torch.where(mask, tensor, torch.zeros_like(tensor))

    active = coords["note.active"]
    host = coords["note.host"]
    template = coords["note.template"]

    # Clamp hosts to valid span range, deactivate invalid notes.
    span_lengths = span_mask.sum(dim=-1)
    for b in range(host.shape[0]):
        max_host = int(span_lengths[b].item())
        invalid = (host[b] < 0) | (host[b] > max_host)
        host[b][invalid] = 0
        template[b][invalid] = 0
        active[b][invalid] = 0

    inactive_or_pad = (~note_mask) | (active == 0)
    host[inactive_or_pad] = 0
    template[inactive_or_pad] = 0

    coords["note.host"] = host
    coords["note.template"] = template
    coords["note.active"] = active
    return coords
