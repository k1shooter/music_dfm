"""Conversion helpers between tensor coordinate dictionaries and FSNTGState objects."""

from __future__ import annotations

from typing import Dict, List

from music_graph_dfm.data.fsntg import FSNTGState
from music_graph_dfm.utils.constants import NOTE_CHANNELS, SPAN_CHANNELS


def coords_to_states(coords: Dict[str, "torch.Tensor"], base_batch: dict) -> List[FSNTGState]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("coords_to_states requires torch") from exc

    span_mask = base_batch["span_mask"]
    note_mask = base_batch["note_mask"]
    bsz = span_mask.shape[0]
    states: List[FSNTGState] = []

    for b in range(bsz):
        s = int(span_mask[b].sum().item())
        n = int(note_mask[b].sum().item())

        span_attrs = {c: coords[f"span.{c}"][b, :s].detach().cpu().tolist() for c in SPAN_CHANNELS}
        note_attrs = {c: coords[f"note.{c}"][b, :n].detach().cpu().tolist() for c in NOTE_CHANNELS}
        e_ns = coords["e_ns.template"][b, :n, :s].detach().cpu().tolist()
        e_ss = coords["e_ss.relation"][b, :s, :s].detach().cpu().tolist()

        ticks_per_span = 1920
        span_starts = [j * ticks_per_span for j in range(s)]
        meta = {}
        if "meta" in base_batch and b < len(base_batch["meta"]):
            meta = dict(base_batch["meta"][b])

        states.append(
            FSNTGState(
                span_attrs=span_attrs,
                note_attrs=note_attrs,
                e_ns=e_ns,
                e_ss=e_ss,
                span_starts=span_starts,
                ticks_per_span=ticks_per_span,
                metadata=meta,
            )
        )
    return states


def clone_coords(coords: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
    return {k: v.clone() for k, v in coords.items()}
