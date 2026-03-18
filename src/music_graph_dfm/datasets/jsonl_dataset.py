"""JSONL dataset and tensor collation for FSNTG-v2 states."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from music_graph_dfm.constants import COORD_ORDER, NOTE_CHANNELS, SPAN_CHANNELS
from music_graph_dfm.representation.state import FSNTGV2State
from music_graph_dfm.utils.io import read_jsonl


class FSNTGV2JSONDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.states = [FSNTGV2State.from_dict(row) for row in read_jsonl(self.path)]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> FSNTGV2State:
        return self.states[index]


def infer_vocab_sizes(states: Iterable[FSNTGV2State]) -> Dict[str, int]:
    maxima = {coord: 0 for coord in COORD_ORDER}
    for st in states:
        for channel in SPAN_CHANNELS:
            maxima[f"span.{channel}"] = max(maxima[f"span.{channel}"], max(st.span_attrs[channel], default=0))
        for channel in NOTE_CHANNELS:
            maxima[f"note.{channel}"] = max(maxima[f"note.{channel}"], max(st.note_attrs[channel], default=0))
        maxima["note.host"] = max(maxima["note.host"], max(st.host, default=0))
        maxima["note.template"] = max(maxima["note.template"], max(st.template, default=0))
        maxima["e_ss.relation"] = max(maxima["e_ss.relation"], max((max(r, default=0) for r in st.e_ss), default=0))
    return {coord: value + 1 for coord, value in maxima.items()}


def collate_states(states: List[FSNTGV2State]) -> dict:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("collate_states requires torch") from exc

    bsz = len(states)
    max_spans = max(st.num_spans for st in states)
    max_notes = max(st.num_notes for st in states)

    batch = {
        "span": {channel: torch.zeros((bsz, max_spans), dtype=torch.long) for channel in SPAN_CHANNELS},
        "note": {channel: torch.zeros((bsz, max_notes), dtype=torch.long) for channel in NOTE_CHANNELS},
        "host": torch.zeros((bsz, max_notes), dtype=torch.long),
        "template": torch.zeros((bsz, max_notes), dtype=torch.long),
        "e_ss": torch.zeros((bsz, max_spans, max_spans), dtype=torch.long),
        "span_mask": torch.zeros((bsz, max_spans), dtype=torch.bool),
        "note_mask": torch.zeros((bsz, max_notes), dtype=torch.bool),
        "ticks_per_span": torch.zeros((bsz,), dtype=torch.long),
        "meta": [st.metadata for st in states],
    }

    for b, st in enumerate(states):
        s = st.num_spans
        n = st.num_notes
        batch["span_mask"][b, :s] = True
        batch["note_mask"][b, :n] = True
        batch["ticks_per_span"][b] = int(st.ticks_per_span)

        for channel in SPAN_CHANNELS:
            batch["span"][channel][b, :s] = torch.as_tensor(st.span_attrs[channel], dtype=torch.long)
        for channel in NOTE_CHANNELS:
            batch["note"][channel][b, :n] = torch.as_tensor(st.note_attrs[channel], dtype=torch.long)

        batch["host"][b, :n] = torch.as_tensor(st.host, dtype=torch.long)
        batch["template"][b, :n] = torch.as_tensor(st.template, dtype=torch.long)
        batch["e_ss"][b, :s, :s] = torch.as_tensor(st.e_ss, dtype=torch.long)

    return batch
