"""Dataset and collation utilities for cached FSNTG states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from music_graph_dfm.data.fsntg import FSNTGState
from music_graph_dfm.utils.constants import NOTE_CHANNELS, SPAN_CHANNELS


def load_states_jsonl(path: Path) -> List[FSNTGState]:
    states: List[FSNTGState] = []
    if not path.exists():
        return states
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        states.append(FSNTGState.from_dict(json.loads(line)))
    return states


def infer_vocab_sizes(states: Iterable[FSNTGState]) -> Dict[str, int]:
    maxima: Dict[str, int] = {}
    all_channels = [
        *(f"span.{c}" for c in SPAN_CHANNELS),
        *(f"note.{c}" for c in NOTE_CHANNELS),
        "e_ns.template",
        "e_ss.relation",
    ]
    for c in all_channels:
        maxima[c] = 0

    for st in states:
        for c in SPAN_CHANNELS:
            values = st.span_attrs[c]
            maxima[f"span.{c}"] = max(maxima[f"span.{c}"], max(values, default=0))
        for c in NOTE_CHANNELS:
            values = st.note_attrs[c]
            maxima[f"note.{c}"] = max(maxima[f"note.{c}"], max(values, default=0))

        maxima["e_ns.template"] = max(maxima["e_ns.template"], max((max(r, default=0) for r in st.e_ns), default=0))
        maxima["e_ss.relation"] = max(maxima["e_ss.relation"], max((max(r, default=0) for r in st.e_ss), default=0))

    return {k: v + 1 for k, v in maxima.items()}


class FSNTGJSONDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.states = load_states_jsonl(self.path)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> FSNTGState:
        return self.states[idx]


def collate_fsntg(states: List[FSNTGState]) -> dict:
    """Pads FSNTG states into tensor-style batch dictionaries.

    This function requires torch at runtime.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("collate_fsntg requires torch") from exc

    bsz = len(states)
    max_s = max(st.num_spans for st in states)
    max_n = max(st.num_notes for st in states)

    batch = {
        "span": {c: torch.zeros((bsz, max_s), dtype=torch.long) for c in SPAN_CHANNELS},
        "note": {c: torch.zeros((bsz, max_n), dtype=torch.long) for c in NOTE_CHANNELS},
        "e_ns": torch.zeros((bsz, max_n, max_s), dtype=torch.long),
        "e_ss": torch.zeros((bsz, max_s, max_s), dtype=torch.long),
        "span_mask": torch.zeros((bsz, max_s), dtype=torch.bool),
        "note_mask": torch.zeros((bsz, max_n), dtype=torch.bool),
        "meta": [st.metadata for st in states],
    }

    for b, st in enumerate(states):
        s = st.num_spans
        n = st.num_notes
        batch["span_mask"][b, :s] = True
        batch["note_mask"][b, :n] = True

        for c in SPAN_CHANNELS:
            batch["span"][c][b, :s] = torch.as_tensor(st.span_attrs[c], dtype=torch.long)
        for c in NOTE_CHANNELS:
            batch["note"][c][b, :n] = torch.as_tensor(st.note_attrs[c], dtype=torch.long)

        batch["e_ns"][b, :n, :s] = torch.as_tensor(st.e_ns, dtype=torch.long)
        batch["e_ss"][b, :s, :s] = torch.as_tensor(st.e_ss, dtype=torch.long)

    return batch


def to_note_tuple_baseline_batch(states: List[FSNTGState]) -> dict:
    """Builds flattened note tuples for baseline comparison script."""
    rows = []
    for st in states:
        for i in range(st.num_notes):
            row = [
                int(st.note_attrs["active"][i]),
                int(st.note_attrs["pitch_token"][i]),
                int(st.note_attrs["velocity"][i]),
                int(st.note_attrs["role"][i]),
            ]
            host = 0
            tpl = 0
            for j, e in enumerate(st.e_ns[i]):
                if e != 0:
                    host = j
                    tpl = e
                    break
            row.extend([host, tpl])
            rows.append(row)
    return {"note_tuples": rows}
