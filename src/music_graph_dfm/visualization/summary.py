"""Simple visualization/report helpers for generated samples."""

from __future__ import annotations

import json
from pathlib import Path

from music_graph_dfm.representation.state import FSNTGV2State


def summarize_state(state: FSNTGV2State) -> dict:
    return {
        "song_id": state.metadata.get("song_id", "unknown"),
        "num_spans": state.num_spans,
        "num_notes": state.num_notes,
        "active_notes": int(sum(1 for x in state.note_attrs["active"] if int(x) == 1)),
        "whole_song_mode": state.metadata.get("whole_song_mode", "segment"),
    }


def visualize_sample_directory(sample_dir: str | Path, out_path: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    out_path = Path(out_path)
    samples_path = sample_dir / "samples.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    rows = []
    for line in samples_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(FSNTGV2State.from_dict(json.loads(line)))

    summary = {
        "num_samples": len(rows),
        "samples": [summarize_state(s) for s in rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path
