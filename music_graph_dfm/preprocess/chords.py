"""Chord parsing and harmonic token extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

NOTE_NAME_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

QUALITY_TO_HARM = {
    "maj": 0,
    "min": 1,
    "dim": 2,
    "aug": 3,
    "sus": 4,
    "7": 5,
    "m7": 6,
}


def parse_chord_label(label: str) -> Tuple[int, int]:
    """Returns (root_pc, harm_token)."""
    s = label.strip()
    if not s or s.upper() in {"N", "NC", "NO_CHORD"}:
        return 0, 0

    m = re.match(r"^([A-G](?:#|b)?)(?::|/|\(|\s|$)?(.*)$", s)
    if not m:
        return 0, 0

    root = m.group(1)
    rest = m.group(2).lower()
    root_pc = NOTE_NAME_TO_PC.get(root, 0)

    if "dim" in rest:
        harm = QUALITY_TO_HARM["dim"]
    elif "aug" in rest:
        harm = QUALITY_TO_HARM["aug"]
    elif "sus" in rest:
        harm = QUALITY_TO_HARM["sus"]
    elif "min7" in rest or "m7" in rest:
        harm = QUALITY_TO_HARM["m7"]
    elif "7" in rest:
        harm = QUALITY_TO_HARM["7"]
    elif "min" in rest or rest.startswith("m"):
        harm = QUALITY_TO_HARM["min"]
    else:
        harm = QUALITY_TO_HARM["maj"]
    return root_pc, harm


def load_pop909_chords(song_dir: Path) -> List[Tuple[int, int, int, int]]:
    """Loads chord intervals as (start_tick, end_tick, key_pc, harm_token).

    POP909 layouts vary; this parser supports several tab/space-separated formats.
    """
    candidates = [
        song_dir / "chord_midi.txt",
        song_dir / "chord_audio.txt",
        song_dir / "chord.txt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []

    rows: List[Tuple[int, int, int, int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue

        try:
            start = int(float(parts[0]))
            end = int(float(parts[1]))
            label = parts[2]
        except Exception:
            continue

        key_pc, harm = parse_chord_label(label)
        rows.append((start, end, key_pc, harm))

    return rows
