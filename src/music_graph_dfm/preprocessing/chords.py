"""Chord parsing for POP909 chord files."""

from __future__ import annotations

import re
from collections import Counter
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


def parse_chord_label(label: str) -> int:
    """Extract root pitch class from symbolic chord label."""
    s = label.strip()
    if not s or s.upper() in {"N", "NC", "NO_CHORD"}:
        return 0
    match = re.match(r"^([A-G](?:#|b)?)", s)
    if not match:
        return 0
    return int(NOTE_NAME_TO_PC.get(match.group(1), 0))


def load_pop909_chords(song_dir: Path) -> List[Tuple[int, int, int, int]]:
    """Return intervals as (start_tick, end_tick, key_pc, harmony_root_pc)."""
    candidates = [song_dir / "chord_midi.txt", song_dir / "chord_audio.txt", song_dir / "chord.txt"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []

    raw_rows: List[Tuple[int, int, int]] = []
    roots = Counter()
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
        except Exception:
            continue
        root = parse_chord_label(parts[2])
        raw_rows.append((start, end, root))
        roots[root] += 1

    tonic = roots.most_common(1)[0][0] if roots else 0
    return [(s, e, tonic, root) for s, e, root in raw_rows]
