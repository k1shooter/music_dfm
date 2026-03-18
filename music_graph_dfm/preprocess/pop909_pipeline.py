"""POP909 preprocessing into FSNTG states with data-driven rhythm template vocabulary."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from music_graph_dfm.data.fsntg import FSNTGState
from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.preprocess.chords import load_pop909_chords
from music_graph_dfm.preprocess.section_labels import derive_repeat_variation_edges, derive_section_labels
from music_graph_dfm.templates.rhythm_templates import (
    RhythmTemplateVocab,
    quantize_duration_to_class,
    quantize_onset_to_bin,
)
from music_graph_dfm.utils.constants import E_NS_NONE, E_SS_NONE, SPAN_RELATIONS


@dataclass
class RawNoteEvent:
    onset_tick: int
    end_tick: int
    pitch: int
    velocity: int
    role: int


def download_pop909(target_dir: Path, force: bool = False) -> Path:
    """Downloads POP909 via git clone if needed."""
    repo = "https://github.com/music-x-lab/POP909-Dataset"
    target_dir = target_dir.expanduser().resolve()
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        return target_dir
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", repo, str(target_dir)])
    return target_dir


def _find_midi_path(song_dir: Path) -> Path | None:
    candidates = [song_dir / "midi.mid", song_dir / "MIDI.mid", song_dir / "melody.mid"]
    for c in candidates:
        if c.exists():
            return c
    all_mid = sorted(song_dir.glob("*.mid"))
    return all_mid[0] if all_mid else None


def load_note_events(midi_path: Path) -> Tuple[List[RawNoteEvent], int, int]:
    """Loads note events with optional role from MIDI tracks."""
    try:
        import miditoolkit
    except Exception as exc:
        raise RuntimeError("miditoolkit is required for MIDI parsing") from exc

    midi = miditoolkit.MidiFile(str(midi_path))
    ticks_per_beat = max(1, int(midi.ticks_per_beat))
    beats_per_bar = 4
    if midi.time_signature_changes:
        beats_per_bar = int(midi.time_signature_changes[0].numerator)

    events: List[RawNoteEvent] = []
    for role, inst in enumerate(midi.instruments):
        for n in inst.notes:
            end_tick = max(int(n.end), int(n.start) + 1)
            events.append(
                RawNoteEvent(
                    onset_tick=int(n.start),
                    end_tick=end_tick,
                    pitch=int(n.pitch),
                    velocity=int(n.velocity),
                    role=role,
                )
            )
    events.sort(key=lambda x: (x.onset_tick, x.pitch, x.end_tick))
    return events, ticks_per_beat, beats_per_bar


def _default_chord_for_span(span_start: int, ticks_per_span: int, chord_rows: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    for s, e, key, harm in chord_rows:
        if s <= span_start < e:
            return key, harm
    return 0, 0


def _safe_reg_center(note_pitches: List[int]) -> int:
    if not note_pitches:
        return 4
    median = sorted(note_pitches)[len(note_pitches) // 2]
    return max(0, min(15, (median - 24) // 6))


def extract_template_records(
    songs: Sequence[Tuple[Path, List[RawNoteEvent], int, int]],
    onset_bins: int,
) -> List[Tuple[int, int, int, int, int]]:
    records: List[Tuple[int, int, int, int, int]] = []
    for _song_dir, events, tpq, beats_per_bar in songs:
        ticks_per_span = tpq * beats_per_bar
        meter_cls = 0 if beats_per_bar == 4 else 1
        duration_table = [tpq // 4, tpq // 2, (tpq * 3) // 4, tpq, (tpq * 3) // 2, tpq * 2, tpq * 3, tpq * 4]
        duration_table = [max(1, x) for x in duration_table]
        for e in events:
            host = e.onset_tick // max(1, ticks_per_span)
            local = e.onset_tick - host * ticks_per_span
            dur = e.end_tick - e.onset_tick
            onset_bin = quantize_onset_to_bin(local, ticks_per_span, onset_bins)
            dur_cls = quantize_duration_to_class(dur, duration_table)
            tie = int(e.end_tick > (host + 1) * ticks_per_span)
            ext = int(min(2, max(0, (e.end_tick - ((host + 1) * ticks_per_span)) // max(1, ticks_per_span))))
            records.append((meter_cls, onset_bin, dur_cls, tie, ext))
    return records


def build_fsntg_state(
    song_id: str,
    events: List[RawNoteEvent],
    ticks_per_beat: int,
    beats_per_bar: int,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    chord_rows: List[Tuple[int, int, int, int]] | None = None,
    include_nonlocal_span_edges: bool = True,
) -> FSNTGState:
    chord_rows = chord_rows or []
    ticks_per_span = ticks_per_beat * beats_per_bar
    max_tick = max([e.end_tick for e in events], default=ticks_per_span)
    num_spans = max(1, (max_tick + ticks_per_span - 1) // ticks_per_span)

    span_starts = [j * ticks_per_span for j in range(num_spans)]
    span_keys = []
    span_harm = []
    span_meter = []
    span_reg_center = []

    for j, start in enumerate(span_starts):
        key, harm = _default_chord_for_span(start, ticks_per_span, chord_rows)
        span_keys.append(int(key))
        span_harm.append(int(harm))
        span_meter.append(0 if beats_per_bar == 4 else 1)
        span_notes = [ev.pitch for ev in events if ev.onset_tick // ticks_per_span == j]
        span_reg_center.append(_safe_reg_center(span_notes))

    span_section = derive_section_labels(num_spans)

    n = len(events)
    active = [1 for _ in range(n)]
    pitch_tokens = [0 for _ in range(n)]
    velocity_bins = [0 for _ in range(n)]
    roles = [0 for _ in range(n)]
    e_ns = [[E_NS_NONE for _ in range(num_spans)] for _ in range(n)]

    duration_table = rhythm_vocab.duration_tick_values
    for i, ev in enumerate(events):
        host = max(0, min(num_spans - 1, ev.onset_tick // ticks_per_span))
        local = ev.onset_tick - host * ticks_per_span
        dur = max(1, ev.end_tick - ev.onset_tick)
        onset_bin = quantize_onset_to_bin(local, ticks_per_span, rhythm_vocab.max_onset_bins)
        dur_cls = quantize_duration_to_class(dur, duration_table)
        tie = int(ev.end_tick > (host + 1) * ticks_per_span)
        ext = int(min(2, max(0, (ev.end_tick - ((host + 1) * ticks_per_span)) // max(1, ticks_per_span))))
        tpl = rhythm_vocab.encode(span_meter[host], onset_bin, dur_cls, tie, ext)
        e_ns[i][host] = int(tpl)

        degree = (ev.pitch - span_keys[host]) % 12
        reg_offset = int(round((ev.pitch - (36 + span_reg_center[host] * 6)) / 12))
        reg_offset = max(min(pitch_codec.register_offsets), min(max(pitch_codec.register_offsets), reg_offset))
        pitch_tokens[i] = pitch_codec.encode(degree, reg_offset)
        velocity_bins[i] = max(0, min(15, ev.velocity // 8))
        roles[i] = int(ev.role)

    e_ss = [[E_SS_NONE for _ in range(num_spans)] for _ in range(num_spans)]
    for j in range(num_spans - 1):
        e_ss[j][j + 1] = SPAN_RELATIONS.index("next")

    if include_nonlocal_span_edges:
        for src, dst, rel in derive_repeat_variation_edges(span_harm):
            if 0 <= src < num_spans and 0 <= dst < num_spans:
                e_ss[src][dst] = rel

    return FSNTGState(
        span_attrs={
            "key": span_keys,
            "harm": span_harm,
            "meter": span_meter,
            "section": span_section,
            "reg_center": span_reg_center,
        },
        note_attrs={
            "active": active,
            "pitch_token": pitch_tokens,
            "velocity": velocity_bins,
            "role": roles,
        },
        e_ns=e_ns,
        e_ss=e_ss,
        span_starts=span_starts,
        ticks_per_span=ticks_per_span,
        metadata={"song_id": song_id},
    )


def split_indices(n: int, train_ratio: float, valid_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = idx[:n_train]
    valid = idx[n_train : n_train + n_valid]
    test = idx[n_train + n_valid :]
    return train, valid, test


def save_state_list(path: Path, states: Iterable[FSNTGState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for state in states:
            f.write(json.dumps(state.to_dict()) + "\n")


def preprocess_pop909(
    data_root: Path,
    output_root: Path,
    top_k_per_meter: int = 32,
    onset_bins: int = 32,
    split_seed: int = 7,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    include_nonlocal_span_edges: bool = True,
) -> dict:
    data_root = data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()

    song_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    songs_raw: List[Tuple[Path, List[RawNoteEvent], int, int]] = []
    for song_dir in song_dirs:
        midi_path = _find_midi_path(song_dir)
        if midi_path is None:
            continue
        try:
            events, tpq, beats_per_bar = load_note_events(midi_path)
        except Exception:
            continue
        if not events:
            continue
        songs_raw.append((song_dir, events, tpq, beats_per_bar))

    records = extract_template_records(songs_raw, onset_bins=onset_bins)
    rhythm_vocab = RhythmTemplateVocab(top_k_per_meter=top_k_per_meter, max_onset_bins=onset_bins)
    rhythm_vocab.fit(records)
    pitch_codec = PitchTokenCodec()

    states: List[FSNTGState] = []
    for song_dir, events, tpq, beats_per_bar in songs_raw:
        chords = load_pop909_chords(song_dir)
        state = build_fsntg_state(
            song_id=song_dir.name,
            events=events,
            ticks_per_beat=tpq,
            beats_per_bar=beats_per_bar,
            rhythm_vocab=rhythm_vocab,
            pitch_codec=pitch_codec,
            chord_rows=chords,
            include_nonlocal_span_edges=include_nonlocal_span_edges,
        )
        states.append(state)

    train_idx, valid_idx, test_idx = split_indices(len(states), train_ratio, valid_ratio, split_seed)
    train_states = [states[i] for i in train_idx]
    valid_states = [states[i] for i in valid_idx]
    test_states = [states[i] for i in test_idx]

    output_root.mkdir(parents=True, exist_ok=True)
    save_state_list(output_root / "train.jsonl", train_states)
    save_state_list(output_root / "valid.jsonl", valid_states)
    save_state_list(output_root / "test.jsonl", test_states)

    (output_root / "rhythm_templates.json").write_text(json.dumps(rhythm_vocab.to_dict(), indent=2), encoding="utf-8")
    (output_root / "pitch_codec.json").write_text(json.dumps(pitch_codec.to_dict(), indent=2), encoding="utf-8")

    stats = {
        "num_songs": len(states),
        "num_train": len(train_states),
        "num_valid": len(valid_states),
        "num_test": len(test_states),
        "rhythm_vocab_size": rhythm_vocab.vocab_size,
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
