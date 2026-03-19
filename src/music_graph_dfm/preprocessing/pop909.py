"""POP909 preprocessing pipeline for FSNTG-v2 caches."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from music_graph_dfm.constants import CACHE_SCHEMA_VERSION, DEFAULT_TICKS_PER_BEAT
from music_graph_dfm.preprocessing.chords import load_pop909_chords
from music_graph_dfm.preprocessing.structure import derive_section_labels, derive_span_relation_matrix
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import (
    RhythmTemplateVocab,
    quantize_duration_class,
    quantize_onset_bin,
)
from music_graph_dfm.representation.state import FSNTGV2State
from music_graph_dfm.utils.io import save_json, write_jsonl


@dataclass
class RawNoteEvent:
    onset_tick: int
    end_tick: int
    pitch: int
    velocity: int
    role: int


@dataclass
class PreprocessConfig:
    raw_root: str
    output_root: str
    span_resolution: str = "beat"  # beat | half_bar | bar
    top_k_per_meter: int = 64
    onset_bins: int = 8
    max_extension_class: int = 4
    split_seed: int = 7
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    min_notes_per_song: int = 16


def _find_midi(song_dir: Path) -> Path | None:
    candidates = [song_dir / "midi.mid", song_dir / "MIDI.mid", song_dir / "melody.mid"]
    for c in candidates:
        if c.exists():
            return c
    mids = sorted(song_dir.glob("*.mid"))
    return mids[0] if mids else None


def _load_note_events(midi_path: Path) -> tuple[List[RawNoteEvent], int, int, int]:
    try:
        import miditoolkit
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("miditoolkit is required for POP909 preprocessing") from exc

    midi = miditoolkit.MidiFile(str(midi_path))
    tpq = max(1, int(getattr(midi, "ticks_per_beat", DEFAULT_TICKS_PER_BEAT)))
    beats_per_bar = 4
    beat_unit = 4
    if midi.time_signature_changes:
        beats_per_bar = int(midi.time_signature_changes[0].numerator)
        beat_unit = int(midi.time_signature_changes[0].denominator)

    events: List[RawNoteEvent] = []
    for role, inst in enumerate(midi.instruments):
        for note in inst.notes:
            end_tick = max(int(note.start) + 1, int(note.end))
            events.append(
                RawNoteEvent(
                    onset_tick=int(note.start),
                    end_tick=end_tick,
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                    role=int(role),
                )
            )
    events.sort(key=lambda e: (e.onset_tick, e.pitch, e.end_tick))
    return events, tpq, beats_per_bar, beat_unit


def _ticks_per_span(span_resolution: str, ticks_per_beat: int, beats_per_bar: int) -> int:
    if span_resolution == "beat":
        return ticks_per_beat
    if span_resolution == "half_bar":
        return max(1, (ticks_per_beat * beats_per_bar) // 2)
    if span_resolution == "bar":
        return ticks_per_beat * beats_per_bar
    raise ValueError(f"Unsupported span resolution: {span_resolution}")


def _span_chord(
    span_start: int,
    chord_rows: Sequence[Tuple[int, int, int, int, int]],
    fallback_key: int,
) -> tuple[int, int, int]:
    for start, end, key, harm_root, harm_quality in chord_rows:
        if start <= span_start < end:
            return int(key), int(harm_root), int(harm_quality)
    return int(fallback_key), int(fallback_key), 0


def _safe_reg_center(pitches: List[int]) -> int:
    if not pitches:
        return 4
    p = sorted(pitches)[len(pitches) // 2]
    return max(0, min(15, int(round((p - 48) / 4.0))))


def _derive_harm_function(key: int, harm_root: int, harm_quality: int) -> int:
    """Lightweight POP909 harmonic-function heuristic.

    Labels:
    - 0 unknown
    - 1 tonic
    - 2 predominant
    - 3 dominant
    """
    degree = (int(harm_root) - int(key)) % 12
    quality = int(harm_quality)
    if quality == 3 or degree in {7, 11}:
        return 3
    if degree in {2, 5, 9}:
        return 2
    if degree in {0, 3, 4, 8}:
        return 1
    return 0


def _collect_template_records(
    songs: Sequence[Tuple[Path, List[RawNoteEvent], int, int, int]],
    cfg: PreprocessConfig,
) -> list[tuple[int, int, int, int, int]]:
    records = []
    temp = RhythmTemplateVocab(
        top_k_per_meter=cfg.top_k_per_meter,
        onset_bins=cfg.onset_bins,
        max_extension_class=cfg.max_extension_class,
    )
    for _song_dir, events, tpq, beats_per_bar, beat_unit in songs:
        tps = _ticks_per_span(cfg.span_resolution, tpq, beats_per_bar)
        meter = beats_per_bar * 16 + beat_unit
        for event in events:
            span_idx = event.onset_tick // tps
            local = event.onset_tick - span_idx * tps
            onset_bin = quantize_onset_bin(local, tps, cfg.onset_bins)
            duration = max(1, event.end_tick - event.onset_tick)
            dur_class = quantize_duration_class(duration, temp.duration_ticks)
            span_end = (span_idx + 1) * tps
            tie = int(event.end_tick > span_end)
            extension = max(0, event.end_tick - span_end) // max(1, tps)
            extension = min(cfg.max_extension_class, extension)
            records.append((meter, onset_bin, dur_class, tie, extension))
    return records


def _build_state(
    song_id: str,
    events: List[RawNoteEvent],
    tpq: int,
    beats_per_bar: int,
    beat_unit: int,
    cfg: PreprocessConfig,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    chord_rows: Sequence[Tuple[int, int, int, int, int]],
) -> FSNTGV2State:
    tps = _ticks_per_span(cfg.span_resolution, tpq, beats_per_bar)
    max_tick = max((e.end_tick for e in events), default=tps)
    num_spans = max(1, (max_tick + tps - 1) // tps)
    span_starts = [idx * tps for idx in range(num_spans)]

    tonic = chord_rows[0][2] if chord_rows else 0
    key = []
    harm_root = []
    harm_quality = []
    harm_function = []
    meter = []
    reg_center = []

    for j, start in enumerate(span_starts):
        span_key, span_harm_root, span_harm_quality = _span_chord(start, chord_rows, tonic)
        key.append(span_key)
        harm_root.append(span_harm_root)
        harm_quality.append(span_harm_quality)
        harm_function.append(_derive_harm_function(span_key, span_harm_root, span_harm_quality))
        meter.append(beats_per_bar * 16 + beat_unit)
        pitches = [e.pitch for e in events if e.onset_tick // tps == j]
        reg_center.append(_safe_reg_center(pitches))

    section = derive_section_labels(num_spans)
    e_ss = derive_span_relation_matrix(harm_root, section)

    active: List[int] = []
    pitch_tokens: List[int] = []
    velocity: List[int] = []
    role: List[int] = []
    host: List[int] = []
    template: List[int] = []

    for event in events:
        span_idx = max(0, min(num_spans - 1, event.onset_tick // tps))
        local = event.onset_tick - span_idx * tps
        onset_bin = quantize_onset_bin(local, tps, cfg.onset_bins)
        duration = max(1, event.end_tick - event.onset_tick)
        dur_class = quantize_duration_class(duration, rhythm_vocab.duration_ticks)
        span_end = (span_idx + 1) * tps
        tie = int(event.end_tick > span_end)
        extension = max(0, event.end_tick - span_end) // max(1, tps)
        extension = min(cfg.max_extension_class, extension)

        meter_class = beats_per_bar * 16 + beat_unit
        template_id = rhythm_vocab.encode(meter_class, onset_bin, dur_class, tie, extension)
        token = pitch_codec.encode_pitch_token(
            abs_pitch=event.pitch,
            host_span_state={
                "key": key[span_idx],
                "harm_root": harm_root[span_idx],
                "harm_quality": harm_quality[span_idx],
                "harm_function": harm_function[span_idx],
                "reg_center": reg_center[span_idx],
            },
        )

        active.append(1)
        pitch_tokens.append(token)
        velocity.append(max(0, min(15, event.velocity // 8)))
        role.append(int(event.role))
        host.append(span_idx + 1)
        template.append(template_id)

    return FSNTGV2State(
        span_attrs={
            "key": key,
            "harm_root": harm_root,
            "harm_quality": harm_quality,
            "harm_function": harm_function,
            "meter": meter,
            "section": section,
            "reg_center": reg_center,
        },
        note_attrs={
            "active": active,
            "pitch_token": pitch_tokens,
            "velocity": velocity,
            "role": role,
        },
        host=host,
        template=template,
        e_ss=e_ss,
        span_starts=span_starts,
        ticks_per_span=tps,
        metadata={
            "song_id": song_id,
            "span_resolution": cfg.span_resolution,
            "meter_class": str(beats_per_bar * 16 + beat_unit),
        },
    )


def _split_indices(num_items: int, train_ratio: float, valid_ratio: float, seed: int):
    idx = list(range(num_items))
    random.Random(seed).shuffle(idx)
    n_train = int(num_items * train_ratio)
    n_valid = int(num_items * valid_ratio)
    train = idx[:n_train]
    valid = idx[n_train : n_train + n_valid]
    test = idx[n_train + n_valid :]
    return train, valid, test


def preprocess_pop909(cfg: PreprocessConfig) -> dict:
    raw_root = Path(cfg.raw_root).expanduser().resolve()
    output_root = Path(cfg.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    song_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    raw_songs: List[Tuple[Path, List[RawNoteEvent], int, int, int]] = []
    for song_dir in song_dirs:
        midi_path = _find_midi(song_dir)
        if midi_path is None:
            continue
        try:
            events, tpq, beats_per_bar, beat_unit = _load_note_events(midi_path)
        except Exception:
            continue
        if len(events) < cfg.min_notes_per_song:
            continue
        raw_songs.append((song_dir, events, tpq, beats_per_bar, beat_unit))

    records = _collect_template_records(raw_songs, cfg)
    rhythm_vocab = RhythmTemplateVocab(
        top_k_per_meter=cfg.top_k_per_meter,
        onset_bins=cfg.onset_bins,
        max_extension_class=cfg.max_extension_class,
    )
    rhythm_vocab.fit(records)
    pitch_codec = PitchTokenCodec()

    states: List[FSNTGV2State] = []
    for song_dir, events, tpq, beats_per_bar, beat_unit in raw_songs:
        chords = load_pop909_chords(song_dir)
        state = _build_state(
            song_id=song_dir.name,
            events=events,
            tpq=tpq,
            beats_per_bar=beats_per_bar,
            beat_unit=beat_unit,
            cfg=cfg,
            rhythm_vocab=rhythm_vocab,
            pitch_codec=pitch_codec,
            chord_rows=chords,
        )
        states.append(state)

    train_idx, valid_idx, test_idx = _split_indices(len(states), cfg.train_ratio, cfg.valid_ratio, cfg.split_seed)
    train = [states[i] for i in train_idx]
    valid = [states[i] for i in valid_idx]
    test = [states[i] for i in test_idx]

    write_jsonl(output_root / "train.jsonl", (st.to_dict() for st in train))
    write_jsonl(output_root / "valid.jsonl", (st.to_dict() for st in valid))
    write_jsonl(output_root / "test.jsonl", (st.to_dict() for st in test))

    save_json(output_root / "rhythm_templates.json", rhythm_vocab.to_dict())
    save_json(output_root / "pitch_codec.json", pitch_codec.to_dict())
    save_json(output_root / "preprocessing_config.json", asdict(cfg))

    stats = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "rhythm_template_vocab_version": "rhythm_templates_v2",
        "pitch_codec_version": "harmony_relative_root_quality_function_v4",
        "harmonic_semantics_version": "root_quality_function_v1",
        "num_songs": len(states),
        "num_train": len(train),
        "num_valid": len(valid),
        "num_test": len(test),
        "span_resolution": cfg.span_resolution,
        "rhythm_template_stats": rhythm_vocab.stats(),
    }
    save_json(output_root / "stats.json", stats)
    return stats
