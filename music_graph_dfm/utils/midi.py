"""MIDI and piano-roll conversion utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

from music_graph_dfm.data.fsntg import FSNTGState, decode_notes
from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab


def fsntg_to_midi(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
):
    try:
        import miditoolkit
    except Exception as exc:
        raise RuntimeError("miditoolkit is required for MIDI export") from exc

    midi = miditoolkit.MidiFile(ticks_per_beat=max(120, state.ticks_per_span // 4))
    notes = decode_notes(state, template_vocab, pitch_codec)

    role_to_inst = {}
    for n in notes:
        inst = role_to_inst.get(n.role)
        if inst is None:
            inst = miditoolkit.Instrument(program=0, is_drum=False, name=f"role_{n.role}")
            role_to_inst[n.role] = inst
            midi.instruments.append(inst)
        inst.notes.append(
            miditoolkit.Note(
                pitch=int(n.pitch),
                velocity=int(n.velocity),
                start=int(n.onset_tick),
                end=int(n.onset_tick + n.duration_tick),
            )
        )
    return midi


def save_midi(
    path: str | Path,
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = fsntg_to_midi(state, template_vocab, pitch_codec)
    midi.dump(str(path))
    return path


def to_pianoroll(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    time_bins: int = 512,
) -> List[List[int]]:
    notes = decode_notes(state, template_vocab, pitch_codec)
    if not notes:
        return [[0 for _ in range(time_bins)] for _ in range(128)]
    max_tick = max(n.onset_tick + n.duration_tick for n in notes)
    roll = [[0 for _ in range(time_bins)] for _ in range(128)]
    for n in notes:
        t0 = int((n.onset_tick / max(1, max_tick)) * (time_bins - 1))
        t1 = int(((n.onset_tick + n.duration_tick) / max(1, max_tick)) * (time_bins - 1))
        for t in range(max(0, t0), min(time_bins, t1 + 1)):
            roll[n.pitch][t] = max(roll[n.pitch][t], n.velocity)
    return roll
