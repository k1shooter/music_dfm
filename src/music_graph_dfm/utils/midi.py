"""MIDI decoding helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import DecodedNote, FSNTGV2State


def decode_state_notes(
    state: FSNTGV2State,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> List[DecodedNote]:
    return state.decode_notes(rhythm_vocab=rhythm_vocab, pitch_codec=pitch_codec)


def save_state_midi(
    state: FSNTGV2State,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    out_path: str | Path,
) -> Path:
    try:
        import miditoolkit
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("miditoolkit is required to export MIDI") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    midi = miditoolkit.MidiFile(ticks_per_beat=480)
    notes = decode_state_notes(state, rhythm_vocab, pitch_codec)

    by_role: dict[int, list[DecodedNote]] = {}
    for note in notes:
        by_role.setdefault(note.role, []).append(note)

    for role, role_notes in sorted(by_role.items()):
        instrument = miditoolkit.Instrument(program=0, is_drum=False, name=f"role_{role}")
        for note in role_notes:
            instrument.notes.append(
                miditoolkit.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=int(note.onset_tick),
                    end=int(note.onset_tick + max(1, note.duration_tick)),
                )
            )
        midi.instruments.append(instrument)

    midi.dump(str(out_path))
    return out_path
