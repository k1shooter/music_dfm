"""FSNTG-v2 state and deterministic graph reconstruction utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from music_graph_dfm.constants import E_SS_NONE, NOTE_CHANNELS, SPAN_CHANNELS
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab


@dataclass
class DecodedNote:
    note_idx: int
    host_span: int
    onset_tick: int
    duration_tick: int
    pitch: int
    velocity: int
    role: int


@dataclass
class AuxiliaryNoteGraph:
    same_onset: List[Tuple[int, int]] = field(default_factory=list)
    overlap: List[Tuple[int, int]] = field(default_factory=list)
    sequential_same_role: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class FSNTGV2State:
    span_attrs: Dict[str, List[int]]
    note_attrs: Dict[str, List[int]]
    host: List[int]
    template: List[int]
    e_ss: List[List[int]]
    span_starts: List[int]
    ticks_per_span: int = 480
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate_shapes()
        self.project_placement_consistency()

    @property
    def num_spans(self) -> int:
        return len(self.span_starts)

    @property
    def num_notes(self) -> int:
        return len(self.note_attrs["active"])

    def copy(self) -> "FSNTGV2State":
        return deepcopy(self)

    def validate_shapes(self) -> None:
        s = len(self.span_starts)
        if s <= 0:
            raise ValueError("FSNTG-v2 requires at least one span")

        for channel in SPAN_CHANNELS:
            values = self.span_attrs.get(channel)
            if values is None:
                raise ValueError(f"Missing span channel: {channel}")
            if len(values) != s:
                raise ValueError(f"Invalid span length for {channel}: {len(values)} != {s}")

        n = len(self.note_attrs.get("active", []))
        for channel in NOTE_CHANNELS:
            values = self.note_attrs.get(channel)
            if values is None:
                raise ValueError(f"Missing note channel: {channel}")
            if len(values) != n:
                raise ValueError(f"Invalid note length for {channel}: {len(values)} != {n}")

        if len(self.host) != n:
            raise ValueError("host length must match number of notes")
        if len(self.template) != n:
            raise ValueError("template length must match number of notes")

        if len(self.e_ss) != s:
            raise ValueError("e_ss shape mismatch")
        for row in self.e_ss:
            if len(row) != s:
                raise ValueError("e_ss shape mismatch")

    def project_placement_consistency(self) -> None:
        for i in range(self.num_notes):
            active = int(self.note_attrs["active"][i])
            host = int(self.host[i])
            template = int(self.template[i])
            if active == 0:
                self.host[i] = 0
                self.template[i] = 0
                continue
            if host < 0 or host > self.num_spans:
                self.host[i] = 0
                self.template[i] = 0
                self.note_attrs["active"][i] = 0
                continue
            if host == 0 or template <= 0:
                self.host[i] = 0
                self.template[i] = 0
                self.note_attrs["active"][i] = 0

    def materialize_note_span_template_adjacency(self) -> List[List[int]]:
        adjacency = [[0 for _ in range(self.num_spans)] for _ in range(self.num_notes)]
        for i in range(self.num_notes):
            host = int(self.host[i])
            template = int(self.template[i])
            if int(self.note_attrs["active"][i]) == 1 and host > 0 and template > 0:
                adjacency[i][host - 1] = template
        return adjacency

    def decode_notes(self, rhythm_vocab: RhythmTemplateVocab, pitch_codec: PitchTokenCodec) -> List[DecodedNote]:
        notes: List[DecodedNote] = []
        for i in range(self.num_notes):
            if int(self.note_attrs["active"][i]) == 0:
                continue
            host = int(self.host[i])
            template_id = int(self.template[i])
            if host <= 0 or template_id <= 0:
                continue

            span_idx = host - 1
            if span_idx < 0 or span_idx >= self.num_spans:
                continue

            onset = self.span_starts[span_idx] + rhythm_vocab.onset_ticks(template_id, self.ticks_per_span)
            duration = rhythm_vocab.duration_ticks_with_semantics(template_id, self.ticks_per_span)
            key = int(self.span_attrs["key"][span_idx])
            harm = int(self.span_attrs["harm"][span_idx])
            reg_center = int(self.span_attrs["reg_center"][span_idx])
            pitch = pitch_codec.absolute_pitch(
                key=key,
                harmonic_root=harm,
                reg_center=reg_center,
                token=int(self.note_attrs["pitch_token"][i]),
            )
            velocity_bin = int(self.note_attrs["velocity"][i])
            velocity = max(1, min(127, 12 + velocity_bin * 8))
            notes.append(
                DecodedNote(
                    note_idx=i,
                    host_span=span_idx,
                    onset_tick=int(onset),
                    duration_tick=int(duration),
                    pitch=int(pitch),
                    velocity=int(velocity),
                    role=int(self.note_attrs["role"][i]),
                )
            )
        return notes

    def to_dict(self) -> dict:
        return {
            "span_attrs": self.span_attrs,
            "note_attrs": self.note_attrs,
            "host": self.host,
            "template": self.template,
            "e_ss": self.e_ss,
            "span_starts": self.span_starts,
            "ticks_per_span": self.ticks_per_span,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FSNTGV2State":
        return cls(
            span_attrs={k: list(v) for k, v in payload["span_attrs"].items()},
            note_attrs={k: list(v) for k, v in payload["note_attrs"].items()},
            host=list(payload["host"]),
            template=list(payload["template"]),
            e_ss=[list(row) for row in payload["e_ss"]],
            span_starts=list(payload["span_starts"]),
            ticks_per_span=int(payload.get("ticks_per_span", 480)),
            metadata=dict(payload.get("metadata", {})),
        )


def reconstruct_aux_graph(
    state: FSNTGV2State,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> AuxiliaryNoteGraph:
    notes = state.decode_notes(rhythm_vocab, pitch_codec)
    graph = AuxiliaryNoteGraph()

    for i in range(len(notes)):
        a = notes[i]
        a_end = a.onset_tick + a.duration_tick
        for j in range(i + 1, len(notes)):
            b = notes[j]
            b_end = b.onset_tick + b.duration_tick
            if a.onset_tick == b.onset_tick:
                graph.same_onset.extend([(a.note_idx, b.note_idx), (b.note_idx, a.note_idx)])
            if a.onset_tick < b_end and b.onset_tick < a_end:
                graph.overlap.extend([(a.note_idx, b.note_idx), (b.note_idx, a.note_idx)])

    by_role: Dict[int, List[DecodedNote]] = {}
    for note in notes:
        by_role.setdefault(note.role, []).append(note)
    for role_notes in by_role.values():
        ordered = sorted(role_notes, key=lambda n: (n.onset_tick, n.note_idx))
        for left, right in zip(ordered[:-1], ordered[1:], strict=False):
            graph.sequential_same_role.append((left.note_idx, right.note_idx))

    return graph


def materialize_dense_note_span_view(state: FSNTGV2State) -> List[List[int]]:
    """Optional dense [num_notes, num_spans] view derived from (host, template)."""
    return state.materialize_note_span_template_adjacency()


def project_host_template_validity(state: FSNTGV2State) -> FSNTGV2State:
    """Projects invalid host/template coordinates to a valid inactive representation."""
    out = state.copy()
    out.project_placement_consistency()
    return out


def cleanup_duplicate_notes(
    state: FSNTGV2State,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> FSNTGV2State:
    """Deactivates duplicates with identical decoded (host,onset,duration,pitch,role)."""
    out = state.copy()
    seen = set()
    for note in out.decode_notes(rhythm_vocab, pitch_codec):
        key = (note.host_span, note.onset_tick, note.duration_tick, note.pitch, note.role)
        if key in seen:
            idx = int(note.note_idx)
            out.note_attrs["active"][idx] = 0
            out.host[idx] = 0
            out.template[idx] = 0
            continue
        seen.add(key)
    out.project_placement_consistency()
    return out


def empty_state(num_spans: int, num_notes: int, ticks_per_span: int = 480) -> FSNTGV2State:
    span_attrs = {channel: [0 for _ in range(num_spans)] for channel in SPAN_CHANNELS}
    note_attrs = {channel: [0 for _ in range(num_notes)] for channel in NOTE_CHANNELS}
    host = [0 for _ in range(num_notes)]
    template = [0 for _ in range(num_notes)]
    e_ss = [[E_SS_NONE for _ in range(num_spans)] for _ in range(num_spans)]
    span_starts = [idx * ticks_per_span for idx in range(num_spans)]
    return FSNTGV2State(
        span_attrs=span_attrs,
        note_attrs=note_attrs,
        host=host,
        template=template,
        e_ss=e_ss,
        span_starts=span_starts,
        ticks_per_span=ticks_per_span,
    )
