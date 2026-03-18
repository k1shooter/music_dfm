"""Factorized Span-Note Template Graph representation and deterministic utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.utils.constants import E_NS_NONE, E_SS_NONE, NOTE_CHANNELS, SPAN_CHANNELS


@dataclass
class AuxNoteGraph:
    same_onset: List[Tuple[int, int]] = field(default_factory=list)
    overlap: List[Tuple[int, int]] = field(default_factory=list)
    sequential_same_role: List[Tuple[int, int]] = field(default_factory=list)


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
class FSNTGState:
    span_attrs: Dict[str, List[int]]
    note_attrs: Dict[str, List[int]]
    e_ns: List[List[int]]
    e_ss: List[List[int]]
    span_starts: List[int]
    ticks_per_span: int = 1920
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate_shapes()

    @property
    def num_spans(self) -> int:
        return len(self.span_starts)

    @property
    def num_notes(self) -> int:
        return len(self.note_attrs["active"])

    def copy(self) -> "FSNTGState":
        return deepcopy(self)

    def validate_shapes(self) -> None:
        s = len(self.span_starts)
        for c in SPAN_CHANNELS:
            if c not in self.span_attrs:
                raise ValueError(f"Missing span channel: {c}")
            if len(self.span_attrs[c]) != s:
                raise ValueError(f"Invalid span length for {c}: {len(self.span_attrs[c])} != {s}")

        n = len(self.note_attrs.get("active", []))
        for c in NOTE_CHANNELS:
            if c not in self.note_attrs:
                raise ValueError(f"Missing note channel: {c}")
            if len(self.note_attrs[c]) != n:
                raise ValueError(f"Invalid note length for {c}: {len(self.note_attrs[c])} != {n}")

        if len(self.e_ns) != n:
            raise ValueError("e_ns row count must match number of notes")
        for row in self.e_ns:
            if len(row) != s:
                raise ValueError("e_ns column count must match number of spans")

        if len(self.e_ss) != s:
            raise ValueError("e_ss row count must match number of spans")
        for row in self.e_ss:
            if len(row) != s:
                raise ValueError("e_ss column count must match number of spans")

    def to_dict(self) -> dict:
        return {
            "span_attrs": self.span_attrs,
            "note_attrs": self.note_attrs,
            "e_ns": self.e_ns,
            "e_ss": self.e_ss,
            "span_starts": self.span_starts,
            "ticks_per_span": self.ticks_per_span,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FSNTGState":
        return cls(
            span_attrs={k: list(v) for k, v in payload["span_attrs"].items()},
            note_attrs={k: list(v) for k, v in payload["note_attrs"].items()},
            e_ns=[list(row) for row in payload["e_ns"]],
            e_ss=[list(row) for row in payload["e_ss"]],
            span_starts=list(payload["span_starts"]),
            ticks_per_span=int(payload.get("ticks_per_span", 1920)),
            metadata=dict(payload.get("metadata", {})),
        )


def choose_host_span(row: List[int], active: int, scores: List[float] | None = None) -> int:
    if not active:
        return -1
    if scores is not None and len(scores) == len(row):
        best_j = max(range(len(row)), key=lambda j: scores[j])
        return int(best_j)

    non_none = [j for j, edge in enumerate(row) if edge != E_NS_NONE]
    return int(non_none[0]) if non_none else 0


def project_one_host_per_active_note(
    state: FSNTGState,
    non_none_scores: List[List[float]] | None = None,
    default_template_id: int = 1,
) -> FSNTGState:
    """Projection that ensures each active note has exactly one host span edge."""
    out = state.copy()
    for i in range(out.num_notes):
        active = int(out.note_attrs["active"][i])
        row = out.e_ns[i]
        scores = non_none_scores[i] if non_none_scores is not None else None
        host = choose_host_span(row, active, scores)
        if not active:
            out.e_ns[i] = [E_NS_NONE for _ in range(out.num_spans)]
            continue

        chosen_edge = row[host] if 0 <= host < len(row) else E_NS_NONE
        if chosen_edge == E_NS_NONE:
            candidates = [x for x in row if x != E_NS_NONE]
            chosen_edge = candidates[0] if candidates else default_template_id
        out.e_ns[i] = [E_NS_NONE for _ in range(out.num_spans)]
        out.e_ns[i][host] = int(chosen_edge)
    return out


def cleanup_duplicate_notes(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> FSNTGState:
    """Deactivates duplicate notes with same (host, onset, duration, pitch, role)."""
    out = state.copy()
    decoded = decode_notes(out, template_vocab, pitch_codec)
    seen = set()
    for note in decoded:
        key = (note.host_span, note.onset_tick, note.duration_tick, note.pitch, note.role)
        if key in seen:
            i = note.note_idx
            out.note_attrs["active"][i] = 0
            out.e_ns[i] = [E_NS_NONE for _ in range(out.num_spans)]
        else:
            seen.add(key)
    return out


def decode_notes(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> List[DecodedNote]:
    notes: List[DecodedNote] = []
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        host = choose_host_span(state.e_ns[i], 1)
        template_id = state.e_ns[i][host] if host >= 0 else E_NS_NONE
        if template_id == E_NS_NONE:
            continue
        onset = state.span_starts[host] + template_vocab.onset_to_ticks(template_id, state.ticks_per_span)
        duration = template_vocab.duration_to_ticks(template_id)
        pitch_token = int(state.note_attrs["pitch_token"][i])
        velocity_bin = int(state.note_attrs["velocity"][i])
        velocity = max(1, min(127, int(16 + velocity_bin * 8)))
        role = int(state.note_attrs["role"][i])
        pitch = pitch_codec.absolute_pitch(
            key=int(state.span_attrs["key"][host]),
            harm=int(state.span_attrs["harm"][host]),
            reg_center=int(state.span_attrs["reg_center"][host]),
            token=pitch_token,
        )
        notes.append(
            DecodedNote(
                note_idx=i,
                host_span=host,
                onset_tick=onset,
                duration_tick=duration,
                pitch=pitch,
                velocity=velocity,
                role=role,
            )
        )
    return notes


def reconstruct_aux_graph(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> AuxNoteGraph:
    """Deterministically derives local note-note relations from decoded notes."""
    notes = decode_notes(state, template_vocab, pitch_codec)
    graph = AuxNoteGraph()
    by_role: Dict[int, List[DecodedNote]] = {}
    for n in notes:
        by_role.setdefault(n.role, []).append(n)

    for a_idx in range(len(notes)):
        for b_idx in range(a_idx + 1, len(notes)):
            a = notes[a_idx]
            b = notes[b_idx]
            if a.onset_tick == b.onset_tick:
                graph.same_onset.append((a.note_idx, b.note_idx))
                graph.same_onset.append((b.note_idx, a.note_idx))
            a_end = a.onset_tick + a.duration_tick
            b_end = b.onset_tick + b.duration_tick
            if a.onset_tick < b_end and b.onset_tick < a_end:
                graph.overlap.append((a.note_idx, b.note_idx))
                graph.overlap.append((b.note_idx, a.note_idx))

    for role, role_notes in by_role.items():
        del role
        ordered = sorted(role_notes, key=lambda n: (n.onset_tick, n.note_idx))
        for k in range(len(ordered) - 1):
            graph.sequential_same_role.append((ordered[k].note_idx, ordered[k + 1].note_idx))

    return graph


def empty_state(num_spans: int, num_notes: int, ticks_per_span: int = 1920) -> FSNTGState:
    span_attrs = {c: [0 for _ in range(num_spans)] for c in SPAN_CHANNELS}
    note_attrs = {c: [0 for _ in range(num_notes)] for c in NOTE_CHANNELS}
    e_ns = [[E_NS_NONE for _ in range(num_spans)] for _ in range(num_notes)]
    e_ss = [[E_SS_NONE for _ in range(num_spans)] for _ in range(num_spans)]
    span_starts = [j * ticks_per_span for j in range(num_spans)]
    return FSNTGState(
        span_attrs=span_attrs,
        note_attrs=note_attrs,
        e_ns=e_ns,
        e_ss=e_ss,
        span_starts=span_starts,
        ticks_per_span=ticks_per_span,
    )
