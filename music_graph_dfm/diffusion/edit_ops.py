"""EditFlow graph-edit operations for FSNTG states."""

from __future__ import annotations

import random
from typing import Dict

from music_graph_dfm.data.fsntg import FSNTGState
from music_graph_dfm.utils.constants import E_NS_NONE, E_SS_NONE, NOTE_CHANNELS


def insert_note(
    state: FSNTGState,
    host_span: int,
    template_id: int,
    pitch_token: int,
    velocity_bin: int,
    role: int,
) -> FSNTGState:
    out = state.copy()
    for c in NOTE_CHANNELS:
        out.note_attrs[c].append(0)
    out.note_attrs["active"][-1] = 1
    out.note_attrs["pitch_token"][-1] = int(pitch_token)
    out.note_attrs["velocity"][-1] = int(velocity_bin)
    out.note_attrs["role"][-1] = int(role)
    row = [E_NS_NONE for _ in range(out.num_spans)]
    row[max(0, min(out.num_spans - 1, host_span))] = int(template_id)
    out.e_ns.append(row)
    out.validate_shapes()
    return out


def delete_note(state: FSNTGState, note_idx: int) -> FSNTGState:
    out = state.copy()
    if not (0 <= note_idx < out.num_notes):
        return out
    for c in NOTE_CHANNELS:
        out.note_attrs[c].pop(note_idx)
    out.e_ns.pop(note_idx)
    out.validate_shapes()
    return out


def substitute_note_content(
    state: FSNTGState,
    note_idx: int,
    pitch_token: int | None = None,
    velocity_bin: int | None = None,
    role: int | None = None,
) -> FSNTGState:
    out = state.copy()
    if not (0 <= note_idx < out.num_notes):
        return out
    if pitch_token is not None:
        out.note_attrs["pitch_token"][note_idx] = int(pitch_token)
    if velocity_bin is not None:
        out.note_attrs["velocity"][note_idx] = int(velocity_bin)
    if role is not None:
        out.note_attrs["role"][note_idx] = int(role)
    return out


def substitute_note_span_template(state: FSNTGState, note_idx: int, host_span: int, template_id: int) -> FSNTGState:
    out = state.copy()
    if not (0 <= note_idx < out.num_notes):
        return out
    if not (0 <= host_span < out.num_spans):
        return out
    out.e_ns[note_idx] = [E_NS_NONE for _ in range(out.num_spans)]
    out.e_ns[note_idx][host_span] = int(template_id)
    return out


def substitute_span_relation(state: FSNTGState, src: int, dst: int, relation: int) -> FSNTGState:
    out = state.copy()
    if not (0 <= src < out.num_spans and 0 <= dst < out.num_spans):
        return out
    out.e_ss[src][dst] = int(relation)
    return out


def random_edit_step(state: FSNTGState, vocab_sizes: Dict[str, int], p_insert: float = 0.2, p_delete: float = 0.2) -> FSNTGState:
    out = state.copy()
    u = random.random()

    if u < p_insert:
        return insert_note(
            out,
            host_span=random.randint(0, max(0, out.num_spans - 1)),
            template_id=random.randint(1, max(1, vocab_sizes.get("e_ns.template", 2) - 1)),
            pitch_token=random.randint(0, max(0, vocab_sizes.get("note.pitch_token", 1) - 1)),
            velocity_bin=random.randint(0, max(0, vocab_sizes.get("note.velocity", 1) - 1)),
            role=random.randint(0, max(0, vocab_sizes.get("note.role", 1) - 1)),
        )

    if u < p_insert + p_delete and out.num_notes > 0:
        return delete_note(out, random.randint(0, out.num_notes - 1))

    # substitutions
    if out.num_notes > 0 and random.random() < 0.5:
        i = random.randint(0, out.num_notes - 1)
        out = substitute_note_content(
            out,
            i,
            pitch_token=random.randint(0, max(0, vocab_sizes.get("note.pitch_token", 1) - 1)),
            velocity_bin=random.randint(0, max(0, vocab_sizes.get("note.velocity", 1) - 1)),
        )
        out = substitute_note_span_template(
            out,
            i,
            host_span=random.randint(0, max(0, out.num_spans - 1)),
            template_id=random.randint(1, max(1, vocab_sizes.get("e_ns.template", 2) - 1)),
        )
    else:
        src = random.randint(0, max(0, out.num_spans - 1))
        dst = random.randint(0, max(0, out.num_spans - 1))
        rel = random.randint(0, max(0, vocab_sizes.get("e_ss.relation", 1) - 1))
        out = substitute_span_relation(out, src, dst, rel)

    return out


def validate_state_consistency(state: FSNTGState) -> bool:
    try:
        state.validate_shapes()
        # every active note must have <= 1 host in edit consistency mode
        for i in range(state.num_notes):
            active = int(state.note_attrs["active"][i])
            non_none = sum(1 for e in state.e_ns[i] if e != E_NS_NONE)
            if active == 0 and non_none != 0:
                return False
            if active == 1 and non_none > 1:
                return False
        # e_ss valid ids non-negative
        for row in state.e_ss:
            for v in row:
                if v < E_SS_NONE:
                    return False
    except Exception:
        return False
    return True
