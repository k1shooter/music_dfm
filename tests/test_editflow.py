import random

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.diffusion.edit_flow import (
    EditMoveType,
    derive_oracle_edit_move,
    sample_edit_ctmc_step,
    sample_forward_edit_ctmc_source,
)
from music_graph_dfm.representation.state import empty_state


def _target_state():
    st = empty_state(num_spans=2, num_notes=2)
    st.note_attrs["active"] = [1, 1]
    st.note_attrs["pitch_token"] = [3, 5]
    st.note_attrs["velocity"] = [8, 9]
    st.note_attrs["role"] = [0, 1]
    st.host = [1, 2]
    st.template = [2, 3]
    st.e_ss[0][1] = 1
    st.project_placement_consistency()
    return st


def test_forward_edit_ctmc_source_is_valid():
    target = _target_state()
    vocab_sizes = {
        "note.pitch_token": 16,
        "note.velocity": 16,
        "note.role": 8,
        "note.host": 4,
        "note.template": 6,
        "e_ss.relation": 6,
    }
    source = sample_forward_edit_ctmc_source(
        target_state=target,
        vocab_sizes=vocab_sizes,
        rng=random.Random(7),
        num_steps=3,
        h=0.5,
    )
    source.validate_shapes()
    source.project_placement_consistency()
    for i, active in enumerate(source.note_attrs["active"]):
        if int(active) == 0:
            assert int(source.host[i]) == 0
            assert int(source.template[i]) == 0
    assert derive_oracle_edit_move(source, target) is None or isinstance(
        derive_oracle_edit_move(source, target).move_type, EditMoveType
    )


def test_edit_sampler_substitute_host_is_offdiagonal():
    state = _target_state()
    # Force SUBSTITUTE_HOST with near-certain jump.
    lam = torch.full((1, 6), -20.0)
    lam[0, int(EditMoveType.SUBSTITUTE_HOST)] = 20.0
    edit_outputs_single = {
        "lambda_type": lam,
        "type_logits": torch.zeros((1, 6)),
        "note_logits": torch.zeros((1, state.num_notes)),
        "host_logits": torch.zeros((1, state.num_notes, 4)),
        "template_logits": torch.zeros((1, state.num_notes, 6)),
        "pitch_logits": torch.zeros((1, state.num_notes, 8)),
        "velocity_logits": torch.zeros((1, state.num_notes, 8)),
        "role_logits": torch.zeros((1, state.num_notes, 4)),
        "span_src_logits": torch.zeros((1, state.num_spans)),
        "span_dst_logits": torch.zeros((1, state.num_spans)),
        "insert_host_logits": torch.zeros((1, 4)),
        "insert_template_logits": torch.zeros((1, 6)),
        "insert_pitch_logits": torch.zeros((1, 8)),
        "insert_velocity_logits": torch.zeros((1, 8)),
        "insert_role_logits": torch.zeros((1, 4)),
        "span_rel_logits": torch.zeros((1, state.num_spans, state.num_spans, 6)),
    }
    edit_outputs_single["host_logits"][0, 0, 2] = 10.0
    next_state = sample_edit_ctmc_step(state=state, edit_outputs_single=edit_outputs_single, h=1.0)
    assert int(next_state.host[0]) != int(state.host[0])

