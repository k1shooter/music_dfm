import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.data.tensor_codec import coords_to_states, states_to_coords
from music_graph_dfm.diffusion.masking import enforce_state_constraints
from music_graph_dfm.representation.state import FSNTGV2State, materialize_dense_note_span_view


def _state() -> FSNTGV2State:
    return FSNTGV2State(
        span_attrs={
            "key": [0, 2],
            "harm_root": [0, 7],
            "harm_quality": [1, 3],
            "meter": [4, 4],
            "section": [0, 1],
            "reg_center": [4, 4],
        },
        note_attrs={
            "active": [1, 0, 1],
            "pitch_token": [3, 5, 7],
            "velocity": [8, 8, 9],
            "role": [0, 1, 0],
        },
        host=[1, 2, 2],
        template=[1, 3, 2],
        e_ss=[[0, 1], [0, 0]],
        span_starts=[0, 480],
        ticks_per_span=480,
    )


def test_coords_roundtrip_fsntg_v2():
    state = _state()
    packed = states_to_coords([state], include_dense_note_span_view=False)
    coords = packed["coords"]
    batch = packed["batch"]

    restored = coords_to_states(coords, batch)[0]
    assert restored.span_attrs == state.span_attrs
    assert restored.note_attrs == state.note_attrs
    assert restored.host == state.host
    assert restored.template == state.template
    assert restored.e_ss == state.e_ss


def test_derived_dense_view_matches_host_template():
    state = _state()
    dense = materialize_dense_note_span_view(state)
    assert dense[0][0] == 1
    assert dense[2][1] == 2
    assert sum(dense[1]) == 0  # inactive note has no adjacency


def test_inactive_notes_force_zero_host_template_after_constraints():
    packed = states_to_coords([_state()], include_dense_note_span_view=False)
    coords = packed["coords"]
    batch = {
        "span_mask": packed["batch"]["span_mask"],
        "note_mask": packed["batch"]["note_mask"],
    }

    coords["note.active"][0, 2] = 0
    coords = enforce_state_constraints(coords, batch)
    assert int(coords["note.host"][0, 2].item()) == 0
    assert int(coords["note.template"][0, 2].item()) == 0
