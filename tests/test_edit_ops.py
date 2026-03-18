from music_graph_dfm.diffusion.edit_flow import apply_edit_move, derive_oracle_edit_move
from music_graph_dfm.representation.state import empty_state


def test_edit_flow_oracle_and_transition_validity():
    target = empty_state(num_spans=2, num_notes=2)
    target.note_attrs["active"] = [1, 1]
    target.note_attrs["pitch_token"] = [3, 5]
    target.note_attrs["velocity"] = [8, 9]
    target.note_attrs["role"] = [0, 1]
    target.host = [1, 2]
    target.template = [2, 3]

    source = empty_state(num_spans=2, num_notes=1)
    source.note_attrs["active"] = [1]
    source.note_attrs["pitch_token"] = [3]
    source.note_attrs["velocity"] = [8]
    source.note_attrs["role"] = [0]
    source.host = [1]
    source.template = [2]

    move = derive_oracle_edit_move(source, target)
    assert move is not None

    updated = apply_edit_move(source, move)
    assert updated.num_notes == 2
    assert updated.note_attrs["active"][1] == 1
    assert updated.host[1] == 2
