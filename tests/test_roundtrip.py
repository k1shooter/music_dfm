from music_graph_dfm.representation.state import FSNTGV2State, empty_state


def test_state_roundtrip_dict():
    state = empty_state(num_spans=2, num_notes=2)
    state.note_attrs["active"] = [1, 1]
    state.note_attrs["pitch_token"] = [3, 4]
    state.host = [1, 2]
    state.template = [1, 2]

    payload = state.to_dict()
    restored = FSNTGV2State.from_dict(payload)
    assert restored.to_dict() == payload
