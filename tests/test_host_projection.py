from music_graph_dfm.representation.state import FSNTGV2State


def test_host_template_projection_validity():
    state = FSNTGV2State(
        span_attrs={
            "key": [0, 0],
            "harm_root": [0, 7],
            "harm_quality": [1, 3],
            "meter": [4, 4],
            "section": [0, 0],
            "reg_center": [4, 4],
        },
        note_attrs={
            "active": [1, 0, 1],
            "pitch_token": [1, 2, 3],
            "velocity": [8, 8, 8],
            "role": [0, 0, 0],
        },
        host=[3, 2, 0],
        template=[1, 4, 5],
        e_ss=[[0, 1], [0, 0]],
        span_starts=[0, 480],
        ticks_per_span=480,
    )

    # note0 host out of range -> deactivated, note1 inactive -> zero placement, note2 host=0 -> deactivated
    assert state.note_attrs["active"] == [0, 0, 0]
    assert state.host == [0, 0, 0]
    assert state.template == [0, 0, 0]
