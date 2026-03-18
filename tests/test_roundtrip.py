from music_graph_dfm.data.fsntg import FSNTGState, empty_state


def test_graph_extraction_roundtrip_dict():
    st = empty_state(num_spans=3, num_notes=2)
    st.note_attrs["active"] = [1, 1]
    st.note_attrs["pitch_token"] = [3, 5]
    st.note_attrs["velocity"] = [8, 10]
    st.note_attrs["role"] = [0, 1]
    st.e_ns[0][0] = 1
    st.e_ns[1][2] = 2
    st.e_ss[0][1] = 1

    payload = st.to_dict()
    st2 = FSNTGState.from_dict(payload)

    assert st2.to_dict() == payload
