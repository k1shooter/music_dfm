from music_graph_dfm.data.fsntg import empty_state, project_one_host_per_active_note


def test_host_projection_uniqueness():
    st = empty_state(num_spans=3, num_notes=2)
    st.note_attrs["active"] = [1, 1]
    st.e_ns[0] = [1, 2, 0]
    st.e_ns[1] = [0, 0, 3]

    out = project_one_host_per_active_note(st)

    assert sum(1 for e in out.e_ns[0] if e != 0) == 1
    assert sum(1 for e in out.e_ns[1] if e != 0) == 1
