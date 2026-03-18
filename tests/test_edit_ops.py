from music_graph_dfm.data.fsntg import empty_state
from music_graph_dfm.diffusion.edit_ops import (
    delete_note,
    insert_note,
    substitute_note_content,
    substitute_note_span_template,
    validate_state_consistency,
)


def test_edit_ops_consistency():
    st = empty_state(num_spans=3, num_notes=1)
    st.note_attrs["active"][0] = 1
    st.e_ns[0][0] = 1

    st = insert_note(st, host_span=2, template_id=1, pitch_token=2, velocity_bin=8, role=0)
    st = substitute_note_content(st, note_idx=0, pitch_token=3)
    st = substitute_note_span_template(st, note_idx=0, host_span=1, template_id=2)
    st = delete_note(st, note_idx=1)

    assert validate_state_consistency(st)
