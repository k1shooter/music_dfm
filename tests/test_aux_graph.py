from music_graph_dfm.data.fsntg import empty_state, reconstruct_aux_graph
from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab


def test_aux_graph_reconstruction():
    vocab = RhythmTemplateVocab(top_k_per_meter=8)
    vocab.fit(
        [
            (0, 0, 3, 0, 0),
            (0, 0, 3, 0, 0),
            (0, 4, 3, 0, 0),
        ]
    )
    t_same = vocab.encode(0, 0, 3, 0, 0)
    t_next = vocab.encode(0, 4, 3, 0, 0)

    st = empty_state(num_spans=2, num_notes=3)
    st.note_attrs["active"] = [1, 1, 1]
    st.note_attrs["pitch_token"] = [1, 2, 3]
    st.note_attrs["velocity"] = [8, 8, 8]
    st.note_attrs["role"] = [0, 0, 0]
    st.e_ns[0][0] = t_same
    st.e_ns[1][0] = t_same
    st.e_ns[2][0] = t_next

    g = reconstruct_aux_graph(st, vocab, PitchTokenCodec())

    assert (0, 1) in g.same_onset or (1, 0) in g.same_onset
    assert len(g.sequential_same_role) >= 1
