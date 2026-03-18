from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import FSNTGV2State, reconstruct_aux_graph


def test_aux_graph_reconstruction_uses_decoded_timing():
    rhythm = RhythmTemplateVocab(top_k_per_meter=16, onset_bins=8)
    rhythm.fit(
        [
            (4, 0, 3, 0, 0),
            (4, 0, 3, 0, 0),
            (4, 3, 3, 0, 0),
        ]
    )
    t_same = rhythm.encode(4, 0, 3, 0, 0)
    t_later = rhythm.encode(4, 3, 3, 0, 0)

    codec = PitchTokenCodec()
    token = codec.encode(0, 0)

    state = FSNTGV2State(
        span_attrs={
            "key": [0, 0],
            "harm": [0, 0],
            "meter": [4, 4],
            "section": [0, 0],
            "reg_center": [4, 4],
        },
        note_attrs={
            "active": [1, 1, 1],
            "pitch_token": [token, token, token],
            "velocity": [8, 8, 8],
            "role": [0, 0, 0],
        },
        host=[1, 1, 1],
        template=[t_same, t_same, t_later],
        e_ss=[[0, 1], [0, 0]],
        span_starts=[0, 480],
        ticks_per_span=480,
    )

    graph = reconstruct_aux_graph(state, rhythm, codec)
    assert (0, 1) in graph.same_onset or (1, 0) in graph.same_onset
    assert len(graph.sequential_same_role) >= 1
