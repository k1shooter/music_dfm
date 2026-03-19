from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import FSNTGV2State, reconstruct_aux_graph

import pytest


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
            "harm_root": [0, 0],
            "harm_quality": [1, 1],
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


def test_model_aux_graph_matches_representation_decoded_graph():
    torch = pytest.importorskip("torch", exc_type=ImportError)

    from music_graph_dfm.data import collate_states, infer_vocab_sizes
    from music_graph_dfm.models import FSNTGV2HeteroTransformer, ModelConfig

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
    token = codec.encode_components(0, 0, 0)

    state = FSNTGV2State(
        span_attrs={
            "key": [0, 0],
            "harm_root": [0, 0],
            "harm_quality": [1, 1],
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

    vocab = infer_vocab_sizes([state])
    model = FSNTGV2HeteroTransformer(
        vocab_sizes=vocab,
        cfg=ModelConfig(hidden_dim=16, num_layers=1, num_heads=2, dropout=0.0),
        template_spec={
            "onset_bin": [rhythm.decode(i).onset_bin for i in range(vocab["note.template"])],
            "duration_class": [rhythm.decode(i).duration_class for i in range(vocab["note.template"])],
            "tie_flag": [rhythm.decode(i).tie_flag for i in range(vocab["note.template"])],
            "extension_class": [rhythm.decode(i).extension_class for i in range(vocab["note.template"])],
            "duration_ticks": list(rhythm.duration_ticks),
            "onset_bins": int(rhythm.onset_bins),
            "tie_extension_fraction": float(rhythm.tie_extension_fraction),
        },
    )
    batch = collate_states([state])
    model_rel = model._reconstruct_aux_relations(batch)[0].to(torch.long)

    ref_graph = reconstruct_aux_graph(state, rhythm, codec)
    ref = torch.zeros_like(model_rel)
    for src, dst in ref_graph.same_onset:
        ref[src, dst] = torch.maximum(ref[src, dst], torch.tensor(1, dtype=torch.long))
    for src, dst in ref_graph.overlap:
        ref[src, dst] = torch.maximum(ref[src, dst], torch.tensor(2, dtype=torch.long))
    for src, dst in ref_graph.sequential_same_role:
        ref[src, dst] = torch.maximum(ref[src, dst], torch.tensor(3, dtype=torch.long))

    assert torch.equal(model_rel, ref)
