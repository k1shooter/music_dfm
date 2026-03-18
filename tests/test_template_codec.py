from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab


def test_template_encode_decode_and_duration_semantics():
    vocab = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8, max_extension_class=4)
    observed = [
        (4, 0, 3, 0, 0),
        (4, 2, 4, 1, 1),
        (4, 2, 4, 1, 2),
        (3, 1, 2, 0, 0),
    ]
    vocab.fit(observed)

    template_id = vocab.encode(4, 2, 4, 1, 2)
    tpl = vocab.decode(template_id)
    assert tpl.onset_bin == 2
    assert tpl.tie_flag == 1
    assert tpl.extension_class == 2

    ticks_per_span = 480
    duration = vocab.duration_ticks_with_semantics(template_id, ticks_per_span=ticks_per_span)
    base = vocab.duration_ticks[tpl.duration_class]
    assert duration >= base + ticks_per_span * 2
