from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab


def test_template_vocab_encode_decode():
    vocab = RhythmTemplateVocab(top_k_per_meter=2)
    observed = [
        (0, 0, 3, 0, 0),
        (0, 4, 3, 0, 0),
        (0, 0, 3, 0, 0),
        (1, 3, 2, 0, 0),
    ]
    vocab.fit(observed)
    tid = vocab.encode(0, 0, 3, 0, 0)
    tpl = vocab.decode(tid)
    assert tpl.meter == 0
    assert tpl.onset_bin == 0
    assert vocab.vocab_size >= 2
