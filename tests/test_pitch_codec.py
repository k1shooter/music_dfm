from music_graph_dfm.data.pitch_codec import PitchTokenCodec


def test_pitch_token_encode_decode():
    codec = PitchTokenCodec(degrees=[0, 2, 4, 7], register_offsets=[-1, 0, 1])
    tok = codec.encode(4, 1)
    pt = codec.decode(tok)
    assert pt.degree == 4
    assert pt.register_offset == 1


def test_absolute_pitch_range():
    codec = PitchTokenCodec()
    tok = codec.encode(7, 0)
    p = codec.absolute_pitch(key=0, harm=0, reg_center=4, token=tok)
    assert 0 <= p <= 127
