from music_graph_dfm.representation.pitch_codec import PitchTokenCodec, decode_pitch_token, encode_pitch_token


def test_harmony_relative_encode_decode():
    codec = PitchTokenCodec(register_offsets=[-1, 0, 1])

    # E4 (64) under harmony root G (7) -> degree_wrt_harmony = 9.
    token = codec.encode_from_absolute_pitch(pitch=64, harmonic_root=7, reg_center=4)
    decoded = decode_pitch_token(codec, token)
    assert decoded.degree_wrt_harmony == 9

    reconstructed = codec.absolute_pitch(key=0, harmonic_root=7, reg_center=4, token=token)
    assert abs(reconstructed - 64) <= 6


def test_codec_helpers_and_compatibility_table_shape():
    codec = PitchTokenCodec()
    token = encode_pitch_token(codec, degree_wrt_harmony=4, register_offset=1)
    point = decode_pitch_token(codec, token)
    assert point.degree_wrt_harmony == 4
    assert point.register_offset == 1

    table = codec.compatibility_table(num_keys=12, num_harm=12)
    assert len(table) == 12
    assert len(table[0]) == 12
    assert len(table[0][0]) == codec.vocab_size
