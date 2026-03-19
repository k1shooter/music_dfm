from music_graph_dfm.representation.pitch_codec import (
    PitchTokenCodec,
    compatibility_table,
    decode_pitch_token,
    encode_pitch_token,
    nearest_token_projection,
)


def test_pitch_token_roundtrip_with_host_state():
    codec = PitchTokenCodec(register_offsets=[-2, -1, 0, 1, 2])
    host = {"key": 0, "harm": 7, "reg_center": 4}
    abs_pitch = 64  # E4

    token = encode_pitch_token(abs_pitch=abs_pitch, host_span_state=host, codec=codec)
    decoded = decode_pitch_token(token=token, host_span_state=host, codec=codec)

    assert 0 <= token < codec.vocab_size
    assert abs(decoded - abs_pitch) <= 6


def test_harmony_relative_decode_changes_with_harmonic_root():
    codec = PitchTokenCodec(register_offsets=[-1, 0, 1])
    host_g = {"key": 0, "harm": 7, "reg_center": 4}
    host_c = {"key": 0, "harm": 0, "reg_center": 4}

    token = encode_pitch_token(abs_pitch=64, host_span_state=host_g, codec=codec)
    pitch_with_g = decode_pitch_token(token=token, host_span_state=host_g, codec=codec)
    pitch_with_c = decode_pitch_token(token=token, host_span_state=host_c, codec=codec)

    assert (pitch_with_g % 12) != (pitch_with_c % 12)


def test_compatibility_table_and_projection_sanity():
    codec = PitchTokenCodec()
    host = {"key": 2, "harm": 9, "reg_center": 5}

    token = nearest_token_projection(abs_pitch=73, host_span_state=host, codec=codec)
    compat = compatibility_table(host_span_state=host, token=token, codec=codec)
    assert compat in {0.0, 1.0}

    full = codec.compatibility_table(num_keys=12, num_harm=12)
    assert len(full) == 12
    assert len(full[0]) == 12
    assert len(full[0][0]) == codec.vocab_size
