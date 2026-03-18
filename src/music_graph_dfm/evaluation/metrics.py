"""Generated-sample symbolic, graph-validity, and structure metrics."""

from __future__ import annotations

from typing import Dict, List

from music_graph_dfm.constants import SPAN_RELATIONS
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import FSNTGV2State
from music_graph_dfm.utils.midi import decode_state_notes


def ook_rate(state: FSNTGV2State, pitch_codec: PitchTokenCodec) -> float:
    total = 0
    bad = 0
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        total += 1
        host = int(state.host[i])
        if host <= 0 or host > state.num_spans:
            bad += 1
            continue
        span_idx = host - 1
        key = int(state.span_attrs["key"][span_idx])
        harm = int(state.span_attrs["harm"][span_idx])
        token = int(state.note_attrs["pitch_token"][i])
        if not pitch_codec.is_compatible(key, harm, token):
            bad += 1
    return bad / max(1, total)


def note_density(state: FSNTGV2State) -> float:
    active = sum(1 for x in state.note_attrs["active"] if int(x) == 1)
    return active / max(1, state.num_spans)


def host_uniqueness(state: FSNTGV2State) -> float:
    total = 0
    good = 0
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        total += 1
        if int(state.host[i]) > 0:
            good += 1
    return good / max(1, total)


def duplicate_note_rate(state: FSNTGV2State, rhythm_vocab: RhythmTemplateVocab, pitch_codec: PitchTokenCodec) -> float:
    notes = decode_state_notes(state, rhythm_vocab, pitch_codec)
    seen = set()
    dup = 0
    for n in notes:
        key = (n.host_span, n.onset_tick, n.duration_tick, n.pitch, n.role)
        if key in seen:
            dup += 1
        seen.add(key)
    return dup / max(1, len(notes))


def invalid_decode_rate(state: FSNTGV2State, rhythm_vocab: RhythmTemplateVocab, pitch_codec: PitchTokenCodec) -> float:
    decoded = decode_state_notes(state, rhythm_vocab, pitch_codec)
    active = sum(1 for x in state.note_attrs["active"] if int(x) == 1)
    return 1.0 - (len(decoded) / max(1, active))


def _voice_leading_large_leap_rate(state: FSNTGV2State, rhythm_vocab: RhythmTemplateVocab, pitch_codec: PitchTokenCodec) -> float:
    notes = decode_state_notes(state, rhythm_vocab, pitch_codec)
    by_role: dict[int, list] = {}
    for n in notes:
        by_role.setdefault(n.role, []).append(n)

    total = 0
    bad = 0
    for role_notes in by_role.values():
        ordered = sorted(role_notes, key=lambda x: (x.onset_tick, x.note_idx))
        for a, b in zip(ordered[:-1], ordered[1:]):
            total += 1
            if abs(int(b.pitch) - int(a.pitch)) > 12:
                bad += 1
    return bad / max(1, total)


def _groove_hist(state: FSNTGV2State, rhythm_vocab: RhythmTemplateVocab) -> List[float]:
    hist = [0 for _ in range(rhythm_vocab.onset_bins)]
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        template_id = int(state.template[i])
        if template_id <= 0:
            continue
        onset = rhythm_vocab.decode(template_id).onset_bin
        onset = max(0, min(rhythm_vocab.onset_bins - 1, onset))
        hist[onset] += 1
    total = sum(hist)
    return [h / max(1, total) for h in hist]


def groove_similarity(generated: FSNTGV2State, reference: FSNTGV2State, rhythm_vocab: RhythmTemplateVocab) -> float:
    a = _groove_hist(generated, rhythm_vocab)
    b = _groove_hist(reference, rhythm_vocab)
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / max(1e-8, den)


def chord_metrics(generated: FSNTGV2State, reference: FSNTGV2State) -> Dict[str, float]:
    n = min(generated.num_spans, reference.num_spans)
    if n <= 0:
        return {"chord_accuracy": 0.0, "chord_similarity": 0.0}

    exact = 0
    root_match = 0
    for i in range(n):
        g_key, g_harm = int(generated.span_attrs["key"][i]), int(generated.span_attrs["harm"][i])
        r_key, r_harm = int(reference.span_attrs["key"][i]), int(reference.span_attrs["harm"][i])
        if g_harm == r_harm:
            root_match += 1
        if g_key == r_key and g_harm == r_harm:
            exact += 1
    return {
        "chord_accuracy": exact / n,
        "chord_similarity": root_match / n,
    }


def span_relation_accuracy(generated: FSNTGV2State, reference: FSNTGV2State) -> float:
    n = min(generated.num_spans, reference.num_spans)
    if n <= 0:
        return 0.0
    total = n * n
    match = 0
    for i in range(n):
        for j in range(n):
            if int(generated.e_ss[i][j]) == int(reference.e_ss[i][j]):
                match += 1
    return match / max(1, total)


def phrase_repetition_consistency(state: FSNTGV2State) -> float:
    repeat = SPAN_RELATIONS.index("repeat")
    variation = SPAN_RELATIONS.index("variation")
    pairs = []
    for i in range(state.num_spans):
        for j in range(state.num_spans):
            if int(state.e_ss[i][j]) in {repeat, variation}:
                pairs.append((i, j))
    if not pairs:
        return 1.0
    good = sum(1 for i, j in pairs if int(state.span_attrs["section"][i]) == int(state.span_attrs["section"][j]))
    return good / len(pairs)


def graph_validity_metrics(state: FSNTGV2State) -> Dict[str, float]:
    active = 0
    invalid_host = 0
    invalid_template = 0
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        active += 1
        host = int(state.host[i])
        template = int(state.template[i])
        if host <= 0 or host > state.num_spans:
            invalid_host += 1
        if template <= 0:
            invalid_template += 1
    return {
        "host_uniqueness": host_uniqueness(state),
        "invalid_host_rate": invalid_host / max(1, active),
        "invalid_template_rate": invalid_template / max(1, active),
    }


def direct_symbolic_metrics(state: FSNTGV2State, rhythm_vocab: RhythmTemplateVocab, pitch_codec: PitchTokenCodec) -> Dict[str, float]:
    notes = decode_state_notes(state, rhythm_vocab, pitch_codec)
    if not notes:
        return {
            "pitch_range": 0.0,
            "mean_duration": 0.0,
            "mean_velocity": 0.0,
        }
    pitches = [n.pitch for n in notes]
    durations = [n.duration_tick for n in notes]
    velocities = [n.velocity for n in notes]
    return {
        "pitch_range": float(max(pitches) - min(pitches)),
        "mean_duration": float(sum(durations) / len(durations)),
        "mean_velocity": float(sum(velocities) / len(velocities)),
    }


def whole_song_metrics(state: FSNTGV2State) -> Dict[str, float]:
    return {
        "whole_song_span_count": float(state.num_spans),
        "whole_song_note_count": float(sum(1 for v in state.note_attrs["active"] if int(v) == 1)),
    }


def evaluate_generated_state(
    generated: FSNTGV2State,
    rhythm_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    reference: FSNTGV2State | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "ook": ook_rate(generated, pitch_codec),
        "note_density": note_density(generated),
        "duplicate_note_rate": duplicate_note_rate(generated, rhythm_vocab, pitch_codec),
        "invalid_decode_rate": invalid_decode_rate(generated, rhythm_vocab, pitch_codec),
        "voice_leading_large_leap_rate": _voice_leading_large_leap_rate(generated, rhythm_vocab, pitch_codec),
        "phrase_repetition_consistency": phrase_repetition_consistency(generated),
    }
    metrics.update(graph_validity_metrics(generated))
    metrics.update(direct_symbolic_metrics(generated, rhythm_vocab, pitch_codec))
    metrics.update(whole_song_metrics(generated))

    if reference is not None:
        metrics.update(chord_metrics(generated, reference))
        metrics["groove_similarity"] = groove_similarity(generated, reference, rhythm_vocab)
        metrics["span_relation_accuracy"] = span_relation_accuracy(generated, reference)
    return metrics


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    return {
        key: float(sum(row.get(key, 0.0) for row in rows) / len(rows))
        for key in keys
    }
