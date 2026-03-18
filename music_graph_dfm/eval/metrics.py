"""Symbolic and structural evaluation metrics for FSNTG outputs."""

from __future__ import annotations

from typing import Dict, List

from music_graph_dfm.data.fsntg import FSNTGState, decode_notes, reconstruct_aux_graph
from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab


def ook_rate(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> float:
    bad = 0
    total = 0
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        total += 1
        host = next((j for j, e in enumerate(state.e_ns[i]) if e != 0), 0)
        key = int(state.span_attrs["key"][host])
        harm = int(state.span_attrs["harm"][host])
        tok = int(state.note_attrs["pitch_token"][i])
        if not pitch_codec.is_compatible(key, harm, tok):
            bad += 1
    return bad / max(1, total)


def note_density(state: FSNTGState) -> float:
    active = sum(1 for x in state.note_attrs["active"] if int(x) == 1)
    return active / max(1, state.num_spans)


def host_uniqueness_rate(state: FSNTGState) -> float:
    good = 0
    total = 0
    for i in range(state.num_notes):
        if int(state.note_attrs["active"][i]) == 0:
            continue
        total += 1
        non_none = sum(1 for e in state.e_ns[i] if e != 0)
        if non_none == 1:
            good += 1
    return good / max(1, total)


def invalid_edge_rate(state: FSNTGState) -> float:
    total = 0
    invalid = 0
    for i in range(state.num_notes):
        active = int(state.note_attrs["active"][i])
        non_none = sum(1 for e in state.e_ns[i] if e != 0)
        total += 1
        if active == 0 and non_none > 0:
            invalid += 1
    return invalid / max(1, total)


def duplicate_note_rate(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> float:
    notes = decode_notes(state, template_vocab, pitch_codec)
    seen = set()
    dup = 0
    for n in notes:
        key = (n.host_span, n.onset_tick, n.duration_tick, n.pitch, n.role)
        if key in seen:
            dup += 1
        seen.add(key)
    return dup / max(1, len(notes))


def voice_leading_penalty_stats(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> Dict[str, float]:
    aux = reconstruct_aux_graph(state, template_vocab, pitch_codec)
    notes = {n.note_idx: n for n in decode_notes(state, template_vocab, pitch_codec)}
    total = 0
    bad = 0
    for i, j in aux.sequential_same_role:
        if i not in notes or j not in notes:
            continue
        total += 1
        if abs(notes[j].pitch - notes[i].pitch) > 12:
            bad += 1
    return {
        "voice_leading_large_leap_rate": bad / max(1, total),
        "voice_leading_edges": float(total),
    }


def groove_similarity(
    gen_state: FSNTGState,
    ref_state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
) -> float:
    def onset_hist(st: FSNTGState):
        hist = [0 for _ in range(template_vocab.max_onset_bins)]
        for i in range(st.num_notes):
            if int(st.note_attrs["active"][i]) == 0:
                continue
            host = next((j for j, e in enumerate(st.e_ns[i]) if e != 0), 0)
            tpl = st.e_ns[i][host]
            b = template_vocab.decode(tpl).onset_bin
            b = max(0, min(len(hist) - 1, b))
            hist[b] += 1
        s = sum(hist)
        return [h / max(1, s) for h in hist]

    g = onset_hist(gen_state)
    r = onset_hist(ref_state)
    num = sum(a * b for a, b in zip(g, r))
    den = (sum(a * a for a in g) ** 0.5) * (sum(b * b for b in r) ** 0.5)
    return num / max(1e-8, den)


def chord_similarity(gen_state: FSNTGState, ref_state: FSNTGState) -> float:
    n = min(gen_state.num_spans, ref_state.num_spans)
    if n == 0:
        return 0.0
    match = 0
    for i in range(n):
        if (
            int(gen_state.span_attrs["key"][i]) == int(ref_state.span_attrs["key"][i])
            and int(gen_state.span_attrs["harm"][i]) == int(ref_state.span_attrs["harm"][i])
        ):
            match += 1
    return match / n


def span_relation_accuracy(gen_state: FSNTGState, ref_state: FSNTGState) -> float:
    n = min(gen_state.num_spans, ref_state.num_spans)
    if n == 0:
        return 0.0
    total = n * n
    match = 0
    for i in range(n):
        for j in range(n):
            if int(gen_state.e_ss[i][j]) == int(ref_state.e_ss[i][j]):
                match += 1
    return match / max(1, total)


def phrase_repetition_consistency(state: FSNTGState) -> float:
    # repeats/variations should mostly connect spans with same section labels
    pairs = []
    for i in range(state.num_spans):
        for j in range(state.num_spans):
            if state.e_ss[i][j] in (2, 3):
                pairs.append((i, j))
    if not pairs:
        return 1.0
    good = sum(1 for i, j in pairs if state.span_attrs["section"][i] == state.span_attrs["section"][j])
    return good / len(pairs)


def reconstruction_sanity(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
) -> Dict[str, float]:
    notes = decode_notes(state, template_vocab, pitch_codec)
    active = sum(1 for a in state.note_attrs["active"] if int(a) == 1)
    valid_decoded = sum(1 for n in notes if n.duration_tick > 0 and n.onset_tick >= 0)
    return {
        "active_notes": float(active),
        "decoded_notes": float(len(notes)),
        "pct_active_with_valid_decode": valid_decoded / max(1, active),
        "pct_active_with_single_host": host_uniqueness_rate(state),
    }


def evaluate_state(
    state: FSNTGState,
    template_vocab: RhythmTemplateVocab,
    pitch_codec: PitchTokenCodec,
    reference_state: FSNTGState | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "ook": ook_rate(state, template_vocab, pitch_codec),
        "note_density": note_density(state),
        "host_uniqueness": host_uniqueness_rate(state),
        "invalid_edge_rate": invalid_edge_rate(state),
        "duplicate_note_rate": duplicate_note_rate(state, template_vocab, pitch_codec),
        "phrase_repetition_consistency": phrase_repetition_consistency(state),
    }
    metrics.update(voice_leading_penalty_stats(state, template_vocab, pitch_codec))
    metrics.update(reconstruction_sanity(state, template_vocab, pitch_codec))

    if reference_state is not None:
        metrics["groove_similarity"] = groove_similarity(state, reference_state, template_vocab)
        metrics["chord_similarity"] = chord_similarity(state, reference_state)
        metrics["span_relation_accuracy"] = span_relation_accuracy(state, reference_state)
    return metrics


def aggregate_metrics(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({k for row in metric_rows for k in row})
    out = {}
    for k in keys:
        vals = [row[k] for row in metric_rows if k in row]
        out[k] = sum(vals) / max(1, len(vals))
    return out
