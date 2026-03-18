"""Whole-song generation: true long-context and stitching baseline."""

from __future__ import annotations

from typing import List

from music_graph_dfm.constants import SPAN_RELATIONS
from music_graph_dfm.representation.state import FSNTGV2State


def stitch_segments_baseline(segments: List[FSNTGV2State]) -> FSNTGV2State:
    """Baseline: independently generated segments then stitched."""
    if not segments:
        raise ValueError("segments must be non-empty")

    out = segments[0].copy()
    next_rel = SPAN_RELATIONS.index("next")

    for segment in segments[1:]:
        if int(segment.ticks_per_span) != int(out.ticks_per_span):
            raise ValueError("All segments must share ticks_per_span for stitching")

        span_offset = out.num_spans

        for channel in out.span_attrs:
            out.span_attrs[channel].extend(segment.span_attrs[channel])

        for channel in out.note_attrs:
            out.note_attrs[channel].extend(segment.note_attrs[channel])

        out.host.extend([(h + span_offset) if h > 0 else 0 for h in segment.host])
        out.template.extend(segment.template)

        old = out.e_ss
        new_size = out.num_spans + segment.num_spans
        expanded = [[0 for _ in range(new_size)] for _ in range(new_size)]
        for i in range(len(old)):
            for j in range(len(old[i])):
                expanded[i][j] = old[i][j]
        for i in range(segment.num_spans):
            for j in range(segment.num_spans):
                expanded[span_offset + i][span_offset + j] = segment.e_ss[i][j]
        expanded[span_offset - 1][span_offset] = next_rel
        out.e_ss = expanded

        last_start = out.span_starts[-1]
        out.span_starts.extend([last_start + out.ticks_per_span * (k + 1) for k in range(segment.num_spans)])

    out.validate_shapes()
    out.project_placement_consistency()
    out.metadata = dict(out.metadata)
    out.metadata["whole_song_mode"] = "stitching_baseline"
    return out


def build_long_context_template(reference_segments: List[FSNTGV2State]) -> FSNTGV2State:
    """True long-context template: single large graph sampled in one pass."""
    template = stitch_segments_baseline(reference_segments)
    template.metadata = dict(template.metadata)
    template.metadata["whole_song_mode"] = "long_context"
    return template


def generate_whole_song(
    sampled_segments: List[FSNTGV2State],
    mode: str,
) -> FSNTGV2State:
    if mode == "long_context":
        if len(sampled_segments) != 1:
            raise ValueError("long_context mode expects one sampled long context state")
        out = sampled_segments[0].copy()
        out.metadata = dict(out.metadata)
        out.metadata["whole_song_mode"] = "long_context"
        return out
    if mode == "stitching_baseline":
        return stitch_segments_baseline(sampled_segments)
    raise ValueError("mode must be 'long_context' or 'stitching_baseline'")
