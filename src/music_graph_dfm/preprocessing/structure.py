"""Section and repetition heuristics for span-level structure graph."""

from __future__ import annotations

from typing import List

from music_graph_dfm.constants import E_SS_NONE, SPAN_RELATIONS


def derive_section_labels(num_spans: int, section_span: int = 16) -> List[int]:
    section_span = max(1, int(section_span))
    return [idx // section_span for idx in range(num_spans)]


def derive_span_relation_matrix(harm_roots: List[int], section_labels: List[int]) -> List[List[int]]:
    num_spans = len(harm_roots)
    relation = [[E_SS_NONE for _ in range(num_spans)] for _ in range(num_spans)]

    next_rel = SPAN_RELATIONS.index("next")
    repeat_rel = SPAN_RELATIONS.index("repeat")
    variation_rel = SPAN_RELATIONS.index("variation")
    contrast_rel = SPAN_RELATIONS.index("contrast")
    modulation_rel = SPAN_RELATIONS.index("modulation")

    for i in range(num_spans - 1):
        relation[i][i + 1] = next_rel

    for i in range(num_spans):
        for j in range(i + 1, num_spans):
            delta = (harm_roots[j] - harm_roots[i]) % 12
            if section_labels[i] == section_labels[j] and harm_roots[i] == harm_roots[j]:
                relation[i][j] = repeat_rel
            elif section_labels[i] == section_labels[j] and delta in {2, 10}:
                relation[i][j] = variation_rel
            elif section_labels[i] != section_labels[j] and delta in {5, 7}:
                relation[i][j] = modulation_rel
            elif section_labels[i] != section_labels[j] and harm_roots[i] != harm_roots[j]:
                relation[i][j] = contrast_rel

    return relation
