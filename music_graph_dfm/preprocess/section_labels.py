"""Optional phrase/section weak labeling and nonlocal span relation heuristics."""

from __future__ import annotations

from typing import List, Tuple


def derive_section_labels(num_spans: int) -> List[int]:
    """Weak section labeling when annotations are unavailable.

    Labels: 0 intro, 1 verse, 2 prechorus, 3 chorus, 4 bridge, 5 outro.
    """
    if num_spans <= 0:
        return []
    labels = [1 for _ in range(num_spans)]
    q = max(1, num_spans // 4)
    for i in range(num_spans):
        if i < q // 2:
            labels[i] = 0
        elif i < 2 * q:
            labels[i] = 1
        elif i < 3 * q:
            labels[i] = 3
        elif i < num_spans - q // 2:
            labels[i] = 4
        else:
            labels[i] = 5
    return labels


def derive_repeat_variation_edges(harm_seq: List[int], min_len: int = 2) -> List[Tuple[int, int, int]]:
    """Heuristic nonlocal span relations from repeated harmonic patterns.

    Returns list of (src, dst, relation_id), where relation_id:
    2 repeat, 3 variation, 4 contrast.
    """
    edges: List[Tuple[int, int, int]] = []
    n = len(harm_seq)
    if n < 2 * min_len:
        return edges

    # Compare short windows and connect starts of similar patterns.
    win = min_len
    for i in range(0, n - win):
        a = harm_seq[i : i + win]
        for j in range(i + win, n - win + 1):
            b = harm_seq[j : j + win]
            matches = sum(1 for x, y in zip(a, b) if x == y)
            ratio = matches / win
            if ratio == 1.0:
                edges.append((i, j, 2))
            elif ratio >= 0.5:
                edges.append((i, j, 3))
            else:
                edges.append((i, j, 4))
    return edges
