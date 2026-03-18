#!/usr/bin/env python
"""Compare FSNTG modeling against a flat note-tuple baseline."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from music_graph_dfm.data.dataset import FSNTGJSONDataset, to_note_tuple_baseline_batch
from music_graph_dfm.eval.metrics import aggregate_metrics, evaluate_state
from music_graph_dfm.utils.training import load_codecs


def flat_baseline_pitch_ce(train_rows, test_rows) -> float:
    # unigram baseline on pitch_token conditioned by role
    role_pitch = {}
    role_total = Counter()
    for r in train_rows:
        role = r[3]
        pitch = r[1]
        role_pitch.setdefault(role, Counter())[pitch] += 1
        role_total[role] += 1

    import math

    ce = 0.0
    count = 0
    for r in test_rows:
        role = r[3]
        pitch = r[1]
        dist = role_pitch.get(role)
        if not dist:
            prob = 1e-8
        else:
            prob = dist[pitch] / max(1, role_total[role])
            prob = max(prob, 1e-8)
        ce += -math.log(prob)
        count += 1
    return ce / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    train = FSNTGJSONDataset(args.data_root / "train.jsonl")
    test = FSNTGJSONDataset(args.data_root / "test.jsonl")
    rhythm_vocab, pitch_codec = load_codecs(args.data_root)

    n = len(test) if args.limit <= 0 else min(len(test), args.limit)
    fsntg_rows = [evaluate_state(test[i], rhythm_vocab, pitch_codec, reference_state=test[i]) for i in range(n)]
    fsntg_metrics = aggregate_metrics(fsntg_rows)

    train_tuples = to_note_tuple_baseline_batch(train.states)["note_tuples"]
    test_tuples = to_note_tuple_baseline_batch(test.states[:n])["note_tuples"]
    flat_ce = flat_baseline_pitch_ce(train_tuples, test_tuples)

    print("=== FSNTG vs Flat Note-Tuple Baseline ===")
    print({"fsntg_metrics": fsntg_metrics, "flat_pitch_cross_entropy": flat_ce})


if __name__ == "__main__":
    main()
