#!/usr/bin/env python
"""Evaluate FSNTG states with symbolic and structural metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_graph_dfm.data.dataset import FSNTGJSONDataset
from music_graph_dfm.eval.metrics import aggregate_metrics, evaluate_state
from music_graph_dfm.utils.io import save_json
from music_graph_dfm.utils.training import load_codecs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--pred_split", type=str, default="test")
    parser.add_argument("--ref_split", type=str, default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("eval_report.json"))
    args = parser.parse_args()

    pred_set = FSNTGJSONDataset(args.data_root / f"{args.pred_split}.jsonl")
    ref_set = FSNTGJSONDataset(args.data_root / f"{args.ref_split}.jsonl")

    rhythm_vocab, pitch_codec = load_codecs(args.data_root)

    n = min(len(pred_set), len(ref_set))
    if args.limit > 0:
        n = min(n, args.limit)

    rows = []
    for i in range(n):
        rows.append(evaluate_state(pred_set[i], rhythm_vocab, pitch_codec, reference_state=ref_set[i]))

    report = {
        "num_examples": n,
        "metrics": aggregate_metrics(rows),
    }
    save_json(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
