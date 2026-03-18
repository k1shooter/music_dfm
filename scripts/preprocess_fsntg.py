#!/usr/bin/env python
"""Preprocess POP909 into FSNTG cached splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_graph_dfm.preprocess.pop909_pipeline import preprocess_pop909


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=Path, default=Path("data/raw/POP909-Dataset/POP909"))
    parser.add_argument("--output_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--top_k_per_meter", type=int, default=32)
    parser.add_argument("--onset_bins", type=int, default=32)
    parser.add_argument("--split_seed", type=int, default=7)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--no_nonlocal_span_edges", action="store_true")
    args = parser.parse_args()

    stats = preprocess_pop909(
        data_root=args.raw_root,
        output_root=args.output_root,
        top_k_per_meter=args.top_k_per_meter,
        onset_bins=args.onset_bins,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        include_nonlocal_span_edges=not args.no_nonlocal_span_edges,
    )
    print(stats)


if __name__ == "__main__":
    main()
