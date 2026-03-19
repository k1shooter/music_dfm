#!/usr/bin/env python
"""Preprocess POP909 into FSNTG-v2 cache format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_graph_dfm.preprocessing import PreprocessConfig, preprocess_pop909


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess POP909 to FSNTG-v2 cache")
    parser.add_argument("--raw-root", type=str, default="data/raw/POP909-Dataset/POP909")
    parser.add_argument("--output-root", type=str, default="data/cache/pop909_fsntg_v2")
    parser.add_argument("--span-resolution", type=str, choices=["beat", "half_bar", "bar"], default="beat")
    parser.add_argument("--top-k-per-meter", type=int, default=64)
    parser.add_argument("--onset-bins", type=int, default=8)
    parser.add_argument("--max-extension-class", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--min-notes-per-song", type=int, default=16)
    args = parser.parse_args()

    cfg = PreprocessConfig(
        raw_root=str(Path(args.raw_root).expanduser().resolve()),
        output_root=str(Path(args.output_root).expanduser().resolve()),
        span_resolution=args.span_resolution,
        top_k_per_meter=args.top_k_per_meter,
        onset_bins=args.onset_bins,
        max_extension_class=args.max_extension_class,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        min_notes_per_song=args.min_notes_per_song,
    )
    stats = preprocess_pop909(cfg)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
