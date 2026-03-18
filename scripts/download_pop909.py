#!/usr/bin/env python
"""Download POP909 dataset repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_graph_dfm.preprocess.pop909_pipeline import download_pop909


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=Path, default=Path("data/raw/POP909-Dataset"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = download_pop909(args.target_dir, force=args.force)
    print(f"POP909 downloaded/available at: {out}")


if __name__ == "__main__":
    main()
