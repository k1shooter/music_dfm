#!/usr/bin/env python
"""Summarize generated sample directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_graph_dfm.visualization import visualize_sample_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sample directory")
    parser.add_argument("--sample-dir", type=str, default="artifacts/samples")
    parser.add_argument("--out", type=str, default="artifacts/visualization_summary.json")
    args = parser.parse_args()

    out = visualize_sample_directory(
        sample_dir=Path(args.sample_dir).expanduser().resolve(),
        out_path=Path(args.out).expanduser().resolve(),
    )
    print(json.dumps({"summary": str(out)}, indent=2))


if __name__ == "__main__":
    main()
