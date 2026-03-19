#!/usr/bin/env python
"""Train fixed-slot DFM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_graph_dfm.config import load_yaml
from music_graph_dfm.training.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DFM model")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cfg.setdefault("train", {})["mode"] = "dfm"
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.device:
        cfg["device"] = args.device

    result = run_training(cfg)
    print(json.dumps(result.get("final", {}), indent=2))


if __name__ == "__main__":
    main()
