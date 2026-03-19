#!/usr/bin/env python
"""Train edit-flow model (one_step_oracle or multistep_expanded)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_graph_dfm.config import load_yaml
from music_graph_dfm.training.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EditFlow model (CTMC-prior source noising)")
    parser.add_argument("--config", type=str, default="configs/train/editflow.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--editflow-mode",
        type=str,
        default="one_step_oracle",
        choices=["one_step_oracle", "multistep_expanded"],
    )
    parser.add_argument("--editflow-source-steps", type=int, default=1)
    parser.add_argument("--allow-multistep-oracle", action="store_true")
    parser.add_argument("--editflow-random-augmentation", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cfg.setdefault("train", {})["mode"] = "editflow"
    cfg["train"]["editflow_mode"] = str(args.editflow_mode)
    cfg["train"]["editflow_source_steps"] = int(args.editflow_source_steps)
    cfg["train"]["allow_multistep_oracle"] = bool(args.allow_multistep_oracle)
    cfg["train"]["editflow_random_augmentation"] = bool(args.editflow_random_augmentation)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.device:
        cfg["device"] = args.device

    result = run_training(cfg)
    print(json.dumps(result.get("final", {}), indent=2))


if __name__ == "__main__":
    main()
