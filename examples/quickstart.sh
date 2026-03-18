#!/usr/bin/env bash
set -euo pipefail

uv sync --extra dev
uv run music-graph-dfm download-pop909 --target-dir data/raw/POP909-Dataset
uv run music-graph-dfm preprocess --raw-root data/raw/POP909-Dataset/POP909 --output-root data/cache/pop909_fsntg_v2
uv run music-graph-dfm train --config configs/train/default.yaml
uv run music-graph-dfm train --config configs/train/editflow.yaml
uv run music-graph-dfm sample --checkpoint artifacts/checkpoints/epoch_20.pt --data-root data/cache/pop909_fsntg_v2 --num-samples 8
uv run music-graph-dfm eval --eval-mode checkpoint --checkpoint artifacts/checkpoints/epoch_20.pt --data-root data/cache/pop909_fsntg_v2
uv run music-graph-dfm visualize --sample-dir artifacts/samples --out artifacts/visualization_summary.json
