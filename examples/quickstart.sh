#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/download_pop909.py --target-dir data/raw/POP909-Dataset
python scripts/preprocess.py --raw-root data/raw/POP909-Dataset/POP909 --output-root data/cache/pop909_fsntg_v2
python scripts/train.py --config configs/train/default.yaml
python scripts/train_editflow.py --config configs/train/editflow.yaml
python scripts/sample.py --checkpoint artifacts/checkpoints/epoch_20.pt --data-root data/cache/pop909_fsntg_v2 --num-samples 8
python scripts/eval.py --eval-mode checkpoint --checkpoint artifacts/checkpoints/epoch_20.pt --data-root data/cache/pop909_fsntg_v2
python scripts/visualize.py --sample-dir artifacts/samples --out artifacts/visualization_summary.json
