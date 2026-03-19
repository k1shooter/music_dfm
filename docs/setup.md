# Setup

## Primary Workflow (venv + pip)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

Run scripts directly:

```bash
python scripts/download_pop909.py --help
python scripts/preprocess.py --help
python scripts/train.py --help
python scripts/train_editflow.py --help
python scripts/sample.py --help
python scripts/eval.py --help
python scripts/visualize.py --help
```

Typical execution sequence:

```bash
python scripts/preprocess.py --raw-root data/raw/POP909-Dataset/POP909 --output-root data/cache/pop909_fsntg_v2
python scripts/train.py --config configs/train/default.yaml
python scripts/train_editflow.py --config configs/train/editflow.yaml --editflow-mode one_step_oracle
python scripts/sample.py --checkpoint artifacts/checkpoints/epoch_20.pt --data-root data/cache/pop909_fsntg_v2
python scripts/eval.py --eval-mode checkpoint --checkpoint artifacts/checkpoints/epoch_20.pt --data-root data/cache/pop909_fsntg_v2
```

Run tests:

```bash
pytest
```

## Optional uv Workflow

```bash
uv sync --extra dev
uv run python scripts/train.py --config configs/train/default.yaml
```

`uv` is optional. The repository workflow is script-first via `python scripts/*.py`.
