# music_graph_dfm

FSNTG-v2 (Factorized Span-Note Template Graph v2) for discrete flow matching and graph edit-flow in symbolic music generation.

## Method Status

Trusted:
- FSNTG-v2 primary state: `X = (S, N, H, Q, E_SS)` with diffusion over
  - span channels (`key,harm_root,harm_quality,meter,section,reg_center`)
  - note channels (`active,pitch_token,velocity,role`)
  - `note.host`, `note.template`
  - `e_ss.relation`
- deterministic note-note auxiliary graph from decoded timing only (`same_onset`, `overlap`, `sequential_same_role`)
- CTMC reverse sampler with strict off-diagonal jumps
- script-first preprocessing/training/sampling/evaluation pipeline

Approximate / experimental:
- graph-kernel path for `span.harm_root` and `note.pitch_token` uses an explicit approximation
  - target distribution: `q_t=(1-kappa)delta_x0 + kappa*K[x1,:]`
  - target rate approximation: off-diagonal Poisson matching with `eta*K[x1,v]`
  - this mode logs warnings and is saved in checkpoint/eval metadata
- one-step-oracle editflow supervision only (`editflow_source_steps` must be `1` unless explicitly overriding for experimental work)
- random edit augmentation remains optional and separately named (`editflow_random_augmentation`), and is not core editflow

## Modes

- Baseline DFM:
  - mixture path (default), factorized FSNTG-v2 coordinates, structure-first schedule
- FSNTG-v2 full method in this repo:
  - baseline DFM + decoded-timing aux note graph + music structure penalties
- Experimental graph-kernel mode:
  - optional approximate path for `span.harm_root` / `note.pitch_token`
- One-step-oracle editflow:
  - explicit edit coordinates + edit heads/loss/sampler, trained with one-step oracle supervision

## Repository Layout

```text
repo/
  README.md
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  configs/
  docs/
    design_note.md
    migration.md
    setup.md
  scripts/
    download_pop909.py
    preprocess.py
    train.py
    train_editflow.py
    sample.py
    eval.py
    visualize.py
  src/music_graph_dfm/
  tests/
  examples/
```

## Setup (Primary)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
```

Detailed setup: [docs/setup.md](docs/setup.md)

## Script-First Workflow

### 1) Download POP909

```bash
python scripts/download_pop909.py --target-dir data/raw/POP909-Dataset
```

### 2) Preprocess

```bash
python scripts/preprocess.py \
  --raw-root data/raw/POP909-Dataset/POP909 \
  --output-root data/cache/pop909_fsntg_v2 \
  --span-resolution beat
```

### 3) Train DFM

```bash
python scripts/train.py --config configs/train/default.yaml
```

Structure-loss efficiency knobs are available in config:
- `train.structure_loss_every_k_steps`
- `train.structure_loss_subsample_notes`
- `train.structure_loss_subsample_pairs`
- `train.fast_music_loss_only`

### 4) Train EditFlow

```bash
python scripts/train_editflow.py \
  --config configs/train/editflow.yaml \
  --editflow-source-steps 1
```

Multi-step source noising is rejected by default in training; current method is one-step oracle editflow.

Optional augmentation-only mode (not core editflow algorithm):

```bash
python scripts/train_editflow.py \
  --config configs/train/editflow.yaml \
  --editflow-random-augmentation
```

### 5) Sample

```bash
python scripts/sample.py \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --data-root data/cache/pop909_fsntg_v2 \
  --num-samples 16 \
  --export-midi
```

Whole-song long-context mode:

```bash
python scripts/sample.py \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --whole-song-mode long_context
```

Whole-song stitching baseline:

```bash
python scripts/sample.py \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --whole-song-mode stitching_baseline
```

### 6) Evaluate

Checkpoint mode (`checkpoint -> generate -> decode -> score`):

```bash
python scripts/eval.py \
  --eval-mode checkpoint \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --data-root data/cache/pop909_fsntg_v2
```

Whole-song comparison report (long-context vs stitching baseline):

```bash
python scripts/eval.py \
  --eval-mode checkpoint \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --whole-song-compare
```

Sample-directory mode:

```bash
python scripts/eval.py \
  --eval-mode sample-dir \
  --sample-dir artifacts/samples \
  --data-root data/cache/pop909_fsntg_v2
```

Reference-only sanity mode:

```bash
python scripts/eval.py \
  --eval-mode reference \
  --data-root data/cache/pop909_fsntg_v2
```

### 7) Visualize

```bash
python scripts/visualize.py \
  --sample-dir artifacts/samples \
  --out artifacts/visualization_summary.json
```

## Metrics

Evaluation includes:
- OOK
- chord accuracy / similarity (when reference is available)
- groove similarity
- note density
- host validity (`host_validity`, `invalid_host_rate`)
- duplicate note rate
- invalid decode rate
- voice-leading large-leap rate
- span relation accuracy (when reference is available)
- phrase repetition consistency

## Optional CLI / uv

Console script remains available:

```bash
music-graph-dfm --help
```

Optional uv usage:

```bash
uv sync --extra dev
uv run python scripts/train.py --config configs/train/default.yaml
```

## Tests

```bash
pytest
```

## Notes

- Design note: [docs/design_note.md](docs/design_note.md)
- Migration note: [docs/migration.md](docs/migration.md)
