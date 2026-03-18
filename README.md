# music_graph_dfm

FSNTG-v2 (Factorized Span-Note Template Graph v2) discrete flow matching and edit-flow for symbolic music generation.

## What Changed

This refactor enforces the factorized state:

- `X = (S, N, H, Q, E_SS)`
- span channels: `key, harm, meter, section, reg_center`
- note channels: `active, pitch_token, velocity, role`
- note placement channels: `host` and `template`
- span relations: `none,next,repeat,variation,contrast,modulation`

Dense note-span adjacency is no longer the diffusion state. It is a deterministic derived view from `(host, template)`.

## Repository Structure

```
repo/
  pyproject.toml
  uv.lock
  README.md
  src/music_graph_dfm/
    __init__.py
    cli/
    config/
    data/
    preprocessing/
    representation/
    models/
    diffusion/
    samplers/
    guidance/
    evaluation/
    visualization/
    utils/
  configs/
  docs/
  tests/
  examples/
```

## Install

```bash
uv sync --extra dev
```

No editable install is required.

## CLI

All commands are available via the console script:

```bash
uv run music-graph-dfm --help
```

### 1) Download POP909

```bash
uv run music-graph-dfm download-pop909 \
  --target-dir data/raw/POP909-Dataset
```

### 2) Preprocess (beat-level default)

```bash
uv run music-graph-dfm preprocess \
  --raw-root data/raw/POP909-Dataset/POP909 \
  --output-root data/cache/pop909_fsntg_v2 \
  --span-resolution beat
```

Artifacts:

- `train.jsonl`, `valid.jsonl`, `test.jsonl`
- `rhythm_templates.json`
- `pitch_codec.json`
- `stats.json`
- `preprocessing_config.json`

### 3) Train (DFM)

```bash
uv run music-graph-dfm train --config configs/train/default.yaml
```

### 4) Train (EditFlow)

```bash
uv run music-graph-dfm train --config configs/train/editflow.yaml
```

Posterior/progress-like baselines:

```bash
uv run music-graph-dfm train --config configs/train/posterior.yaml
uv run music-graph-dfm train --config configs/train/progress_like.yaml
```

### 5) Sample

```bash
uv run music-graph-dfm sample \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --data-root data/cache/pop909_fsntg_v2 \
  --num-samples 16 \
  --export-midi
```

EditFlow sampler:

```bash
uv run music-graph-dfm sample \
  --checkpoint artifacts/checkpoints_editflow/epoch_20.pt \
  --sampler-mode editflow
```

Whole-song modes are explicit:

- true long-context (single pass):

```bash
uv run music-graph-dfm sample \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --whole-song-mode long_context
```

- stitching baseline:

```bash
uv run music-graph-dfm sample \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --whole-song-mode stitching_baseline
```

### 6) Evaluate generated samples

From checkpoint (generation included):

```bash
uv run music-graph-dfm eval \
  --eval-mode checkpoint \
  --checkpoint artifacts/checkpoints/epoch_20.pt \
  --sampler-mode dfm \
  --data-root data/cache/pop909_fsntg_v2
```

From pre-generated sample directory:

```bash
uv run music-graph-dfm eval \
  --eval-mode sample-dir \
  --sample-dir artifacts/samples \
  --data-root data/cache/pop909_fsntg_v2
```

Reference-only sanity mode:

```bash
uv run music-graph-dfm eval \
  --eval-mode reference \
  --data-root data/cache/pop909_fsntg_v2
```

### 7) Visualize sample directory

```bash
uv run music-graph-dfm visualize \
  --sample-dir artifacts/samples \
  --out artifacts/visualization_summary.json
```

## Tests

```bash
uv run pytest
```

## Design and Migration Notes

- [Design note](docs/design_note.md)
- [Migration note](docs/migration.md)
