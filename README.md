# music_graph_dfm

Factorized Span-Note Template Graph (FSNTG) discrete flow matching for symbolic music generation.

## What This Project Implements

- Factorized graph state: `X=(S, N, E_NS, E_SS)`
- Span channels: `key, harm, meter, section, reg_center`
- Note channels: `active, pitch_token, velocity, role`
- Note-span edge labels: `none | rhythmic_template_id`
- Span-span relation labels: `none | next | repeat | variation | contrast | modulation`
- Deterministic auxiliary note-note graph reconstruction (`same_onset`, `overlap`, `sequential_same_role`)
- Velocity-parameterized CTMC reverse generator with rate matching + auxiliary denoising + music structure losses
- Optional EditFlow graph-edit operations

## Project Layout

```
music_graph_dfm/
  configs/
  music_graph_dfm/
    data/
    preprocess/
    templates/
    models/
    diffusion/
    samplers/
    guidance/
    eval/
    viz/
    utils/
  scripts/
  tests/
  docs/
```

## Install

```bash
cd /home/dmsdmswns/music_graph_dfm
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e .[dev,pyg,wandb,audio]
```

## Data Automation (POP909)

1. Download dataset repo:

```bash
python scripts/download_pop909.py --target_dir data/raw/POP909-Dataset
```

2. Preprocess to FSNTG caches:

```bash
python scripts/preprocess_fsntg.py \
  --raw_root data/raw/POP909-Dataset/POP909 \
  --output_root data/cache/pop909_fsntg \
  --top_k_per_meter 32 \
  --onset_bins 32
```

3. Print discovered rhythmic template vocabulary stats:

```bash
python scripts/print_template_vocab_stats.py --data_root data/cache/pop909_fsntg --split train
```

## Training

### Full method (FSNTG velocity)

```bash
python scripts/train_fsntg_dfm.py \
  model=fsntg_velocity \
  diffusion=mixture \
  data.cache_root=data/cache/pop909_fsntg
```

### Baseline mode (`progress_like`)

```bash
python scripts/train_fsntg_dfm.py \
  model=progress_like \
  diffusion=mixture \
  data.cache_root=data/cache/pop909_fsntg
```

### EditFlow mode

```bash
python scripts/train_fsntg_editflow.py \
  model=fsntg_velocity \
  train=editflow \
  data.cache_root=data/cache/pop909_fsntg
```

## Sampling

Sample a graph and export graph plot + MIDI + piano-roll:

```bash
python scripts/sample_graph.py \
  --data_root data/cache/pop909_fsntg \
  --checkpoint checkpoints/epoch_40.pt \
  --split test \
  --index 0 \
  --out_dir samples
```

Whole-song generation:

```bash
python scripts/sample_whole_song.py \
  --data_root data/cache/pop909_fsntg \
  --checkpoint checkpoints/epoch_40.pt \
  --mode direct \
  --out_dir samples/whole_song
```

Segment stitching whole-song generation:

```bash
python scripts/sample_whole_song.py \
  --data_root data/cache/pop909_fsntg \
  --checkpoint checkpoints/epoch_40.pt \
  --mode stitch \
  --segments 4 \
  --out_dir samples/whole_song_stitch
```

## Evaluation

```bash
python scripts/eval_fsntg.py \
  --data_root data/cache/pop909_fsntg \
  --pred_split test \
  --ref_split test \
  --out eval_report.json
```

Implemented metric families:

- Symbolic: OOK, groove similarity, chord similarity, note density
- Graph validity: host uniqueness, invalid edge rate, duplicate note rate
- Structure: phrase repetition consistency, span relation accuracy
- Reconstruction sanity: active-note host validity and decode validity
- Voice-leading penalty statistics

## Visualization Utilities

Visualize one training example as graph + decoded score:

```bash
python scripts/visualize_training_example.py \
  --data_root data/cache/pop909_fsntg \
  --split train \
  --index 0 \
  --out_dir viz_examples
```

## FSNTG vs Flat Baseline Comparison

```bash
python scripts/compare_fsntg_vs_flat.py --data_root data/cache/pop909_fsntg
```

## Tests

```bash
pytest
```

Covered tests:

- graph extraction roundtrip
- pitch token encode/decode
- template encode/decode
- host projection
- deterministic auxiliary note-note reconstruction
- CTMC validity (shape/non-negative)
- edit op consistency

## Config Modes

- `baseline_graph_mode=progress_like`
  - fixed/weak structure assumptions
  - mixture path
  - posterior-like setup
- `full_graph_mode=fsntg_velocity`
  - factorized channels + separate heads
  - structure-first schedule
  - velocity parameterization
  - optional edit ops and optional graph-kernel path

See [`docs/design_note.md`](docs/design_note.md) for math-to-code mapping.
