# Migration Note (Old -> FSNTG-v2)

## Structural Changes

- Package moved to `src/music_graph_dfm`.
- CLI moved to one entry point: `music-graph-dfm`.
- Editable install is no longer required.

## Representation Changes

Old state used dense note-span edge tensor as primary diffusion state.

New state uses:

- `note.host`
- `note.template`

Dense note-span adjacency is now derived when needed.

## Sampler Changes

Old CTMC logic could resample current category on jump.

New sampler excludes current category from jump targets and uses hazard-induced stay probability.

## EditFlow Changes

Old edit mode was random augmentation.

New edit mode uses explicit edit coordinates, edit-rate heads, CTMC edit sampling, and editflow loss.

## Evaluation Changes

Old evaluation compared cached splits only.

New evaluation supports:

- generate from checkpoint then evaluate
- evaluate pre-generated sample directory
- reference-only sanity mode

## Command Changes

Old:

- multiple `scripts/*.py`
- hydra config path coupling

New:

- `uv run music-graph-dfm download-pop909 ...`
- `uv run music-graph-dfm preprocess ...`
- `uv run music-graph-dfm train ...`
- `uv run music-graph-dfm sample ...`
- `uv run music-graph-dfm eval ...`
