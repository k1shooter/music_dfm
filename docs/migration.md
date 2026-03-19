# Migration Note (FSNTG-v1 Dense E_NS -> FSNTG-v2 Host/Template)

## Audit: Doc/Code Mismatch Treated As Bug

Before the refactor, the repository had inconsistent definitions across docs and code.
Those mismatches were treated as implementation bugs, not documentation style issues.

| Area | Before | After |
|---|---|---|
| README representation | `X=(S,N,E_NS,E_SS)` | `X=(S,N,H,Q,E_SS)` |
| Design note representation | `X=(S,N,H,Q,E_SS)` | kept, aligned to implemented code |
| Primary model/data state | Dense `E_NS` note-span matrix | factorized `host` + `template` coordinates |
| Model aux note graph | template-id arithmetic proxies | deterministic from decoded onset/duration |
| CTMC jump semantics | jump could sample current category | strict off-diagonal jump destinations |
| "editflow" mode | random augmentation behavior | edit-CTMC with edit-rate heads and edit objective |

## Representation Migration

Old primary diffusion coordinates included a dense note-span tensor.
That representation made multi-host inconsistencies possible in-state.

New FSNTG-v2 coordinates are:

- `span.*` channels
- `note.*` channels
- `note.host`
- `note.template`
- `e_ss.relation`

Dense note-span adjacency is now a derived view only:

- `materialize_dense_note_span_view(state)`

## Sampler Migration

Old CTMC behavior allowed re-sampling current category after jump decision.

New CTMC behavior:

1. `lam = softplus(...)`
2. `pi = softmax(logits)`
3. remove current category probability
4. renormalize off-diagonal support
5. `p_jump = 1 - exp(-h * lam)`
6. jump samples only from off-diagonal support

## EditFlow Migration

Old "editflow" mode perturbed targets and reused non-edit training behavior.

New edit mode uses explicit edit coordinates and rate heads:

- insert note
- delete note
- substitute content
- substitute host
- substitute template
- substitute span relation

Training/sampling use edit-CTMC transitions and edit losses.

## Training API Migration

Previous training entrypoint mixed DFM and editflow logic in one loop.

New explicit entrypoints:

- `run_training_dfm(cfg)`
- `run_training_editflow(cfg)`
- `run_training(cfg)` dispatcher

Checkpoints now save model/data/config metadata for reproducible sampling/eval.

## Package and CLI Migration

Old workflow relied on root-package imports and editable install assumptions.

New workflow uses `src/` layout and package entrypoint:

- `uv sync --extra dev`
- `uv run music-graph-dfm download-pop909 ...`
- `uv run music-graph-dfm preprocess ...`
- `uv run music-graph-dfm train ...`
- `uv run music-graph-dfm sample ...`
- `uv run music-graph-dfm eval ...`
- `uv run music-graph-dfm visualize ...` (or `viz`)
