# Migration Note: Second-Pass Corrections

This note records the targeted correction pass from partially aligned FSNTG-v2 code to the current implementation.

## Old Mismatch -> New Fix

| Area | Old mismatch | New fix |
|---|---|---|
| Docs vs code status | README/design note could overstate exactness | Method status now separates trusted vs approximate parts |
| Harmonic span semantics | Single `harm` channel mixed root/function semantics | Span channels split into `harm_root` + `harm_quality`, propagated through preprocessing/state/model/loss/eval |
| Pitch token semantics | API and usage were mixed between component-level and context-level semantics | Primary API is now host-context encode/decode (`abs_pitch <-> token`) with compatibility/projection helpers |
| CTMC sampler | Degenerate off-diagonal mass could still trigger non-faithful behavior | Degenerate case now forces stay; masked/padded coordinates always stay |
| Editflow training source process | Multi-step-corrupted sources were supervised with single-step oracle targets | Two explicit modes: stable `one_step_oracle` and experimental `multistep_segment` trajectory supervision |
| Graph-kernel path | Approximation not fully explicit in metadata/reporting | Approximation formula and coordinates are stored in checkpoint/eval metadata; warnings are logged |
| Developer UX | uv-first workflow in docs/examples | Script-first `python scripts/*.py` workflow with venv+pip primary |

## File-by-File Change Summary

- `README.md`
  - switched to script-first workflow
  - added explicit method-status and approximation notes
  - documented whole-song evaluation separation
- `docs/design_note.md`
  - added **Method Status** section
  - clarified graph-kernel target distribution/rate approximation
  - clarified editflow source process and augmentation separation
- `docs/setup.md` (new)
  - venv + pip setup and script-first commands
  - uv moved to optional section
- `docs/migration.md` (this file)
  - recorded mismatch->fix mapping and limitations
- `requirements.txt`, `requirements-dev.txt` (new)
  - pip-installable dependency entrypoints
- `scripts/*.py` (new)
  - direct developer entrypoints:
    - `download_pop909.py`
    - `preprocess.py`
    - `train.py`
    - `train_editflow.py`
    - `sample.py`
    - `eval.py`
    - `visualize.py`
- `src/music_graph_dfm/representation/pitch_codec.py`
  - context-aware primary API:
    - `encode_pitch_token(abs_pitch, host_span_state)`
    - `decode_pitch_token(token, host_span_state)`
    - `compatibility_table(host_span_state, token)`
    - `nearest_token_projection(...)`
  - decode uses both `harm_root` and `harm_quality` semantics
- `src/music_graph_dfm/preprocessing/pop909.py`
  - preprocessing now stores `harm_root` + `harm_quality` span channels and host-context pitch tokens
- `src/music_graph_dfm/preprocessing/chords.py`
  - chord parsing now exports harmony root and quality IDs
- `src/music_graph_dfm/constants.py`
  - `SPAN_CHANNELS` and coordinate order updated to `harm_root` + `harm_quality`
  - graph-kernel approximate coordinates now reference `span.harm_root`
- `src/music_graph_dfm/diffusion/ctmc.py`
  - strict off-diagonal normalization with explicit degenerate stay behavior
- `src/music_graph_dfm/diffusion/edit_flow.py`
  - added forward edit-CTMC source sampling utilities
  - added forward edit-CTMC trajectory sampling and trajectory-segment supervision helper
  - renamed augmentation utility (`random_edit_augmentation_step`)
  - strengthened edit sampler validity and off-diagonal substitutions
  - fixed edit loss argument handling for span relation and insert content heads
  - edit loss now skips no-op samples instead of forcing invalid type targets
- `src/music_graph_dfm/models/hetero_transformer.py`
  - added insert-content heads (`insert_pitch/velocity/role`)
- `src/music_graph_dfm/models/simple_baseline.py`
  - added insert-content heads to keep editflow API consistent
- `src/music_graph_dfm/diffusion/state_ops.py`
  - graph-kernel usage metadata and warning emission
- `src/music_graph_dfm/diffusion/paths.py`
  - normalized graph-kernel targets and explicit target-rate approximation helper
- `src/music_graph_dfm/diffusion/losses.py`
  - graph-kernel target rate approximation wired explicitly for supported coords
  - decoded structure-loss cadence/subsampling knobs added (`structure_loss_every_k_steps`, note/pair subsampling, fast mode)
- `src/music_graph_dfm/training/runner.py`
  - editflow mode is explicit (`one_step_oracle` or `multistep_segment`)
  - multistep mode trains with trajectory-segment supervision over adjacent CTMC states
  - optional augmentation path preserved under explicit flag
  - graph-kernel approximation metadata added to checkpoints with loud warnings
  - editflow mode/experimental metadata saved in checkpoints and used by sampling paths
  - decoded structure-loss execution cadence logged and recorded in summaries
- `src/music_graph_dfm/evaluation/pipeline.py`
  - checkpoint eval now stores checkpoint metadata in report
  - report includes explicit `experimental` flag and editflow mode metadata
  - optional MIDI export in checkpoint eval mode
  - whole-song mode propagated through checkpoint evaluation
- Compatibility wrappers added for expected paths:
  - `src/music_graph_dfm/train_runner.py`
  - `src/music_graph_dfm/data/pitch_codec.py`
  - `src/music_graph_dfm/models/hetero_fsntg_transformer.py`
  - `src/music_graph_dfm/samplers/ctmc_sampler.py`
  - `src/music_graph_dfm/samplers/edit_sampler.py`
- Tests added/updated:
  - `tests/test_pitch_codec.py`
  - `tests/test_state_roundtrip.py`
  - `tests/test_ctmc_sampler.py`
  - `tests/test_editflow.py`
  - `tests/test_graph_kernel_sanity.py`
  - `tests/test_scripts_smoke.py`
  - updated `tests/test_checkpoint_metadata.py`

## Remaining Approximations / Limitations

- Graph-kernel path remains approximate by design in this codebase.
- Experimental multistep editflow currently uses trajectory-segment supervision (not exact full expanded-state marginalization).
- Harmony compatibility scoring is lightweight rule-based (not a learned music-theory model).
- Section/repetition heuristics remain deterministic and rule-based.
