# FSNTG-v2 Design Note

## Method Status

Trusted implementation parts:

- FSNTG-v2 factorized state and coordinate set (`S,N,H,Q,E_SS`)
- CTMC reverse sampler with strict off-diagonal jump semantics
- deterministic auxiliary note-note graph from decoded timing only
- harmony-relative pitch-token encode/decode under host span context
- end-to-end evaluation modes (`checkpoint`, `sample-dir`, `reference`)

Approximate / experimental parts:

- graph-kernel path for `span.harm_root` and `note.pitch_token`:
  - target distribution uses kernel-mixture surrogate
  - target rates use off-diagonal Poisson approximation
  - runtime logs and checkpoint/eval metadata explicitly mark this as approximate
- editflow `multistep_segment` mode is experimental:
  - supervision uses sampled trajectory segments from forward edit CTMC trajectories
  - this is a tractable approximation to exact expanded-state marginalization

## Representation

State:

`X = (S, N, H, Q, E_SS)`

- `S`: span channels `key, harm_root, harm_quality, meter, section, reg_center`
- `N`: note channels `active, pitch_token, velocity, role`
- `H`: `host[i] in {0..J}` (`0` means inactive/no-host)
- `Q`: `template[i] in {0..|Q|}` (`0` means inactive/no-template)
- `E_SS`: span-span relation matrix in `{none,next,repeat,variation,contrast,modulation}`

Primary diffusion coordinates are factorized channels, not dense note-span edges.

## Pitch Token (Harmony-Relative)

`pitch_token` is a flattened categorical over:

- `degree_wrt_harmony`
- `role_class` (`chord_tone | scale_tone | chromatic`)
- `register_offset`

Encoding is relative to host-span harmony root (`span.harm_root`), not key tonic.
Decode uses `harm_root` directly for base pitch-class reconstruction and `harm_quality` for chord-tone
compatibility and degree snapping behavior. Reconstruction is not a simple `(key + degree) mod 12` rule.

Provided API:

- `encode_pitch_token(abs_pitch, host_span_state)`
- `decode_pitch_token(token, host_span_state)`
- `compatibility_table(host_span_state, token)`
- `nearest_token_projection(abs_pitch, host_span_state, ...)`
- `PitchTokenCodec.absolute_pitch(...)`

## Rhythmic Templates

Each template stores:

- `onset_bin`
- `duration_class`
- `tie_flag`
- `extension_class`

Decode uses all fields:

`duration = base_duration + tie_bonus + extension_class * ticks_per_span`

## Deterministic Auxiliary Note Graph

Note-note relations are rebuilt from decoded timing only:

- `same_onset`
- `overlap`
- `sequential_same_role`

No template-id arithmetic is used as a structural proxy.

## Forward Path and Target Rates

Default path is coordinate-wise mixture with structure-first scheduling:

1. span channels
2. span relations
3. placement (`host/template`)
4. note content

Source distribution is factorized over coordinates with sparse priors for structure:

- Bernoulli prior on note activation
- sparse non-none prior on span relations
- placement priors (`host/template`) constrained by masks and note activity
- categorical priors for content channels

Graph-kernel mode is optional for `span.harm_root` / `note.pitch_token` and marked approximate.

Approximate target distribution and rate in graph-kernel mode:

- distribution: `q_t=(1-kappa)delta_x0 + kappa*K[x1,:]`
- rate approximation: off-diagonal Poisson matching with `eta*K[x1,v]`, `v != x_t`

## Reverse Generator and CTMC Sampler

Per coordinate:

`R_t^theta(x->x^{c<-v}) = lambda_c * pi_c(v)`

Implementation details:

1. `lambda = softplus(...)`
2. `pi = softmax(...)`
3. zero current category
4. renormalize on `v != current`
5. `p_jump = 1 - exp(-h * lambda)`
6. jump Bernoulli
7. if jump: sample from off-diagonal `pi`

Stay probability is hazard-induced only.
Optional debug assertions can enforce that any sampled jump never lands on the current category.

## Masking Rules

- losses ignore padded coordinates via coordinate masks
- sampling updates ignore padded coordinates
- `host/template` are forced to zero for inactive or padded notes

## Edit-Flow Extension

Separate edit CTMC coordinates:

- insert note
- delete note
- substitute note content
- substitute host
- substitute template
- substitute span relation

Includes:

- edit-state definition
- edit-rate heads (`forward_edit`)
- edit sampler (`sample_edit_ctmc_step`)
- edit training objective (`editflow_rate_loss`)
- forward noising for editflow training via edit-CTMC prior (`sample_forward_edit_ctmc_source`)

This is separate from fixed-slot DFM training. Random edit augmentation remains optional and is not treated as editflow.

Training modes:

- `one_step_oracle` (stable default):
  - source is one forward edit-CTMC step (or optional augmentation)
  - supervise one oracle reverse edit move
- `multistep_segment` (experimental):
  - sample `z_0 -> ... -> z_K` from forward edit CTMC
  - choose adjacent segment `(z_k, z_{k+1})`
  - supervise reverse move from `z_{k+1}` toward `z_k`
  - avoids incorrectly supervising a multi-step-corrupted source with a single direct oracle-to-target move

Sampling modes are also separated:

- one-step edit sampler path
- multistep edit sampler path with micro-steps per time slice

## Decode Projection

After every sampling step:

- padded coordinates are zeroed by masks
- inactive notes force `host=0` and `template=0`
- invalid host indices are projected to inactive

Derived dense note-span adjacency is materialized only when needed from `(host, template)`.

## Evaluation Protocol

Evaluation modes:

1. checkpoint -> generation -> metrics
2. pre-generated sample directory
3. reference-only sanity mode

Reports include:

- checkpoint/sample metadata fields for `editflow_mode`, graph-kernel target-rate mode
- top-level `experimental` flag when approximate graph-kernel or experimental editflow mode is active

Metrics include:

- OOK
- chord accuracy/similarity
- groove similarity
- note density
- host validity / invalid host rate
- duplicate note rate
- invalid decode rate
- voice-leading large-leap penalty
- span relation accuracy
- phrase repetition consistency
- direct symbolic and whole-song metrics
- graph validity metrics

## Structure Loss Efficiency Knobs

Decoded-note semantics remain the source of truth. Runtime controls:

- `train.structure_loss_every_k_steps`
- `train.structure_loss_subsample_notes`
- `train.structure_loss_subsample_pairs`
- `train.fast_music_loss_only`

When full decoded penalties are skipped by cadence, training logs report full decoded-structure evaluation steps.

## Approximations

- graph-kernel target rates are approximate in current implementation
- section/repetition heuristics are rule-based
- harmony compatibility uses a lightweight consonance + key-scale rule
