# FSNTG Design Note (Math -> Code)

## Representation

State is implemented as a product discrete space:

- `X_S` (span channels): `key, harm, meter, section, reg_center`
- `X_N` (note channels): `active, pitch_token, velocity, role`
- `X_E_NS` (note-span template edges): `none | template_id`
- `X_E_SS` (span relations): `none | next | repeat | variation | contrast | modulation`

Code:

- `music_graph_dfm/data/fsntg.py`
- `music_graph_dfm/templates/rhythm_templates.py`
- `music_graph_dfm/data/pitch_codec.py`

## Deterministic Auxiliary Graph

Local note-note relations are not diffusion coordinates. They are rebuilt from the sampled graph:

- same-onset
- overlap
- sequential within role

Code:

- `reconstruct_aux_graph` in `music_graph_dfm/data/fsntg.py`
- `reconstruct_aux_relations` in `music_graph_dfm/models/hetero_fsntg_transformer.py`

## Forward Path and Scheduling

Default path is coordinate-wise mixture:

\[
q_t^c(x_t^c|x_0^c,x_1^c)=(1-\kappa_c(t))\delta_{x_0^c}+\kappa_c(t)\delta_{x_1^c}
\]

with structure-first order (`span > e_ss > e_ns > note`).

Code:

- `music_graph_dfm/diffusion/schedules.py`
- `music_graph_dfm/diffusion/paths.py`
- `music_graph_dfm/diffusion/state_ops.py`

Optional generalized graph-kernel path hooks are also provided (`paths.py`).

## Reverse Generator / Velocity Parameterization

Model outputs per coordinate channel:

\[
R_t^\theta(X\to X^{c\leftarrow v}) = \lambda_c^\theta(X_t,D(X_t),t)\,\pi_c^\theta(v|X_t,D(X_t),t)
\]

where `lambda` is positive via softplus and `pi` is categorical via softmax.

Code:

- `music_graph_dfm/models/hetero_fsntg_transformer.py`
- separate heads for each channel and edge family

## Loss

Implemented objective:

\[
L = L_{rate} + \beta L_{aux} + \gamma L_{music}
\]

with:

- rate matching (`L_rate`)
- auxiliary denoising CE (`L_aux`)
- music regularizers (`L_music`): host uniqueness, harmonic compatibility, duplicate penalty, voice-leading penalty, repetition consistency

Code:

- `music_graph_dfm/diffusion/losses.py`

## Sampling

Always-valid CTMC jump step:

\[
P(\text{stay}) = e^{-h\lambda_c},\quad
P(\text{jump to }v)=(1-e^{-h\lambda_c})\pi_c(v)
\]

Code:

- `music_graph_dfm/samplers/ctmc_sampler.py`

Post-sampling projections:

- one host per active note
- duplicate note cleanup

Code:

- `project_one_host_per_active_note` and `cleanup_duplicate_notes` in `music_graph_dfm/data/fsntg.py`

## EditFlow Extension

Optional edit CTMC includes:

- insert note + note-span edge
- delete note
- substitute note content
- substitute note-span template
- substitute span-span relation

Code:

- `music_graph_dfm/diffusion/edit_ops.py`
- enabled in `scripts/train_fsntg_editflow.py`

## End-to-End Automation

- Download: `scripts/download_pop909.py`
- Preprocess: `scripts/preprocess_fsntg.py`
- Train DFM: `scripts/train_fsntg_dfm.py`
- Train EditFlow: `scripts/train_fsntg_editflow.py`
- Sample: `scripts/sample_graph.py`, `scripts/sample_whole_song.py`
- Evaluate: `scripts/eval_fsntg.py`
- Vocab stats: `scripts/print_template_vocab_stats.py`
- Visualization: `scripts/visualize_training_example.py`
- Baseline comparison: `scripts/compare_fsntg_vs_flat.py`
