# Changelog

Versions are per-package; tags are `core-v<version>` and `cli-v<version>`.

## pfnstudio-core 0.9.1

### Fixed
- **Trainer support for the axial-attention blocks + distributional heads.**
  0.9.0 shipped the blocks (`grid_preprocessor`, `axial_attention_block`, …) and
  a bar-distribution head pattern, but the training/predict loops didn't feed
  them the plumbing they need, so a study using them failed with
  "grid_preprocessor requires single_eval_pos > 0". The default step and the
  predict forward now:
  - thread **`single_eval_pos`** (the train/test boundary from the prior's
    `n_ctx`) to blocks declaring `needs_single_eval_pos`, and unwrap
    `(out, kv)` tuples;
  - run the generic **`setup(*, prior, hp, device)`** hook before training (so a
    bar-distribution head fits its bucket borders from the prior);
  - honor a head's **`loss(query_output, target)`** hook (bucketized NLL)
    before falling back to MSE.

## pfnstudio 0.8.14

### Added
- **Runner per-job core sync (opt-in)** — a self-hosted runner can refresh
  `pfnstudio-core` before each job so it picks up newer core blocks (e.g. the
  axial-attention library) instead of failing with "No block registered".
  Enable when launching the runner:
  - `PFNSTUDIO_RUNNER_SYNC_CORE=1` — `pip install -U --no-deps pfnstudio-core`
  - `PFNSTUDIO_RUNNER_CORE_SPEC=<spec>` — upgrade an exact spec (pinned version
    or a git URL) instead.
  `--no-deps` keeps it fast; a failed sync is non-fatal (the job runs on the
  installed core).

## pfnstudio-core 0.9.0

Headline: the **axial-attention block library** and the **scorer registry** are
now public, so paper-faithful studies (Do-PFN, cascade discovery) run and
reproduce entirely from open packages.

### Added
- **Axial-attention block library** (`blocks/grid.py` + `_grid_preprocessing.py`):
  `grid_preprocessor`, `tabular_cell_embedder`, `along_row_attention`,
  `along_column_attention`, `axial_attention_block`, `row_pool_for_head`, and
  the `TabularPreprocessor` they wrap. Registered on import.
- **Scorer registry** — `@register_scorer(slug)` / `get_scorer(slug)`. Studies
  ship their own eval scorers (`evals/<slug>.py`) discovered at run time;
  `pfnstudio_core.scorers.BUILTIN_SCORERS` remains for the generic scorers.
- **Generic distributional-head hooks** — a head block can declare
  `is_head = True` (so the trainer fans it out without a hardcoded allowlist)
  and expose `to_prediction(output)` (so the predict path reduces per-bucket
  logits to a point estimate). Composes with the existing `setup()` / `loss()`
  training hooks — enough to build a bar-distribution head as project code.
- **Project-block discovery** — `discover_in_project` imports `blocks/*.py`;
  `get_block` tolerates `-`/`_` slug variants.

### Fixed
- **Multi-submodule checkpoints** — a block holding more than one `nn.Module`
  (e.g. a gated residual with both an `mlp` and a `gate`) no longer collides in
  the checkpoint. Submodules are namespaced by attribute; single-module blocks
  keep the legacy flat keying, so existing checkpoints still load.

### Notes
- `networkx` is **not** a core dependency. Studies that need it (the Do-PFN
  random-DAG prior) carry it in `priors/<slug>/requirements.txt`.

## pfnstudio 0.8.13

### Changed
- Eval scoring resolves a project's `@register_scorer` (loaded by
  `discover_in_project`) **before** falling back to core's builtins, so a
  study's own scorer wins.
- Requires `pfnstudio-core>=0.9.0`.
