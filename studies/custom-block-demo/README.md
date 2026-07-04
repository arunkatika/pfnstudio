# Custom block demo

A minimal example of **defining your own architecture block** and using it in a
model that actually **trains and scores** — the block is *project code*, exactly
like a prior or an eval. Nothing is added to `pfnstudio-core`.

## What's here

```
priors/bayesian_linear/         in-context linear regression prior (y = a·x + b + ε)
blocks/gated_residual.py         ← the custom block: @register_block("gated_residual")
blocks/gated_residual.yaml       optional metadata (name, config fields for the composer)
models/demo_model.yaml           embedder → encoder → gated_residual → scalar_head
evals/in_context_regression_ols.yaml   PFN vs closed-form OLS (real scored metric)
runs/sanity_run.yaml             a quick sanity run (CPU, 500 steps, ~10s)
```

## Run it

It trains **and** produces a real number: launch `sanity_run` from the Runs
tab. It draws in-context linear-regression tasks from the prior, trains the
model (with your custom block in the stack), and the `in_context_regression_ols`
eval scores the trained PFN against closed-form OLS — the Bayesian-optimal
predictor for this prior. The bundled run is a **sanity run** (500 steps, ~10s)
that proves the whole path assembles, trains, and scores end-to-end; it won't
fully converge. Bump `steps` toward ~4000 for a PFN that lands within ~1.5× of
OLS.

## The idea

A **block** is one stage of a model. The built-ins (embedders, encoders, pools,
heads) cover most needs — but when you need something new, you define it in the
project instead of editing core:

1. Write a class decorated with `@register_block("<type>")` in `blocks/<slug>.py`.
   It takes config kwargs in `__init__` and maps a `(B, N, d_model)` tensor to
   `(B, N, d_model)` in `__call__`. Build `torch` submodules as attributes —
   they're collected for training automatically. Import `torch` *inside* the
   methods.
2. Reference it from a model by that `type` (see `models/demo_model.yaml`, the
   `gated_residual` block between the encoder and the head).
3. At run time, `discover_in_project` imports `blocks/*.py`, `@register_block`
   registers the type, and the model assembles + trains with your block.

`blocks/gated_residual.py` is heavily commented — copy its shape for your own
block. Edit it right in the Files tab, or ask Plip to build one.

## Beyond forward-pass blocks

`gated_residual` is a plain forward-pass block, so it needs nothing special. A
block that trains with a **non-standard loss** or needs **pre-training setup**
(e.g. a distributional head that computes bucket borders from the prior) can
additionally implement two optional hooks the training loop calls generically:

- `setup(self, *, prior, hp, device)` — prepare before the first forward pass.
- `loss(self, query_output, target)` — own the loss instead of the default MSE.

Both are opt-in and duck-typed; a block without them behaves exactly as here.
