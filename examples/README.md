# Examples

Each script is **self-contained** — clone the repo, install
`pfnstudio-core[torch]`, run the file, get a result. No notebook
state, no out-of-band setup.

## What's here today

These run in order — each is a small step up in commitment.

### `00_hello_pfn.py` — does the install work?

The smallest possible PFN Studio program. Defines a prior, samples
one task, prints the result. Runs in under a second. **No PyTorch
needed** — nothing trains here. Run this first after
``pip install pfnstudio-core`` to confirm the library installs and
that you understand what a prior produces.

```bash
pip install pfnstudio-core
python examples/00_hello_pfn.py
```

### `01_linear_regression.py` — the hello-PFN (regression)

Train a 2-layer transformer on a Bayesian linear-regression prior in
~5 minutes on CPU. Compares the trained model to a closed-form OLS
baseline. The smallest end-to-end demonstration that the PFN
training paradigm works.

```bash
pip install pfnstudio-core[torch]
python examples/01_linear_regression.py
```

Expected output: the PFN beats the mean baseline by 5–20×, and lands
within ~2× of OLS. At this scale and step count it won't *match* OLS —
the published paper trains substantially longer with larger models —
but the in-context-regression signal is unambiguous.

### `02_two_moons.py` — the hello-PFN (classification)

Same training shape as `01_`, different target: each task samples a
fresh random two-moons geometry and the PFN learns to classify any new
2-D point as belonging to moon A or moon B. Compares against a
majority-class baseline and a closed-form logistic-regression baseline
(which is *bounded above* by the linear decision boundary it can
express, since two-moons is non-linear).

```bash
python examples/02_two_moons.py
```

Expected output: PFN accuracy above the majority baseline (so it's
using the inputs at all) and ideally above logistic regression (so
it's picking up the non-linear geometry — though at this scale that's
not guaranteed; the printout is explicit either way).

## What's planned

Notebooks and scripts we'd like in here. Open a PR if any of these
look approachable — they're all reasonable first contributions.

- `03_sine_wave_forecasting.py` — time-series PFN that forecasts new
  sine waves from a few in-context observations.
- `04_chain_scm_discovery.py` — discovery PFN that recovers the
  directed graph X → Y → Z from random linear SCM samples.
- `05_kinpfn_demo.py` — probabilistic PFN that predicts a CDF of
  first-passage times from a Weibull-mixture prior. Paper-flavoured
  but small enough for CPU.
- `06_load_pretrained.py` — load a checkpoint produced by the studio
  and run inference locally without re-training.

Notebook variants of the above (`.ipynb`) are also welcome — the
scripts are the source of truth so notebooks should be
auto-generatable from them via `jupytext`.

## Conventions

- **One file = one concept.** No multi-step tutorials split across
  files; if it's a tutorial, put it in `docs/`.
- **Runs on CPU in under 10 minutes.** GPU-required examples live in
  the cloud studio, not here.
- **Self-contained imports.** No `sys.path` hacks; install the
  packages and the example just works.
- **Honest output.** If the demo's metric is bad on this small scale,
  the script's final print explains why and points at what would close
  the gap (more steps, bigger model, real data, etc.).
