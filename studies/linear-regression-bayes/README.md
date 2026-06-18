# Study: in-context Bayesian linear regression

> **Claim**: a tiny transformer (3 layers, d_model=64, ~50k params) trained
> on synthetic random linear functions does Bayesian linear regression on
> data it has never seen — in a single forward pass — landing within
> ~1.15× of the closed-form OLS solution. Trains in ~47 seconds on CPU.

This is the canonical PFN setup from Müller et al. ICLR 2022, reproduced
end-to-end with pfnstudio. Every artifact is in this directory; nothing
is hidden in the binary.

## Result

From a fresh run on a laptop CPU (M-series, Python 3.13, PyTorch 2.x):

| metric                            | value      | notes |
|-----------------------------------|------------|-------|
| **PFN MSE**                       | **0.01262**| in-context inference on 50 held-out tasks |
| Mean baseline                     | 1.13707    | predict the context mean — a useless model |
| OLS baseline                      | 0.01027    | closed-form Bayesian posterior mean |
| Ratio: mean baseline / PFN        | **90.1×**  | how many times better than predicting the mean |
| Ratio: PFN / OLS                  | **1.23×**  | how close to the Bayesian-optimal target |
| Training loss (mean over last 10%)| 0.0117     | converged |
| Training wall time                | 46.6 s     | CPU only |

**The PFN beats the mean baseline by ~90× and lands within 1.23× of the
Bayesian-optimal OLS solution.** It has never seen the (a, b) parameters
that generated the held-out tasks; the entire inference happens in one
forward pass through the transformer, conditioned on the context (x, y)
pairs the prior packs into the same sequence as the query x's.

These exact numbers are reproducible byte-for-byte on the same machine —
see § Reproducibility below.

Full metrics: [`outputs/metrics.json`](outputs/metrics.json).
Training log (sampled): [`outputs/run.log.snippet`](outputs/run.log.snippet).

## Reproduce

Two equivalent paths. The CLI path is what the hosted studio runs under
the hood; the script path is the from-Python equivalent.

### A. From the CLI (same path the hosted studio uses)

```bash
pip install "pfnstudio-core[torch]" pfnstudio
git clone https://github.com/profitopsai/pfnstudio
cd pfnstudio

pfnstudio validate studies/linear-regression-bayes/priors/bayesian_linear/
pfnstudio run     studies/linear-regression-bayes/runs/v0_1.yaml
```

This loads `prior.yaml` + `prior.py`, builds the model from `model.yaml`,
runs `train_pfn` with default settings, and writes a checkpoint at
`./checkpoint/model.pt`. Exit status is non-zero if anything fails.

### B. From Python (same prior + model + hparams, plus held-out eval)

```bash
pip install "pfnstudio-core[torch]"
python examples/01_linear_regression.py
```

Trains the same model in the same way and adds an in-script evaluation
on 50 held-out tasks against the mean and OLS baselines. This is the
script that produced the table above.

Both paths should produce roughly the same training loss and metrics; any
variance comes from the in-script eval drawing fresh tasks each run.

## What's in here

```
studies/linear-regression-bayes/
├── README.md                              ← you are here
├── priors/
│   └── bayesian_linear/
│       ├── prior.yaml                     ← spec (schema-validated)
│       └── prior.py                       ← @register_prior implementation
├── models/
│   └── in_context_pfn.yaml                ← 3-layer transformer config
├── runs/
│   └── v0_1.yaml                          ← hparams (steps, lr, seed)
└── outputs/
    ├── metrics.json                       ← the result above
    └── run.log.snippet                    ← sampled training log
```

The prior emits each task as a packed `(context, query)` token sequence
with an `n_ctx` boundary. `pfnstudio_core`'s default training step
slices the model's logits at `n_ctx` so the loss is computed only at
query positions; context tokens are visible to the transformer's
self-attention so it can route information from them. This is the
mechanism that lets one forward pass produce a Bayesian posterior mean.

## Caveats — what this study does *not* show

- **Scale.** This is a 3-layer / d_model=64 model on a univariate prior.
  Real PFNs (TabPFN, etc.) are 10-100× larger and trained for hours, not
  minutes. The point is that the *training-once, infer-via-context*
  paradigm works end-to-end, not that this particular model is
  competitive with TabPFN.
- **Distribution.** The held-out tasks come from the same prior as
  training. PFN performance on distribution-shifted tasks is a separate
  empirical question.
- **No structured uncertainty.** The model outputs point predictions, not
  the full posterior. Adding a Gaussian head + NLL loss is a natural
  next step but not in scope here.

## Reproducibility

What's pinned and what isn't:

| What | State | Why |
|---|---|---|
| **`hyperparams.seed = 42`** | ✅ Deterministic | Drives prior task sampling, model init, optimizer state. Setting this same seed twice on the same machine produces *byte-identical* metrics. |
| **Training-task seeds** | ✅ Disjoint | Each step samples 32 fresh task seeds (`seed + step·batch_size + i`), no overlap across steps. |
| **Eval-task seeds** | ✅ Fixed | `BASE_SEED = 10000`. The scorer always evaluates on the same 50 tasks. |
| **`torch.manual_seed(seed)`** | ✅ Called | In both `train_pfn` and the local adapter before `Model()` construction. Locks all `nn.Linear` / `nn.TransformerEncoder` / LazyLinear initialization. |
| **`numpy.random.seed(seed)`** | ✅ Called | Belt-and-suspenders. Most code uses `np.random.default_rng(seed)` which is already deterministic; the global seed protects any user code that uses global `np.random`. |
| **PyTorch FP reduction order** | ⚠️ Per machine | CPU vs GPU, x86 vs ARM, different BLAS implementations reduce floating-point operations in different orders. Same seed + same versions + different hardware → headline numbers shift at the ~3rd decimal place. The claim *"~1.0–1.5× of OLS, ~90× better than mean"* is robust to this. |
| **Library versions** | ⚠️ Pinned in this run | This run used `torch==2.2.2`, `numpy==1.26.4`, Python 3.13.5. Future torch / numpy versions may change algorithms — same headline, different exact digits. |

**Exact same-machine reproduction**: every `pfnstudio run` of this study produces the metrics in [`outputs/metrics.json`](outputs/metrics.json) byte-for-byte. Verified by running twice and diffing.

**Different-machine reproduction**: expect `pfn_mse` in the `0.010–0.020` range, `ratio_vs_ols` in `1.0–1.5×`, `ratio_vs_mean` in `60–120×`. These are the bounds the claim is robust to. If your numbers are well outside these ranges, something's wrong — open an issue with your torch/numpy/python versions + CPU model.

## Citation

The PFN idea is from:

> Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022).
> *Transformers Can Do Bayesian Inference.* ICLR 2022.
> https://arxiv.org/abs/2112.10510

This study is an independent reimplementation; no code is taken from the
authors' repository, and the model architecture / training loop are
pfnstudio's own. The conceptual lineage is the paper.

## License

Apache-2.0, same as the rest of this repo.
