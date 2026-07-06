# Causal Sensitivity Paper Replica

A PFN Studio project replicating the core method from **Javurek et al., *Amortizing Causal Sensitivity Analysis via Prior-Fitted Networks* (arXiv:2605.10590)**. The goal is to train a transformer-based PFN that amortizes the expensive Lagrangian bound-computation of Marginal Sensitivity Model (MSM) causal sensitivity analysis — replacing per-dataset optimisation with a single forward pass at inference time.

---

## What this PFN does

Given a context of observed `(X, A, Y)` triples and a query `(x, a, Γ)`, the model predicts the **MSM sensitivity bounds θ\*** (lower and upper) on the causal query, without ever observing the hidden confounder `U`. The partial-identification setting mirrors the real-world scenario: `U` is simulated during training to generate ground-truth bound labels, but is withheld from the model at both train and eval time.

---

## Repository layout

```
priors/
  causal_sensitivity_packed/      # Full-scale packed prior (N=1024, D=10, n_lambda=5)
  causal_sensitivity_optimized/   # Faster live prior (N=256, D=10) — main training prior
  causal-sensitivity-msm/         # Early MSM prior (v1)
  causal-sensitivity-msm-v2/      # Studio-compatible packed MSM prior (v2)

models/
  causal-sensitivity-optimized-pfn  # tabular_embedder → transformer(d=128,L=10) → scalar_head
  causal-sensitivity-msm-bounds-v2  # tabular_embedder → transformer(d=256,L=4) → estimation_head
  estimation-stack                  # Prototype estimation stack (d=128)

evals/
  causal-sensitivity-bounds-theta-rmse   # RMSE/MAE/R² on θ* from optimized prior
  causal-sensitivity-bounds-v2           # Regression sweep on packed prior
  causal-sensitivity-bounds-regression-eval
  synthetic-regression-sweep

runs/  (18 total; key completed runs below)
  egret-v1                  ✅ completed — optimized prior + optimized PFN, 50 steps
  sanity-test-v7            ✅ completed — packed prior + optimized PFN, 20 steps
  sanity-v1-msm-bounds-010  ✅ completed — optimized prior + optimized PFN, 50 steps
```

---

## Priors

### `causal_sensitivity_packed` ⭐ (full-scale prior, verified)

Full-scale port of the authors' **Setting_standard** pipeline. Outputs a single packed token matrix consumed directly by a `tabular_embedder`-based model.

**Verified output shapes (smoke-tested):**

| Variable | Shape | Role |
|---|---|---|
| `X` | `(1664, 15)` | packed tokens: 1024 context + 640 query rows |
| `theta_star` | `(640, 1)` | MSM bound labels (target) |
| `n_ctx` | scalar = 1024 | context/query boundary |
| `gamma` | `(640, 1)` | Γ value per query (diagnostic) |
| `bound_type` | `(640, 1)` | 0=upper, 1=lower (diagnostic) |
| `query_id` | `(640, 1)` | which patient (diagnostic) |
| `source_row_index` | `(640, 1)` | context row index (diagnostic) |

**Packed token layout — D+5 = 15 columns (by design):**

| cols 0–9 | col 10 | col 11 | col 12 | col 13 | col 14 |
|---|---|---|---|---|---|
| `x0…x9` | `a` | `y` (0 on query) | `γ` (0 on context) | `bound_flag` | `is_context` |

`is_context=1` on all context rows, `=0` on all query rows. This lets a plain `tabular_embedder` distinguish context from query **without NaNs** — it is the 15th column by deliberate design, not an accident. Forcing 14 columns by dropping it would break the context/query distinction. The trained run `sanity-test-v7` converged (loss 140 → 2.19) with `d_in=15`, confirming correctness.

**M arithmetic:** `M = 4 × n_queries × n_lambda = 4 × 32 × 5 = 640` where the factor 4 = 2 arms × 2 bounds.

**Paper fidelity:**

| Hyperparameter | Paper (Table 2) | This prior | Note |
|---|---|---|---|
| `num_bins` | 16 | **16** ✅ | `_sample_impl` always passed 16; `FlowConfig` default now also 16 |
| `k_train` | 128 | 128 ✅ | — |
| `k_eval` | 4096 | 4096 ✅ | — |
| `steps/λ` | 350 | 350 ✅ | — |
| `tail_bound` | 6.0 | 6.0 ✅ | — |
| `N` | 1024 | 1024 ✅ | — |
| `D` | 10 | 10 ✅ | — |
| `n_lambda` | 50 | **5** ⚠️ | deliberate speed trade-off; documented |
| Spline direction | forward | forward ✅ | verified in running code |
| Token width | — | D+5=15 ✅ | `is_context` column is intentional |

Only genuine deviation: **`n_lambda=5` vs the paper's 50-point grid** — deliberate, documented speed trade-off.

---

### `causal_sensitivity_optimized` ⭐ (main training prior)
A faithful live port of the authors' **Setting_standard** pipeline with smaller settings for faster live sampling:

1. **SCM / DGP** — covariates `X` from a layered random DAG; treatment `A` from MLP propensity `f_A`; outcome `Y = f_BNN(X, A, U)` with hidden confounder `U ~ N(0,1)`.
2. **Queries** — sample context patients, query both arms `(a=0, a=1)`.
3. **Bounds** — authors' Lagrangian sweep: rational-quadratic spline flow over `U`, MSM divergence `γ = max(max r, 1/min r)`, warm-started descending λ-sweep.
4. **Repair** — cumulative monotonicity repair per bound curve.
5. **Return** — packed `X`, `y` (θ\*), `n_ctx`.

### `causal-sensitivity-msm` / `causal-sensitivity-msm-v2`
Earlier MSM priors used during initial development and sanity-testing.

---

## Models

### `causal-sensitivity-optimized-pfn` ⭐ (main model)
| Block | Config |
|---|---|
| `tabular_embedder` | d_model=128, d_in=15 |
| `transformer_encoder` | d_model=128, n_heads=4, n_layers=10, dropout=0.1, ff_mult=4 |
| `scalar_head` | d_out=1 |

Output head: `theta_star` (regression).

### `causal-sensitivity-msm-bounds-v2`
Deeper embedding (d=256), 4-layer transformer, estimation head for direct bound-pair prediction.

---

## Evals

| Slug | Task | Metrics | Prior source |
|---|---|---|---|
| `causal-sensitivity-bounds-theta-rmse` | regression | RMSE, MAE, R² | `causal_sensitivity_optimized` |
| `causal-sensitivity-bounds-v2` | regression | — | `causal_sensitivity_packed` |
| `causal-sensitivity-bounds-regression-eval` | regression | — | — |
| `synthetic-regression-sweep` | regression | — | synthetic draws |

---

## Key completed runs

| Run | Prior | Model | Steps | Result |
|---|---|---|---|---|
| `egret-v1` | `causal_sensitivity_optimized` | `causal-sensitivity-optimized-pfn` | 50 | ✅ completed |
| `sanity-v1-msm-bounds-010` | `causal_sensitivity_optimized` | `causal-sensitivity-optimized-pfn` | 50 | ✅ completed |
| `sanity-test-v7` | `causal_sensitivity_packed` | `causal-sensitivity-optimized-pfn` | 20 | ✅ completed, loss 140→2.19 |

---

## Reproducing a training run

1. Open the **Runs** tab in PFN Studio.
2. Select a completed run as a template, or create a new run wiring prior + model + eval.
3. Set compute target and launch.

**Recommended starting point:**
- Prior: `causal_sensitivity_packed`
- Model: `causal-sensitivity-optimized-pfn`
- Eval: `causal-sensitivity-bounds-theta-rmse`
- Hyperparams: `steps=500, batch_size=16, lr=1e-3`

---

## Reference

> Javurek et al., *Amortizing Causal Sensitivity Analysis via Prior-Fitted Networks*, arXiv:2605.10590.

The prior pipelines in this project are derived from the authors' published method. Numerics (Table 2) are matched where feasible within live sampling constraints.
