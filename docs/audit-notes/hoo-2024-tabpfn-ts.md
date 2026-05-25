# TabPFN-TS audit notes — Hoo et al. 2024

**Paper**: *From Tables to Time: How TabPFN-v2 Outperforms Specialised Time-Series Forecasting Models*
**arXiv**: [2501.02945](https://arxiv.org/abs/2501.02945)
**Catalog entry**: `TABPFN_TS_TEMPLATE` in `apps/web/src/app/projects/templates.ts`
**Wrapper**: `templateId: 'tabpfn-ts'`

## Status: ⚠️ Audited — paper reports GIFT-Eval aggregate, catalog measures per-dataset MASE

After reading the PDF, the paper reports **aggregate scores across 23
GIFT-Eval datasets** (Figure 4.1) — not per-dataset values for M4
monthly specifically. So the catalog's `m4_monthly_mse` eval can't
directly cite a paper scalar; the per-dataset breakdown is in the
GIFT-Eval leaderboard, not in the paper.

## What the paper reports

**Figure 4.1 — GIFT-Eval univariate forecasting (23 datasets):**

| Model | Mean WQL Rank | Relative WQL | Relative MASE |
|---|---|---|---|
| TiRex | 2.515 | 0.413 | 0.642 |
| Toto-1.0 | 4.237 | 0.437 | 0.673 |
| Moirai-2 | 4.464 | 0.436 | 0.654 |
| **TabPFN-TS** | **5.392** | **0.460** | **0.692** |
| TimesFM-2.0 | 5.454 | 0.465 | 0.680 |
| Chronos-Bolt-Base | 5.866 | 0.485 | 0.725 |
| Sundial-Base | 6.402 | 0.472 | 0.673 |
| PatchTST | 6.943 | 0.496 | 0.762 |
| TFT | 7.222 | 0.511 | 0.822 |
| DeepAR | 9.624 | 0.721 | 1.206 |
| Auto-Arima | 10.284 | 0.770 | 0.964 |
| Auto-Theta | 10.923 | 1.051 | 0.978 |
| Seasonal-Naive | 11.675 | 1.000 | 1.000 |

Relative MASE/WQL are **normalized to Seasonal-Naive = 1.000** and
aggregated geometrically across the 23 datasets. So 0.692 means
"on aggregate, TabPFN-TS gets 0.692× the MASE of Seasonal-Naive."

**Figure 4.2 — Covariate-informed forecasting (28 fev-bench tasks
with dynamic covariates):** TabPFN-TS leads at 0.503 / 0.666
(Relative WQL / MASE). As the only model directly incorporating
covariates, it outperforms all others including multivariate
foundation models.

## Why the paper numbers don't slot into the catalog today

The catalog has two evals:

1. **`naive_baseline_mse`** — synthetic series, MSE vs seasonal-naive
   and last-value-carry-forward. Baselines are runtime/analytic.
2. **`m4_monthly_mse`** — real M4-monthly data, absolute MSE/MASE.
   Activates when M4 is downloaded.

The paper's MASE/WQL are **relative to Seasonal-Naive, aggregated
across 23 datasets**. Neither catalog eval matches that aggregation —
the catalog measures *absolute* error on a single benchmark.

Also: the inference path the paper uses is the **pretrained TabPFN-v2
checkpoint** (`2noar4o2`) plus lag + calendar featurization. The
catalog's template trains a from-scratch forecaster on synthetic
prior data — useful as a methodology check, not as a path to match
the paper's GIFT-Eval numbers.

## What would close this gap

Two layered options:

1. **Cheap qualitative win** — add a third eval `seasonal_naive_relative_mase`
   that reports the catalog brain's MASE *divided by* Seasonal-Naive's MASE
   on M4-monthly. Then cite the paper's 0.692 as the
   GIFT-Eval-aggregate target. Verdict: "matches paper's claim if our
   relative-MASE on M4 lands ≤ 1.0." Honest about the aggregation gap.

2. **Full reproduction** — plumb the pretrained TabPFN-v2 checkpoint
   download into the runtime, run the paper's exact inference path on
   each GIFT-Eval dataset, compute aggregate relative-MASE. This is
   the only way to claim "we matched 0.692 on GIFT-Eval." Significant
   plumbing work.

Until one of those lands, the `/benchmarks` row for `tabpfn-ts`
correctly says "computed at training time" because neither eval has
paper-scalar baselines.

## Hyperparameters

`seedRuns[0].hyperparams`: `lr=5e-4`, `batch_size=16`, `steps=5000`.
**These are for the catalog's mirror-training-on-synthetic-data
flow, not for reproducing Hoo's paper numbers.** Reproducing Hoo
2024 doesn't involve training a new PFN — it involves running
inference with the pretrained TabPFN-v2 checkpoint + lag/calendar
featurization. The catalog comment around `templates.ts:2466` is
the canonical documentation of this distinction.

## Notes

- The brain page's "Paper reproduction" card currently compares
  measured score to `baselines[*].score`. For TabPFN-TS that
  comparison is approximate at best because the inference paths
  differ. Per-template verdict language ("matches paper's
  qualitative claim" vs "matches paper Table N to within X%") would
  be a future improvement.
