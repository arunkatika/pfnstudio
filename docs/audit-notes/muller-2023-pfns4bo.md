# PFNs4BO audit notes — Müller et al. 2023

**Paper**: *PFNs4BO: In-Context Learning for Bayesian Optimization* (ICML 2023)
**arXiv**: [2305.17535](https://arxiv.org/abs/2305.17535) · [ICML PDF](https://proceedings.mlr.press/v202/muller23a/muller23a.pdf)
**Catalog entry**: `PFNS4BO_TEMPLATE` in `apps/web/src/app/projects/templates.ts`
**Wrapper**: `templateId: 'pfns4bo'`

## Status: ⚠️ Audited — paper benchmarks BO regret, catalog measures analytical GP posterior

After reading the PDF, the paper's numerical tables (Tables 3, 4, 8)
benchmark **mean rank and mean regret on HPO-B** after 50 BO trials.
The catalog's `gp_posterior_rmse` eval is the **analytical
GP-posterior-match check** — verifying that a PFN trained on the GP-RBF
prior matches the closed-form GP posterior mean.

These are two structurally different eval surfaces:
- Catalog: "Can the PFN approximate a single closed-form GP?"
  Verdict: analytical. No paper-table value applies.
- Paper: "Does PFNs4BO+HEBO+ beat HEBO on HPO-B BO regret?"
  Verdict: tabulated, but requires a different eval suite the catalog
  doesn't ship.

## What the paper reports

**Table 1 — BO with minimal confounding factors:** PFN vs Empirical
Bayes wins/ties/losses. Qualitative only.

**Table 3 — HPO-B (XGBoost, large search spaces), 50 trials:**

| Method | Rank Mean | Regret Mean |
|---|---|---|
| Random | 2.765 | 0.076 |
| HEBO | 2.406 | 0.070 |
| PFN (HEBO+, no ignored features) | 2.639 | 0.103 |
| **PFN (HEBO+, 30% ignored features)** | **2.189** | **0.091** |

**Table 4 — HPO-B test search spaces (all 9 spaces, 51 tasks):**

| Method | Rank Mean | Regret Mean |
|---|---|---|
| Random | 7.658 | 0.125 |
| HEBO | 5.467 | 0.092 |
| GP | 5.539 | 0.075 |
| DNGO | 5.332 | 0.057 |
| DGP | 5.604 | 0.059 |
| **EI (PFN HEBO+ default)** | **4.969** | **0.053** |
| EI(predict mean) | 4.883 | 0.059 |
| UCB (0.95 percentile) | 4.932 | 0.053 |

Also benchmarked on Bayesmark, PD1, and synthetic functions —
detailed numbers in Appendix.

## Why the catalog can't cite these today

The catalog's eval is:

```ts
seedEvals: [{
  slug: 'gp_posterior_rmse',
  metrics: [
    { name: 'rmse_vs_posterior', ... },   // RMSE vs analytic GP mean
    { name: 'rmse_vs_true', ... },        // RMSE vs noise-free GP draw
  ],
  baselines: [
    { name: 'Analytic GP posterior mean', score: 0.0, source: 'analytic' },
    { name: 'Mean baseline', score: 0.0, source: 'analytic' },
  ],
}]
```

The metrics measure how close the PFN's predictions are to a single
closed-form GP — verdict-by-analytic. The paper's metrics
(rank / regret on HPO-B trial sequences) require running a full BO
loop with the PFN as a surrogate. Different evaluation pipeline.

## What would close this gap

To get a paper-cited row on `/benchmarks` for `pfns4bo`:

1. **Add a new eval** `hpob_bo_regret` (or similar) that runs PFN as
   surrogate for HPO-B BO over 50 trials. Requires HPO-B benchmark
   files + a BO loop wrapped around the PFN's predictive distribution.
2. **Add baselines with `metric: 'regret_mean'`**:
   ```ts
   { name: 'PFN (HEBO+) target (paper)', metric: 'regret_mean',
     score: 0.053, source: 'Müller 2023 Table 4 — HPO-B 9 test spaces, 50 trials' },
   { name: 'HEBO', metric: 'regret_mean', score: 0.092,
     source: 'Müller 2023 Table 4' },
   { name: 'DNGO', metric: 'regret_mean', score: 0.057,
     source: 'Müller 2023 Table 4' },
   { name: 'GP', metric: 'regret_mean', score: 0.075,
     source: 'Müller 2023 Table 4' },
   { name: 'Random Search', metric: 'regret_mean', score: 0.125,
     source: 'Müller 2023 Table 4' },
   ```

This is a meaningful addition because HPO-B regret is the paper's
headline claim. The current analytical eval is useful as a sanity
check (does the PFN approximate a GP at all?) but doesn't prove the
paper's actual contribution.

## Hyperparameters

`seedRuns[0].hyperparams` are pulled from the official
[automl/PFNs4BO](https://github.com/automl/PFNs4BO) `config_heboplus`:
- `lr=1e-4`, `batch_size=128`, `steps=51_200` (50 epochs × 1024 steps_per_epoch)
- `warmup_epochs=5`, `weight_decay=0.0`
- `emsize=512`, `nlayers=12` (HEBO+) / 6 (BNN)
- `aggregate_k_gradients=2` (HEBO+) / 1 (BNN)
- Optimizer: Adam, cosine schedule with warmup
