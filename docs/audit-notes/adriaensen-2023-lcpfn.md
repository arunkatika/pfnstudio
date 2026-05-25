# LC-PFN audit notes — Adriaensen et al. 2023

**Paper**: *Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks* (NeurIPS 2023)
**arXiv**: [2310.20447](https://arxiv.org/abs/2310.20447)
**Catalog entry**: `LC_PFN_TEMPLATE` in `apps/web/src/app/projects/templates.ts`
**Wrapper**: `templateId: 'lc-pfn'`

## Status: ⚠️ Audited — paper reports log-likelihood, catalog measures MSE

After reading the PDF, the paper's headline numbers in **Table 3** are
in **log-likelihood (LL)** on synthetic prior curves. The catalog's
`extrapolation_mse` eval reports **MSE** on synthetic prior curves —
same dataset family, different unit. So the paper's absolute numbers
don't slot into the existing metric without adding `log_likelihood`
to the eval (which requires runtime scorer support).

Per-dataset MSE on real benchmarks (LCBench / NAS-Bench-201 / Taskset
/ PD1) appears in the paper only as **average ranks** in Figure 4 and
in distribution plots in Appendix Figures 11–12. No single scalar
MSE value is reported.

## What the paper reports

**Table 3 — Synthetic prior curves, LL by cutoff (higher better):**

| Variant | 10% | 20% | 40% | 80% | Runtime |
|---|---|---|---|---|---|
| M1 (MCMC, Domhan original) | 1.628 | 1.939 | 2.265 | 2.469 | 54.4 s |
| M2 (MCMC, thinned) | 1.641 | 1.958 | 2.277 | 2.477 | 45.2 s |
| M3 (MCMC best) | 1.642 | 1.956 | 2.285 | 2.486 | 103.2 s |
| P1 (PFN, 3-layer, 128) | 1.58 | 1.99 | 2.28 | 2.43 | 0.004 s |
| P2 (PFN, 3-layer, 256) | 1.65 | 2.04 | 2.35 | 2.49 | 0.006 s |
| **P3 (PFN, 12-layer, 512)** | **1.76** | **2.13** | **2.40** | **2.52** | **0.050 s** |

P3 is the headline LC-PFN configuration — 26M params, the variant
that matches the catalog's `lc_pfn_transformer` (12 layers, emsize
512). P3 beats the best MCMC variant (M3) on all cutoffs while being
~2000× faster.

**Real-data benchmarks (Section 4.2 / Figure 4):** LC-PFN P3 vs MCMC
variants on LCBench, NAS-Bench-201, Taskset, PD1 (5 000 curves each).
LC-PFN is never truly outperformed by MCMC-PP. Only **average ranks**
are reported as scalars — there is no per-benchmark MSE/LL table in
the paper.

## How this maps to the catalog today

`LC_PFN_TEMPLATE.seedEvals[0]` (`extrapolation_mse`):
- Metrics: `mse_tail`, `mse_full` — both MSE units.
- Baselines: `Pow3 MLE fit`, `Last-value carry-forward` — both
  analytical/runtime, score 0.0.

The catalog's MSE metric and the paper's LL metric are different
units; no direct paper-cited value applies. Current
"computed at training time" framing on `/benchmarks` is honest.

## What would close this gap

To get a paper-cited row on `/benchmarks` for `lc-pfn`:

1. **Add `log_likelihood` to the eval's metrics array**.
2. **Wire the runtime scorer to compute LL** of the held-out tail
   under the PFN's predictive distribution.
3. **Add baselines with `metric: 'log_likelihood'`**:
   ```ts
   { name: 'LC-PFN target (P3, paper)', metric: 'log_likelihood',
     score: 1.76, source: 'Adriaensen 2023 Table 3 — synthetic prior, 10% cutoff, LL' },
   { name: 'MCMC-PP best (M3)', metric: 'log_likelihood',
     score: 1.642, source: 'Adriaensen 2023 Table 3 — 10% cutoff' },
   { name: 'MCMC-PP Domhan original (M1)', metric: 'log_likelihood',
     score: 1.628, source: 'Adriaensen 2023 Table 3 — 10% cutoff' },
   ```

Choose cutoff to match the catalog's eval setup. The eval description
says "first 50% of each curve" → use the paper's 40% column (nearest
reported), giving P3=2.40, M3=2.285.

## Hyperparameters

`seedRuns[0].hyperparams`: `lr=1e-4`, `batch_size=100`,
`steps=100_000` (1000 epochs × 100 steps_per_epoch), `emsize=512`,
`nlayers=12`, `nhead=4`, `warmup_epochs=250`. Pulled verbatim from
[automl/lcpfn](https://github.com/automl/lcpfn) →
`lcpfn/train_lcpfn.py` defaults. Matches paper's P3 configuration.
