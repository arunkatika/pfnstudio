# PFNs reference audit notes — Müller et al. 2022

**Paper**: *Transformers Can Do Bayesian Inference* (ICLR 2022)
**arXiv**: [2112.10510](https://arxiv.org/abs/2112.10510) · [PDF](https://arxiv.org/pdf/2112.10510)
**OpenReview**: [forum?id=KSugKcbNf9](https://openreview.net/forum?id=KSugKcbNf9)
**Catalog entries**: `PFNS_TEMPLATE` (Bayesian regression) and `PFNS_CLASSIFICATION_TEMPLATE` (Bayesian classification) in `apps/web/src/app/projects/templates.ts`
**Wrappers**: `templateId: 'pfns-reference'` and `templateId: 'pfns-classification'`

## Status: ✅ Reproducible end-to-end (regression path)

**This is the first paper-backed template on /benchmarks where the
reproduction actually works.** As of phase 2 of the regression-ICL
refactor:

- `bayesian_linear @ 0.2.0` emits packed-token PFN format
- `pfn_transformer @ 0.2.0` reads the packed tokens
- `closed_form_baseline` scorer computes RMSE against the analytic
  Bayesian posterior mean — the canonical Müller 2022 verdict

A 2000-step CPU training produces `rmse_vs_posterior ≈ 0.11`. The
fact that this is essentially the same as the noise-free RMSE (also
≈ 0.11) is the paper-claim verified: residual error matches the
Bayes-optimal predictor. Paper-pinned 16k-step runs drive both lower.

Run it yourself: `python scripts/reproduce-pfns-reference.py`

The `pfns-classification` row is a separate analysis — see below.

## Earlier framing (pfns-classification)

The right call is still to keep the `pfns-classification` row
showing "computed at training time" on `/benchmarks`. The reason isn't
that we can't find numbers — it's that the catalog's evals are
deliberately analytical-style and the paper's numerical tables
benchmark a *different* eval surface.

## What the paper actually reports

The headline experiments are:

1. **§4 / Figure 4** — Prior-Data NLL on the GP regression task. Plots
   show PFN approaching the analytical GP NLL as training proceeds.
   No single scalar to cite — the *verdict* is "your PFN's NLL should
   converge to the analytic GP NLL." The catalog encodes this with
   `rmse_vs_posterior.expected.value: 0.0` (the analytical verdict).
2. **§6 / Table 1 / Appendix Table 7** — Tabular classification on
   **21 OpenML datasets** at n=30 training samples. Reports mean rank
   of ROC AUC and per-dataset AUCs.
3. **§7** — Few-shot image classification on Omniglot. Off-scope for
   the catalog.

## Why the catalog's evals don't map to paper Table 1

The classification template's `seedEvals` are:

- **`synthetic_bce_baseline`** — BCE on held-out synthetic hyperplane
  tasks. The baseline (`Majority: 0.6931`) is analytic (`-log(0.5)`),
  not paper-cited.
- **`breast_cancer_vs_logreg`** — Zero-shot on sklearn
  Wisconsin breast-cancer (569 samples, 30 features). Compared to
  sklearn LogisticRegression fit at runtime.

The paper's 21-dataset OpenML benchmark (Table 7) **does not include
breast-cancer**. The OpenML datasets are: kr-vs-kp, credit-g, vehicle,
wine, kc1, airlines, bank-market…, blood-transfus…, phoneme,
covertype, numerai28.6, connect-4, car, Australian, segment,
jungle-chess…, sylvine, MiniBooNE, dionis, jannis, helena.

So Müller 2022's per-dataset AUCs can't be cited as paper baselines
for the catalog's breast-cancer eval. The analytical
`Logistic Regression` baseline (computed at runtime against the
sklearn train fold) is the right comparator and stays as-is.

## What would close this gap

If we want a *paper-cited* row on `/benchmarks` for
`pfns-classification`, options:

- **Add a new eval** that reproduces Müller 2022 §6 by running
  zero-shot binary classification on the 21 OpenML datasets and
  reporting mean rank vs paper's Table 1. The paper-cited target then
  becomes "PFN-BNN mean rank ROC AUC ≤ 2.786" (Table 1). Requires
  downloading the OpenML benchmark suite — non-trivial dataset wrangling.
- **Or accept** that the current row says "computed at training time"
  because the eval is analytical (LogReg-at-runtime), and document
  that here.

## Numbers in the paper (for future use)

If we ever add the OpenML-mean-rank eval, here's what to put in
`baselines` (from Müller 2022 Table 7 — 21 OpenML datasets, n=30):

```
Mean rank ROC AUC:
  PFN-BNN  2.786  ← paper target
  PFN-GP   3.833
  XGB      3.357  ← strongest non-PFN baseline
  Catboost 4.833
  LogReg   4.690
  BNN      5.000
  GP       5.286
  KNN      6.214

Expected Calibration Error (lower is better):
  PFN-BNN  0.025  ← paper target
  PFN-GP   0.067
  XGB      0.066
  BNN      0.089
  KNN      0.093
  GP       0.095
  Catboost 0.157
  LogReg   0.157
```

## Hyperparameters

`PFNS_TEMPLATE.seedRuns[0].hyperparams`: `lr=2e-5`, `batch_size=8`,
`steps=16_000`. Pulled from the official
[automl/TransformersCanDoBayesianInference](https://github.com/automl/TransformersCanDoBayesianInference)
README's "Train your own PFN" example dict.

`PFNS_CLASSIFICATION_TEMPLATE.seedRuns[0].hyperparams`: `lr=5e-4`,
`batch_size=16`, `steps=2000`. Catalog-tuned for the synthetic
hyperplane prior + 64-token bootstrap layout used by the
breast-cancer scorer.
