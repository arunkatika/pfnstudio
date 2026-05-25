# ifBO audit notes — Rakotoarison et al. 2024

**Paper**: *In-Context Freeze-Thaw Bayesian Optimization* (ICML 2024)
**arXiv**: [2404.16795](https://arxiv.org/abs/2404.16795) · HTML mirror: [arxiv.org/html/2404.16795v2](https://arxiv.org/html/2404.16795v2)
**Catalog entry**: `IFBO_TEMPLATE` in `apps/web/src/app/projects/templates.ts`
**Wrapper**: `templateId: 'ifbo'` in `apps/web/src/app/teacher/priors-catalog.ts`

## Source of truth

The paper's Table 1 (Appendix F, headline benchmark) is the canonical
reproduction target. It reports FT-PFN (ifBO's surrogate) against
DyHPO and DPL on three learning-curve datasets: **LCBench**, **PD1**,
**Taskset**. Each row gives log-likelihood + MSE on the unobserved
tail, at varying context-set sizes (50, 100, 250, 500, 1000).

## Values used in the catalog

Population pulled from arxiv HTML Table 1, **LCBench, 1000-sample
context** (the canonical / largest-context row):

| Method | Log-likelihood | MSE |
|---|---|---|
| **FT-PFN (paper's claim)** | **2.044** | **0.004** |
| DyHPO | −0.426 | 0.031 |
| DPL | −11.983 | 0.007 |

Catalog mapping (`seedEvals[0].baselines`):

```ts
{ name: 'FT-PFN target (paper)', score: 0.004,  source: 'Rakotoarison 2024 Table 1 — LCBench, 1000 context, MSE' },
{ name: 'DyHPO',                 score: 0.031,  source: 'Rakotoarison 2024 Table 1 — LCBench, MSE' },
{ name: 'DPL',                   score: 0.007,  source: 'Rakotoarison 2024 Table 1 — LCBench, MSE' },
```

## What's not in the catalog yet

- **PD1 + Taskset rows** — Table 1 has these too. Could be a second
  eval in the catalog if we want multi-dataset reproduction verdicts.
- **Smaller-context rows (50, 100, 250, 500)** — the paper reports
  FT-PFN remains best at every context size. A "context-size sweep"
  eval would be a richer reproduction claim than the single 1000
  number above.
- **Speedup claim** — the paper reports "10× to 100× faster than DPL
  and DyHPO". Not yet wired as a metric.

## Hyperparameter source

`seedRuns[0].hyperparams` in templates.ts cites Appendix A.3:
- `optimizer: adam` · `lr: 1e-4` · `batch_size: 25`
- `steps: 80_000` (approximated from "2M synthetic datasets / batch_size=25"
  — paper doesn't publish exact step count)
- `emsize: 512` · `nlayers: 6` · `nhead: 4` · `hidden: 1024` (14.69M params)
- Cosine annealing + linear warmup over first 25% of epochs

## Open questions for the authors

1. **Exact total step count** — Appendix A.3 says "2M synthetic
   datasets, training: ~8 GPU hours on RTX 2080" but doesn't pin a
   step count. Our 80k is back-computed.
2. **Per-task evaluation seeds** — Table 1 numbers are averaged over
   how many seeds? Not surfaced in the paper text we can fetch.
3. **Verdict tolerance** — "matches the paper" within ±5% MSE? ±10%?
   The paper doesn't specify a tolerance for downstream reproductions.
