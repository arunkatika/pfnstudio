# Study: zero-shot TabPFN-style classification on breast-cancer

> **Claim**: a 6-layer transformer (~700k params) trained on synthetic
> random-hyperplane binary-classification tasks does meaningful
> zero-shot classification on the real breast-cancer (Wisconsin)
> dataset — **83.2% accuracy / 0.87 AUC**, beating the majority
> baseline by 20 points, without ever seeing a single real-data label.

This is a small reproduction in the spirit of [TabPFN](https://arxiv.org/abs/2207.01848)
(Hollmann et al. ICLR 2023). It's not at TabPFN's scale or accuracy — the
prior is intentionally simple (linear decision boundaries; TabPFN uses
mixtures of BNN-generated and SCM-generated tasks), the model is
intentionally small (700k params; TabPFN is ~25M), and the training is
intentionally short (~14 minutes on a laptop CPU). The point is to show
the *in-context-classification-on-real-data* mechanism end-to-end with
artifacts and a recipe you can reproduce in one command.

## Result

From a fresh run on a laptop CPU (M-series, Python 3.13.5, PyTorch 2.2.2):

| metric                           | value     | notes |
|----------------------------------|-----------|-------|
| **PFN accuracy (zero-shot)**     | **0.832** | 30 bootstrap passes, 48-context + 16-query each |
| **PFN AUC**                      | **0.872** | same |
| LogisticRegression baseline      | 0.986     | fit on the full 426-sample train fold |
| LogReg AUC                       | 0.998     | same |
| Majority class baseline          | 0.627     | predict the dominant class blindly |
| In-distribution accuracy (sanity) | 0.770    | 100 fresh tasks from the training prior |
| Training final loss              | 0.439     | BCE-with-logits, query positions only |
| Training wall time               | 815 s     | 13.6 min on CPU |

These numbers are byte-identical across reruns on the same machine
+ same versions — see § Reproducibility below.
| Training wall time               | 862 s     | CPU only, 10k steps |

**The PFN beats the majority baseline by 12 points zero-shot.** LogReg
beats the PFN by 24 points, which is the expected gap when comparing a
synthetic-only prior against a model that fits on the actual labels.
Closing the gap is what richer priors (BNN-generated tasks, SCM
mixtures — TabPFN's actual prior set) and longer training buy you.

Full metrics: [`outputs/metrics.json`](outputs/metrics.json).

## Reproduce

Two equivalent paths (same as `linear-regression-bayes`).

### A. From the CLI (same path the hosted studio uses)

```bash
pip install "pfnstudio-core[torch]" pfnstudio scikit-learn
git clone https://github.com/profitopsai/pfnstudio
cd pfnstudio

pfnstudio validate studies/tabpfn-style-classification/priors/random_binary_classification/
pfnstudio run     studies/tabpfn-style-classification/runs/v0_1.yaml
```

This loads `prior.yaml` + `prior.py`, builds the 6-layer transformer
from `model.yaml`, trains for 10k steps, and runs the
`breast_cancer_vs_logreg` scorer at the end. Total ~15 minutes on CPU.
The scorer reads `sklearn.datasets.load_breast_cancer()` internally, so
no external download is needed.

### B. From the hosted studio

Paste this URL into the "Import from GitHub" panel on the projects page:

```
https://github.com/profitopsai/pfnstudio/tree/main/studies/tabpfn-style-classification
```

The whole study materializes as a workspace project; click Run on the
`v0_1` run and the same training + eval executes via the studio's
runner. The post-run page surfaces the same headline metrics.

## What's in here

```
studies/tabpfn-style-classification/
├── README.md                                          ← you are here
├── priors/
│   └── random_binary_classification/
│       ├── prior.yaml                                 ← spec
│       └── prior.py                                   ← @register_prior class
├── models/
│   └── in_context_classifier.yaml                     ← 6-layer transformer, d_model=128
├── runs/
│   └── v0_1.yaml                                      ← 10k steps, lr=3e-4, batch=32
├── evals/
│   └── breast_cancer_vs_logreg.yaml                   ← eval spec
└── outputs/
    └── metrics.json                                   ← committed result
```

Each prior emits packed (context, query) token sequences with an
`n_ctx` boundary — the same convention as `linear-regression-bayes`.
`pfnstudio_core`'s default training step slices the model's logits at
`n_ctx` so loss is computed only at query positions while the
transformer can still attend to context tokens. The scorer rebuilds
the same token shape from the breast-cancer features (30 columns) +
labels at eval time.

## Bug fixes this study surfaced

Two real `pfnstudio_core` bugs got fixed during this work — both
silently affected every prior in the catalog:

1. **`_default_step` only collected nn.Module parameters via `mod` or
   `mod.module`**, missing the common `mod._linear` / `mod.proj`
   pattern that `TabularEmbedder` and `ScalarHead` use. Embedder + head
   weights stayed at random init for the entire training run. The
   regression studies still converged because random-linear projections
   carry enough signal for low-dim tasks; the high-dim classification
   prior wouldn't.

2. **`train_pfn` passed `seed=base+step` to `prior.sample_batch`**,
   which made consecutive steps share `batch_size-1` seeds. Each task
   was shown ~`batch_size` times in a row and never again, causing
   classic recency overfitting. The fix is `seed=base + step*batch_size`
   so every task is seen exactly once across training.

Both fixes are in [packages/core/pfnstudio_core/training/loop.py](../../packages/core/pfnstudio_core/training/loop.py).

## Caveats — what this study does *not* show

- **Not competitive with TabPFN**. TabPFN's real-data accuracy on
  benchmark tabular datasets is in the same range as XGBoost (often
  ~98% on breast-cancer). This study is ~83%. Closing the gap requires
  richer priors and far more compute.
- **One dataset**. CC18-scale benchmarking (5–18 OpenML datasets) is a
  natural next step; this study is the smallest meaningful proof
  point.
- **30-feature ceiling**. The prior dimensionality is hardcoded to 30
  to match breast-cancer. A real PFN prior generalizes across feature
  counts; ours doesn't.

## Reproducibility

What's pinned and what isn't:

| What | State | Why |
|---|---|---|
| **`hyperparams.seed = 42`** | ✅ Deterministic | Drives prior task sampling, model init, optimizer state. Same seed on the same machine → byte-identical metrics. |
| **Training-task seeds** | ✅ Disjoint | Each step samples 32 fresh task seeds (`seed + step·batch_size + i`), no overlap across steps. Same recipe as `linear-regression-bayes`. |
| **Eval-task seeds (breast-cancer)** | ✅ Fixed | `random_state=42` on the sklearn split, `BOOTSTRAP_SEED=7` on the 30-pass averaging, hard-coded test indices. |
| **`torch.manual_seed(seed)`** | ✅ Called | In both `train_pfn` and the local adapter before `Model()` construction. Locks all `nn.Linear` / `nn.TransformerEncoder` init across the 700k params. |
| **`numpy.random.seed(seed)`** | ✅ Called | Belt-and-suspenders for any code that uses the global numpy RNG. |
| **Sklearn LogReg fit** | ✅ Deterministic | `random_state=42`, full-batch fit on 426 samples. Always 0.986 / AUC 0.998 on this split. |
| **PyTorch FP reduction order** | ⚠️ Per machine | 6 transformer layers × 10000 steps × 30-feature inputs is enough FP noise that cross-hardware reruns shift the 3rd decimal. Same-machine same-version is byte-identical. |
| **Library versions** | ⚠️ Pinned in this run | torch==2.2.2, numpy==1.26.4, scikit-learn (recent), Python 3.13.5. Different versions may shift exact digits. |

**Exact same-machine reproduction**: every `pfnstudio run` produces the metrics in [`outputs/metrics.json`](outputs/metrics.json) byte-for-byte.

**Different-machine reproduction**: expect `pfn_accuracy` in `0.80–0.85`, `pfn_auc` in `0.85–0.90`, `accuracy_gap_vs_logreg` in `-0.12 to -0.18`. The headline claim (zero-shot in-context classification meaningfully beats majority baseline, lags LogReg by ~15 points) holds robustly. If your numbers are well outside these bounds, open an issue with your torch/numpy/sklearn versions + CPU model.

## Citation

The TabPFN paper:

> Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023).
> *TabPFN: A Transformer That Solves Small Tabular Classification
> Problems in a Second.* ICLR 2023.
> https://arxiv.org/abs/2207.01848

Conceptually related (the original PFN paper):

> Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022).
> *Transformers Can Do Bayesian Inference.* ICLR 2022.
> https://arxiv.org/abs/2112.10510

This study is an independent reimplementation; no code is taken from
the TabPFN authors' repository.

## License

Apache-2.0, same as the rest of this repo.
