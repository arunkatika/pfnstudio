# Causal Sensitivity Analysis PFN Studio Project

This repository contains a PFN Studio project for **causal sensitivity analysis using Prior-Fitted Networks (PFNs)**.

The implementation is based on the paper **“Amortizing Causal Sensitivity Analysis via Prior-Fitted Networks”** by Javurek et al. The goal is to train a transformer-based PFN that can predict causal sensitivity bounds from observed data, replacing expensive per-query optimization with one model forward pass after training.

---

## What this project does

The project generates synthetic causal datasets from a structural causal model and trains a PFN to predict causal sensitivity bounds.

For each generated dataset, the prior creates:

- observed covariates `X`
- treatment `A`
- outcome `Y`
- query treatment arms
- sensitivity parameter `gamma` / `Γ`
- upper and lower causal sensitivity bound targets

The hidden confounder is used inside the synthetic data-generating process and optimization procedure, but it is **not given to the model as an input**. This matches the real causal sensitivity setting, where hidden confounding is unknown at inference time.

---

## Repository structure

```text
casual_sensitivity_analysis/
├── README.md
├── ROADMAP.md
├── priors/
│   └── causal_sensitivity_packed/
│       ├── prior.py
│       ├── prior.yaml
│       └── requirements.txt
├── models/
│   └── causal-sensitivity-optimized-pfn.yaml
├── evals/
│   └── causal-sensitivity-bounds-v2.yaml
└── runs/
    └── sanity-test-v7.yaml
```

### Important note

The zip file also contains macOS metadata files such as `.DS_Store` and `__MACOSX/`. These are not part of the actual PFN Studio project and can be safely ignored or removed before sharing the repository.

---

## Main components

### 1. Prior

Location:

```text
priors/causal_sensitivity_packed/
```

The prior is implemented in:

```text
priors/causal_sensitivity_packed/prior.py
```

It generates a packed PFN input sequence containing both context rows and query rows.

The default configuration is:

| Parameter | Value |
|---|---:|
| `D` | 10 |
| `N` | 1024 |
| `n_queries` | 32 |
| `n_lambda` | 5 |

The prior computes:

```text
M = 4 × n_queries × n_lambda
```

With the default values:

```text
M = 4 × 32 × 5 = 640
```

So the packed input has:

```text
N + M = 1024 + 640 = 1664 rows
```

---

## Packed input format

The model receives one packed matrix called `X`.

The packed input shape is:

```text
X: (N + M, D + 5)
```

With the default values:

```text
X: (1664, 15)
```

Each row has this structure:

| Columns | Meaning |
|---|---|
| `0` to `D-1` | Covariates `X` |
| `D` | Treatment `A` |
| `D+1` | Outcome `Y` for context rows, `0` for query rows |
| `D+2` | Gamma `Γ` for query rows, `0` for context rows |
| `D+3` | Bound flag: `0 = upper`, `1 = lower` |
| `D+4` | Context flag: `1 = context row`, `0 = query row` |

For `D = 10`, the columns are:

```text
0-9   : covariates
10    : treatment
11    : outcome
12    : gamma
13    : bound flag
14    : context flag
```

The context flag is important because the model needs to distinguish observed context rows from query rows.

---

## Prior outputs

The prior returns the following values:

| Output | Shape | Purpose |
|---|---|---|
| `X` | `(N + M, D + 5)` | Packed context and query matrix |
| `theta_star` | `(M, 1)` | Target causal sensitivity bound |
| `n_ctx` | scalar | Number of context rows |
| `gamma` | `(M, 1)` | Gamma value for each query row |
| `bound_type` | `(M, 1)` | Upper/lower bound indicator |
| `query_id` | `(M, 1)` | Query group identifier |
| `source_row_index` | `(M, 1)` | Source context row used for the query |

`theta_star` is the supervised training target.

---

## Model

Location:

```text
models/causal-sensitivity-optimized-pfn.yaml
```

The model uses the following architecture:

```text
tabular_embedder
    ↓
transformer_encoder
    ↓
scalar_head
```

Model configuration:

| Component | Configuration |
|---|---|
| Tabular embedder | `d_model = 128` |
| Transformer encoder | `d_model = 128` |
| Attention heads | `4` |
| Transformer layers | `10` |
| Dropout | `0.1` |
| Feed-forward multiplier | `4` |
| Output task | Regression |

The model is trained to predict `theta_star`.

---

## Evaluation

Location:

```text
evals/causal-sensitivity-bounds-v2.yaml
```

The evaluation is configured as a regression task using the packed prior.

Metrics:

- RMSE
- MAE
- R²

---

## Training run

Location:

```text
runs/sanity-test-v7.yaml
```

This run connects the prior, model, and eval configuration.

Run configuration:

| Setting | Value |
|---|---:|
| Learning rate | `0.001` |
| Batch size | `16` |
| Steps | `20` |
| Eval every | `10` |
| Seed | `42` |
| Compute target | Remote |

Recorded run result:

| Result | Value |
|---|---:|
| Status | `ok` |
| Input dimension | `15` |
| Steps completed | `20` |
| Final loss | `3.0138` |
| Mean first 10% loss | `140.6173` |
| Mean last 10% loss | `2.1901` |

This confirms that the packed 15-column input format was accepted by the PFN Studio training pipeline.

---

## How to use this project in PFN Studio

1. Upload or open the repository in PFN Studio.
2. Register the prior from:

```text
priors/causal_sensitivity_packed/prior.yaml
```

3. Register the model from:

```text
models/causal-sensitivity-optimized-pfn.yaml
```

4. Register the eval from:

```text
evals/causal-sensitivity-bounds-v2.yaml
```

5. Launch the run from:

```text
runs/sanity-test-v7.yaml
```

---

## Recommended next steps

For a stronger final experiment, increase the training steps beyond the sanity test.

Recommended starting point:

```yaml
steps: 500
batch_size: 16
lr: 0.001
eval_every: 50
```

For a more paper-faithful grid, increase:

```yaml
n_lambda: 50
```

However, this will significantly increase sequence length and GPU memory usage.

---

## Known implementation note

The implementation uses `n_lambda = 5` in the current default prior configuration. The paper uses a larger lambda grid. This project uses the smaller value as a practical speed and memory trade-off for PFN Studio training.

---

## Reference

Javurek et al., **“Amortizing Causal Sensitivity Analysis via Prior-Fitted Networks”**, arXiv:2605.10590.
