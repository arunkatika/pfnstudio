# Study: Do-PFN — in-context interventional outcome prediction (Robertson et al., 2025)

> **Claim**: a single transformer pre-trained on synthetic SCMs learns
> to predict the conditional interventional distribution
> `p(Y | do(T=t), X)` from *observational data alone*, without ever
> seeing the underlying causal graph. The model beats naive
> observational baselines on Conditional Average Treatment Effect
> (CATE) estimation under unobserved confounding.

Reproducible port of [Robertson, Reuter, Guo, Hollmann, Hutter,
Schölkopf — *Do-PFN: In-Context Learning for Causal Effect
Estimation*](https://arxiv.org/abs/2506.06039), NeurIPS 2025,
reimplemented end-to-end in PFN Studio. Every artifact is in this
directory; nothing is hidden in the binary.

## Status

| What | State |
|---|---|
| Prior — `do_pfn_scm` (MLP-SCM with confounded propensity + outcome) | ✅ Faithful reimplementation from the paper (no upstream code copied) |
| Model — 6-layer Do-PFN (d=192) | ✅ Uses PFN Studio's `transformer_encoder` + `scalar_head` |
| Run — paper-pinned hyperparams | ✅ `lr=5e-4, bs=32, AdamW, wd=1e-5, clip=1.0, steps=12000` |
| Eval — CID + CATE recovery vs naive + oracle | ✅ |
| Results table | ⏳ Pending first end-to-end run on H100/H200 |
| v0.2 — real-data benchmarks (IHDP, Twins) | 🗺️ Roadmapped |

## What this study does

1. **Draws synthetic SCM tasks**: each task is a fresh random SCM with
   - X ∈ ℝ¹⁰ drawn from N(0, Σ) with random block correlation
   - Unobserved confounder U coupled into both the propensity and the outcome
   - Binary treatment T ~ Bernoulli(propensity(X, U))
   - Continuous outcome Y(X, T, U) + ε via a random 2-layer MLP
2. **Packs each task as a (context, query) sequence**:
   - Context tokens carry the **observational** distribution `(X, T_obs, Y_obs)`
   - Query tokens carry the **desired intervention** `(X, t_query, 0)`
3. **Trains the model to predict the interventional outcome**
   `Y_{do(T = t_query)}` at each query position — drawn cleanly from
   the SCM under the do-operation, never visible in the observational
   context.
4. **Evaluates** on 50 fresh DGPs, scoring:
   - **CID-MSE**: predicted Y vs oracle Y_int per query
   - **CATE-MSE**: derived `pred Y do(1) - pred Y do(0)` vs latent CATE
   - **Baselines**: per-task ridge regression (the confounded
     observational estimator) and the latent CATE (Monte-Carlo floor)

## The "aha" moment

The naive observational estimator `E[Y|T=1,X] - E[Y|T=0,X]` is biased
under unobserved confounding — that's a textbook causal-inference
fact. The Do-PFN architecture, trained only on synthetic SCMs with
random unobserved confounders, learns to *systematically correct for
that bias* in-context. One forward pass per new dataset; no
per-instance optimization; no access to the causal graph.

This is closely related to the [`causal-sensitivity-pfn`](../causal-sensitivity-pfn/)
study (Javurek et al.), but Do-PFN predicts the **full conditional
interventional distribution** rather than just the bound — CATE falls
out of paired CID queries on the same X.

## Result

| metric | value | notes |
|---|---|---|
| **PFN CID-MSE** | _pending_ | predicted Y vs oracle Y_int across queries |
| **PFN CATE-MSE** | _pending_ | derived CATE vs latent CATE |
| Naive (per-task ridge) CATE-MSE | _pending_ | biased observational baseline |
| Oracle CATE-MSE | ≈ 0 | Monte-Carlo floor |
| Ratio: naive / PFN (CATE) | _pending_ | how many times better than naive |
| Ratio: PFN / oracle (CATE) | _pending_ | how close to perfect causal recovery |
| Training wall time | _pending_ | H100/H200 single GPU |

Numbers fill in on first end-to-end run.

## Reproduce

```bash
pip install "pfnstudio-core[torch]" pfnstudio
git clone https://github.com/profitopsai/pfnstudio
cd pfnstudio

pfnstudio validate studies/do-pfn/priors/do_pfn_scm/
pfnstudio run      studies/do-pfn/runs/v0_1.yaml
pfnstudio eval     studies/do-pfn/evals/cid_recovery.yaml
```

Or:

```bash
./studies/do-pfn/reproduce.sh
```

## What's in here

```
studies/do-pfn/
├── README.md
├── reproduce.sh
├── priors/
│   └── do_pfn_scm/
│       ├── prior.yaml
│       └── prior.py            ← faithful reimplementation of the paper's SCM family
├── models/
│   └── do_pfn.yaml             ← 6-layer transformer (d=192, h=8)
├── runs/
│   └── v0_1.yaml               ← paper-pinned hparams
└── evals/
    └── cid_recovery.yaml       ← CID + CATE vs naive + oracle
```

## What this study does *not* show (yet)

- **Real-data benchmarks.** The paper evaluates on IHDP, Twins, and
  six synthetic case studies; v0.1 here ships only the synthetic
  reproduction. v0.2 will add the real-data scorers (IHDP first,
  Twins second).
- **Paper-scale training.** Upstream pre-trains on millions of DGPs.
  This v0.1 trains on fresh-per-step batches for 12k steps — same
  prior family, less data, faster turnaround. Useful to validate the
  mechanism; not a competitive headline number.
- **Distributional outputs.** Do-PFN's `predict_full` returns a full
  distribution; this v0.1 uses a `scalar_head` for point estimates
  (the trainer's MSE loss). A v0.2 with a quantile head would match
  the paper's full-CID output more faithfully.

## Reproducibility

What's pinned and what isn't:

| What | State | Why |
|---|---|---|
| **`hyperparams.seed = 42`** | ✅ Deterministic | Drives prior sampling, model init, optimizer state. |
| **Training-task seeds** | ✅ Disjoint | Each step samples 32 fresh task seeds (`seed + step·batch_size + i`). |
| **Eval-task seeds** | ✅ Fixed | The scorer always evaluates on the same 50 tasks. |
| **PyTorch determinism** | ⚠️ Best-effort | `torch.use_deterministic_algorithms(True)` is set inside the trainer, but CUDA kernels can introduce small drift across hardware. |
| **Pre-generated DGP pool** | ❌ Not used | Fresh-per-step sampling. The paper trains on a much larger fixed pool. |

## Citation

```bibtex
@misc{robertson2025dopfn,
  title  = {Do-PFN: In-Context Learning for Causal Effect Estimation},
  author = {Robertson, Jake and Reuter, Arik and Guo, Siyuan and Hollmann, Noah and Hutter, Frank and Sch\"olkopf, Bernhard},
  year   = {2025},
  eprint = {2506.06039},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2506.06039}
}
```

## License

This study (the YAML specs, prior.py, README) is **Apache-2.0**, same
as the rest of PFN Studio.

The underlying paper is on arXiv (CC BY 4.0). The upstream reference
code at [github.com/jr2021/Do-PFN](https://github.com/jr2021/Do-PFN)
**currently has no license declared**, so this study is a **faithful
reimplementation from the paper's description**, not a port of that
code. If/when the upstream adds an Apache-2.0 (or compatible) license,
a future version of this study may include a closer line-by-line port
of the SCM sampler and the DoPFNRegressor inference wrapper.
