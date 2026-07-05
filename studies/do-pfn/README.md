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
| Prior — `do_pfn_scm` (random-DAG SCM with latent-confounder nodes) | ✅ Faithful reimplementation from the paper (no upstream code copied). Needs `networkx` — see `priors/do_pfn_scm/requirements.txt` |
| Model — 12-layer axial-attention Do-PFN (d=192) | ✅ Paper-faithful primitives: `grid_preprocessor` → `tabular_cell_embedder` → `axial_attention_block × 12` → `row_pool_for_head` → `bar_distribution_head` |
| Bar-distribution head | ✅ Project block (`blocks/bar_distribution_head.py`) — full outcome distribution over 100 equal-mass buckets, bucketized NLL |
| Run — paper-pinned hyperparams | ✅ `lr=5e-4, bs=32, AdamW, wd=1e-5, clip=1.0, steps=12000` |
| Eval — CID + CATE recovery vs naive + Monte-Carlo oracle, plus PICP calibration | ✅ `evals/cid_recovery.py` scorer — `cate_true` is the MC oracle `E[Y|do(1),X]−E[Y|do(0),X]` |
| `pfnstudio validate` + `lint` | ✅ Pass. Pipeline (prior → 12-layer axial model → trainer → eval) verified end-to-end on CPU |
| Results table | ⏳ Pending first full training run on GPU (H100/H200) |
| v0.2 — real-data benchmarks (RealCause, Amazon, Law School) | 🗺️ Roadmapped |

## What this study does

1. **Draws synthetic SCM tasks**: each task is a fresh **random DAG** with
   - A random directed acyclic graph over K = d + unobserved + 2 nodes
     (edge density `p` sampled per task, so tasks range from sparse to dense)
   - Covariates X ∈ ℝ⁶ read off non-treatment/outcome nodes
   - Latent (unobserved) node(s) — the K − d − 2 nodes not exposed as
     covariates — which *may* confound T and Y depending on the draw
   - A binarized treatment node T (chosen among nodes with descendants)
   - The outcome Y read off a descendant of T; structural equations are
     random linear maps through γ ∈ {x², tanh, ReLU} nonlinearities + noise
2. **Packs each task as a (context, query) sequence**:
   - Context tokens carry the **observational** distribution `(X, T_obs, Y_obs)`
   - Query tokens carry the **desired intervention** `(X, t_query, 0)`
3. **Trains the model to predict the interventional outcome**
   `Y_{do(T = t_query)}` at each query position — drawn cleanly from
   the SCM under the do-operation, never visible in the observational
   context.
4. **Evaluates** on 50 fresh DGPs, scoring:
   - **CID-MSE** (+ range-normalized **CID-NMSE**, the paper's primary
     metric): predicted Y vs oracle Y_int per query
   - **CATE-MSE / -NMSE**: derived `pred Y do(1) - pred Y do(0)` vs the
     **Monte-Carlo oracle CATE** `E[Y|do(1),X] − E[Y|do(0),X]` (integrated
     over the exogenous noise with covariates pinned per row — *not* a
     single-draw contrast)
   - **PICP**: the paper's uncertainty metric — how often the true
     interventional Y lands in the bar head's central 90% predictive
     interval (well-calibrated ⇒ ≈0.90)
   - **Baselines**: per-task ridge regression (the confounded
     observational estimator) and the oracle CATE (the achievable floor)

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
pip install -r studies/do-pfn/priors/do_pfn_scm/requirements.txt   # networkx

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
│       ├── prior.py            ← random-DAG SCM sampler (paper Algorithm 1)
│       └── requirements.txt    ← networkx (installed before the prior imports)
├── blocks/
│   ├── bar_distribution_head.py   ← project block: distributional output head
│   └── bar_distribution_head.yaml
├── models/
│   └── do_pfn.yaml             ← axial-attention transformer (d=192, 12 layers)
├── runs/
│   └── v0_1.yaml               ← paper-pinned hparams
└── evals/
    ├── cid_recovery.yaml       ← CID + CATE vs naive + oracle
    └── cid_recovery.py         ← the scorer (@register_scorer)
```

## What this study does *not* show (yet)

- **Real-data benchmarks.** The paper evaluates on the RealCause
  datasets, an Amazon sales dataset, and a Law School Admissions
  dataset, alongside six structured synthetic case studies (observed
  confounder, mediator, unobserved confounder, back-door, front-door,
  …). v0.1 here ships only the random-DGP synthetic reproduction.
  v0.2 will add the real-data scorers (RealCause first).
- **Structured case studies.** The paper hand-builds confounder /
  mediator / back-door / front-door graphs; v0.1 relies on *random*
  DAGs where confounding is incidental (a latent node may or may not
  couple T and Y in any given draw), not deliberately constructed.
  A consequence: on the 50 random eval tasks a sizeable fraction
  (~30%) have a **near-zero oracle CATE** — the treatment barely moves
  the outcome, or its mediators are pinned — so `cate_nmse` is reported
  over the non-degenerate subset (the scorer surfaces the count in
  `meta.nmse_cate_tasks`). These tasks aren't *wrong* (model and oracle
  both →0 and agree), but they dilute the "beats-naive" signal. The
  structured case studies in v0.2 will give a stronger, non-degenerate
  test bed.
- **Paper-scale training.** Upstream pre-trains on millions of DGPs.
  This v0.1 trains on fresh-per-step batches for 12k steps — same
  prior family, less data, faster turnaround. Useful to validate the
  mechanism; not a competitive headline number.
- **Total-effect CATE under mediators.** The oracle pins the observed
  covariates (do(X=x)) before intervening on T, so when a covariate is
  a descendant of T the indirect path is blocked — a controlled-direct-
  effect flavour of CATE. Model and oracle share this convention (the
  model conditions on the same covariate columns), so the comparison is
  fair, but it is not the paper's total-effect estimand for mediated
  graphs. See the note atop `priors/do_pfn_scm/prior.py`.

Distributional outputs *are* shipped and now *evaluated*: the
`bar_distribution_head` predicts a full outcome distribution over 100
equal-mass buckets (bucketized NLL); the point estimate for MSE is the
distribution mean, and calibration is measured via **PICP** (90%
predictive-interval coverage) in the eval.

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
