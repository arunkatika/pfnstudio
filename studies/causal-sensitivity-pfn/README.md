# Study: amortized causal sensitivity analysis (Javurek et al., 2026)

> **Claim**: a single 4-layer transformer trained on synthetic MLP-SCM
> tasks recovers the latent CATE in-context — despite never observing
> the unobserved confounder — beating the naive observational estimator
> by a sizeable margin. One forward pass per task; no per-instance
> optimization. v0.1 ships CATE recovery; v0.2 will swap CATE for the
> paper's full Lagrangian-labeled sensitivity bound.

Port of [Javurek, Frauen, Brockschmidt, Schweisthal, Feuerriegel,
*Amortizing Causal Sensitivity Analysis via Prior Data-Fitted
Networks*](https://arxiv.org/abs/2605.10590), reimplemented end-to-end
in pfnstudio. Every artifact is in this directory; nothing is hidden
in the binary.

## Status

| What | State |
|---|---|
| Prior — MLP-SCM sampler | ✅ Faithful reimplementation from the paper (no upstream code copied) |
| Model — 4-layer Causal PFN | ✅ Uses pfnstudio's `transformer_encoder` + `scalar_head` |
| Run — paper-pinned hyperparams | ✅ `lr=1e-3, bs=32, optim=Adam, wd=1e-5, clip=1.0` (matches upstream training defaults) |
| Eval — naive vs PFN vs oracle | ✅ |
| Results table | ⏳ Pending first end-to-end run on H100/H200 — values will be filled once `outputs/metrics.json` lands |
| v0.2 — Lagrangian-labeled bounds + GMM head | 🗺️ Roadmapped (requires a new `gmm_head` block in `pfnstudio_core`) |

## What this study does

1. **Draws synthetic MLP-SCM tasks**: each task is a fresh random SCM
   with X ∈ R¹⁰, an unobserved confounder U coupled to the outcome only,
   binary treatment T sampled from a per-task propensity MLP, and a
   continuous outcome Y from a per-task outcome MLP.
2. **Packs each task as a (context, query) sequence** the transformer
   attends across — context tokens carry (X, T, Y), query tokens have T
   and Y masked. The target at each query is the latent CATE for that X
   (computed by Monte-Carlo integration over U inside the prior).
3. **Trains one PFN on 8000 batches of fresh tasks**. The model never
   sees U; it has to learn to deconfound from the context pairs alone.
4. **Evaluates on 50 fresh tasks**, comparing PFN MSE against:
   - **Naive**: per-task ridge regression of E[Y|T=1,X] - E[Y|T=0,X] on
     the context. Biased under unobserved confounding.
   - **Oracle**: the latent CATE. Floor — what a perfect causal
     estimator scores.

## Result

| metric | value | notes |
|---|---|---|
| **PFN MSE** | _pending_ | in-context inference on 50 held-out tasks |
| Naive (per-task ridge) | _pending_ | biased under unobserved confounding |
| Oracle (latent CATE) | ≈ 0 | Monte-Carlo floor |
| Ratio: naive / PFN | _pending_ | how many times better than naive |
| Ratio: PFN / oracle | _pending_ | how close to perfect causal recovery |
| Training wall time | _pending_ | H100/H200 single GPU |

Numbers filled in on first end-to-end run; this README is currently a
scaffold.

## Reproduce

Two equivalent paths.

### A. From the CLI (same path the hosted studio uses)

```bash
pip install "pfnstudio-core[torch]" pfnstudio
git clone https://github.com/profitopsai/pfnstudio
cd pfnstudio

pfnstudio validate studies/causal-sensitivity-pfn/priors/mlp_sensitivity_scm/
pfnstudio run      studies/causal-sensitivity-pfn/runs/v0_1.yaml
pfnstudio eval     studies/causal-sensitivity-pfn/evals/cate_recovery.yaml
```

### B. Via the one-line script

```bash
./studies/causal-sensitivity-pfn/reproduce.sh
```

Same thing, plus a friendly progress narration.

## What's in here

```
studies/causal-sensitivity-pfn/
├── README.md                                ← you are here
├── reproduce.sh                             ← one-line end-to-end
├── priors/
│   └── mlp_sensitivity_scm/
│       ├── prior.yaml                       ← spec (schema-validated)
│       └── prior.py                         ← @register_prior implementation
├── models/
│   └── causal_pfn.yaml                      ← 4-layer transformer config
├── runs/
│   └── v0_1.yaml                            ← paper-pinned hparams
└── evals/
    └── cate_recovery.yaml                   ← naive vs PFN vs oracle
```

## What this study does *not* show (yet)

- **Sensitivity bounds.** The paper's central contribution is amortizing
  the Marginal Sensitivity Model's [lower, upper] bound on the causal
  effect given a Γ-parameter. v0.1 ships CATE recovery (Γ → ∞ limit, in
  a sense); v0.2 adds the Lagrangian-labeled bound targets and a GMM
  head over the bound distribution. Both are required to match the
  paper's headline plot.
- **Per-instance baseline comparison.** The paper benchmarks against
  per-instance optimization methods like Kallus-Zhou and ZSB. v0.1's
  baseline is per-task naive ridge regression — the *observational*
  estimator. The per-instance bound comparison comes with v0.2.
- **Scale.** The upstream repo trains for 150 epochs over a much larger
  pre-generated DGP pool. This v0.1 trains on 8000 fresh-per-step
  batches — same prior family, less data, faster turnaround.

## Reproducibility

What's pinned and what isn't:

| What | State | Why |
|---|---|---|
| **`hyperparams.seed = 42`** | ✅ Deterministic | Drives prior task sampling, model init, optimizer state. |
| **Training-task seeds** | ✅ Disjoint | Each step samples 32 fresh task seeds (`seed + step·batch_size + i`). |
| **Eval-task seeds** | ✅ Fixed | The scorer always evaluates on the same 50 tasks. |
| **PyTorch determinism** | ⚠️ Best-effort | `torch.use_deterministic_algorithms(True)` is set inside the trainer, but CUDA kernels can still introduce small drift across hardware. |
| **Pre-generated DGP pool** | ❌ Not used | We sample fresh DGPs each step — same prior family as upstream, no pickled-on-disk DGP cache. |

## Citation

If you use this study or build on it, please cite the original paper:

```bibtex
@misc{javurek2026csa,
  title={Amortizing Causal Sensitivity Analysis via Prior Data-Fitted Networks},
  author={Javurek, Emil and Frauen, Dennis and Brockschmidt, Marie and Schweisthal, Jonas and Feuerriegel, Stefan},
  year={2026},
  eprint={2605.10590},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.10590}
}
```

## License

This study (the YAML specs, prior.py, README) is **Apache-2.0**, same as
the rest of pfnstudio.

The underlying paper is on arXiv and is © the authors. The upstream
reference code at
[github.com/EmilJavurek/Amortizing-Causal-Sensitivity-Analysis-via-PFNs](https://github.com/EmilJavurek/Amortizing-Causal-Sensitivity-Analysis-via-PFNs)
currently has no license declared, so this study is a **faithful
reimplementation from the paper's description**, not a port of that
code. If/when the upstream adds an Apache-2.0 license, a future version
of this study may include a closer line-by-line port of the SCM
generator and the GMM head.
