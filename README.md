# PFN Studio

> A toolkit for training [prior-fitted foundation models](https://arxiv.org/abs/2112.10510) — and a marketplace of priors to fit them to.

Prior-fitted networks (PFNs) are a different shape of foundation model. Instead of pretraining on a giant scraped corpus, you write a **prior** — a synthetic data generator that captures what you believe your domain looks like — and train a transformer offline to do in-context Bayesian inference over samples from it. At runtime, the model takes a small set of real (input, output) pairs and predicts on new inputs in one forward pass. No fine-tuning. No SGD on the new data.

Müller et al. published the idea in [*Transformers Can Do Bayesian Inference*](https://arxiv.org/abs/2112.10510) (ICLR 2022). PFN Studio is the tooling that paradigm has been missing: priors as first-class artifacts, reproducible training runs, evals against synthetic ground truth, and a marketplace of pre-built priors that the community can grow.

```
prior.py + prior.yaml          →   pfnstudio run runs/v0_1.yaml
synthetic data generator           ↓
                                   transformer that does
                                   in-context inference on
                                   any new dataset
```

---

## What's in this repo

**[`priors/`](priors/)** — **13 reference priors**, each a self-contained `prior.py` + `prior.yaml` + `README.md`. Categories: regression, classification, time series, probabilistic, causal discovery. Forkable — copy a prior directory into your project, edit the Python, train.

**[`packages/core/`](packages/core/)** (`pfnstudio-core` on PyPI) — the runtime: `Prior` interface, block registry (`tabular_embedder`, `transformer_encoder`, `scalar_head`, `discovery_head`, `causal_attention_pool`, `estimation_head`), training loop, dataset registry, eval scorers.

**[`packages/cli/`](packages/cli/)** (`priorstudio` on PyPI) — the command-line interface: `validate`, `lint`, `sample`, `run`, `predict`, `export`.

**[`packages/studio/`](packages/studio/)** (`pfnstudio-studio` on PyPI) — a static-site renderer that turns a PFN Studio project directory into a browsable HTML site (`studio build`, `studio serve`).

**[`schemas/`](schemas/)** — JSON Schema for every artifact type (`prior.yaml`, `model.yaml`, `eval.yaml`, `run.yaml`, `initiative.md`).

**[`starters/fm-project/`](starters/fm-project/)** — the starter project scaffold that `pfnstudio init` copies to create a new project.

**[`examples/`](examples/)** — end-to-end scripts and notebooks. `01_linear_regression.py` trains a small PFN on the Bayesian linear regression prior in <10 minutes on CPU and verifies that the model matches the closed-form posterior mean.

**[`docs/`](docs/)** — concepts, architecture, compute targets, getting-started.

## What's *not* in this repo

The **hosted studio** at [pfnstudio.com](https://pfnstudio.com) — the visual designer, run orchestration, GPU scheduling, sharing infrastructure, marketplace publishing, and team features — is a closed-source product. The studio uses this repo's packages as its training engine and reads `priors/` as its marketplace catalog.

If you want a visual editor and someone else running the GPUs, sign up at [pfnstudio.com](https://pfnstudio.com). If you want to train PFNs yourself, hack on priors locally, or contribute to the catalog, this repo is everything you need.

---

## Quickstart

Requires Python 3.10+ and (for actually training models) PyTorch.

```bash
# 1. Install — no PyTorch yet, so it's lightweight
pip install pfnstudio-core

# 2. Hello PFN. Sample one task from a prior; runs in under a second.
git clone https://github.com/profitopsai/pfnstudio
cd priorstudio
python examples/00_hello_pfn.py

# 3. Validate one of the bundled priors against the schema
pip install pfnstudio                                       # the CLI
pfnstudio validate priors/linear-regression/

# 4. Sample 3 tasks from it to see what training looks like
pfnstudio sample priors/linear-regression/prior.yaml --seeds 3

# 5. Train a small PFN locally — 5–10 min on CPU. Needs PyTorch.
pip install "pfnstudio-core[torch]"
python examples/01_linear_regression.py
```

The training loop emits one JSON line per step on stdout so you can pipe it through `jq` or watch in real time. Final output: a checkpoint at `checkpoint/model.pt` plus a metrics dict that — for the linear-regression prior — matches the analytic Bayesian posterior mean within numerical noise. That's the headline test: a transformer trained on synthetic random linear functions does Bayesian inference correctly on data it has never seen, in a single forward pass.

---

## The marketplace

The 13 priors in [`priors/`](priors/) are the curated v0 catalog — each is a complete training-ready spec with a Python implementation, parameter ranges, output schema, and a citation trail. Click through:

| Slug | What it learns | Category |
|---|---|---|
| `linear-regression` | `y = ax + b + ε` — the reference PFN | regression |
| `polynomial-regression` | Random polynomials up to degree D | regression |
| `gp-regression` | GP draws with RBF kernel | regression |
| `two-moons` | Random interlocking half-moons + 0/1 labels | classification |
| `gaussian-mixture-classification` | k-class Gaussian-mixture problems | classification |
| `logistic-interactions` | Random logistic models with interaction terms | classification |
| `sine-wave` | `y = a·sin(ωt + φ) + ε` | time series |
| `ar2-process` | Stationary AR(2) — closer to industrial data | time series |
| `seasonal-trend` | Trend + seasonal + noise decomposition | time series |
| `coin-flip` | Bernoulli with a Beta-distributed bias | probabilistic |
| `hierarchical-normal` | Two-level Normal–Normal hierarchical models | probabilistic |
| `linear-scm` | Random linear structural causal models | causal |
| `chain-scm` | Chain-shaped DAGs (X → Y → Z) | causal |

Every prior is a forkable starting point. Open one, change a sampling distribution, retrain, share the checkpoint. The same priors are what the [pfnstudio.com](https://pfnstudio.com) marketplace serves — this directory is the canonical source.

---

## Paper-replication studies (work in progress)

We've scaffolded several published PFN papers as PFN Studio projects. The studio's marketplace surfaces them as "Importable" projects:

- **PFNs** — Müller et al., ICLR 2022 (Apache-2.0). The reference implementation.
- **LC-PFN** — Adriaensen et al., NeurIPS 2023 (MIT). Learning-curve extrapolation.
- **ifBO** — Rakotoarison et al., ICML 2024 (MIT). Freeze-thaw Bayesian optimisation.
- **PFNs4BO** — Müller et al., ICML 2023 (Apache-2.0). In-context Bayesian optimisation.
- **TabPFN-TS** — Hoo et al., arXiv 2501.02945 (Apache-2.0). Time-series forecasting as tabular regression.
- **KinPFN** — Scheuer et al., ICLR 2025 (Apache-2.0). RNA folding kinetics.

**Honest disclaimer:** these are *demonstration* scaffolds, not faithful reproductions of the papers' numerical results. Each one captures the central idea (the prior, the model shape, the eval baseline) at a tractable scale that runs on CPU in minutes. To match a paper's headline table we'd need its real eval dataset, the paper's full prior, training at paper scale, and (in some cases) a pretrained base model. We're upfront about this in each template's README and in their roadmap files.

The first eval that runs against real data end-to-end is the **M4-monthly forecast scorer**, which loads the M4 competition's monthly series and computes MSE + MASE on held-out tails. See [`packages/core/pfnstudio_core/scorers/m4_monthly_forecast.py`](packages/core/pfnstudio_core/scorers/m4_monthly_forecast.py).

---

## Architecture in 60 seconds

```
                 ┌───────────────────────────────────────────┐
                 │   priors/<slug>/                          │
                 │     prior.py      sample(seed, **params)  │
                 │     prior.yaml    parameter spec + outputs│
                 │     README.md     human-readable          │
                 └────────────────────┬──────────────────────┘
                                      │
                                      ▼
   pfnstudio run runs/v0_1.yaml ──► training loop ──► model.pt
                                      │
                  ┌───────────────────┼────────────────────┐
                  │                   │                    │
                  ▼                   ▼                    ▼
            block registry      dataset registry      eval scorers
            (transformer,        (M4, LCBench,        (MSE, MASE,
             pooling, heads)      HPO-B, …)            KS, RMSE-vs-
                                                       posterior, …)
```

Every artifact is YAML-spec plus a Python implementation. Runs are reproducible from configs alone — no notebook state, no dataloaders to wire up, no glue code.

- **Concepts**: [`docs/concepts.md`](docs/concepts.md) — the five first-class artifact types
- **Architecture**: [`docs/architecture.md`](docs/architecture.md) — how the pieces fit together
- **Compute targets**: [`docs/compute.md`](docs/compute.md) — local, Modal, RunPod, Vast, HF Spaces

---

## The block registry

Architectures are composed from registered blocks. Today's library, all in `pfnstudio_core/blocks/`:

```yaml
# A standard tabular-PFN model
blocks:
  - type: tabular_embedder
    config: { d_model: 128 }
  - type: transformer_encoder
    config: { d_model: 128, n_heads: 4, n_layers: 4 }
  - type: scalar_head
    config: { d_model: 128, d_out: 1 }
```

Add your own with `@register_block("name")` and PFN Studio picks it up at runtime — no fork required.

---

## Contributing

We want this repo to be the canonical PFN marketplace, and that only works if it's easy to contribute a prior or fix a scorer. See [CONTRIBUTING.md](CONTRIBUTING.md) for the basics.

Good first contributions:
- **Add a prior**: a `prior.py + prior.yaml + README.md` triple under `priors/<slug>/` plus a smoke test. Examples of what's missing from the catalog: censored-data regression, multi-arm bandit, ordinal classification, count regression, change-point time series.
- **Improve a scorer**: the M4 scorer in [`packages/core/pfnstudio_core/scorers/`](packages/core/pfnstudio_core/scorers/) is the worked example; LC-PFN / ifBO / KinPFN / PFNs4BO scorers are wanted.
- **Pick a paper to replicate** properly — each paper-template's `roadmap.md` lists the gaps between the scaffold and the published results.

---

## License + citation

[Apache 2.0](LICENSE). If you use PFN Studio in a paper, please cite the original PFN paper:

```bibtex
@inproceedings{muller2022pfn,
  title   = {Transformers Can Do Bayesian Inference},
  author  = {Müller, Samuel and Hollmann, Noah and Pineda Arango, Sebastian
             and Grabocka, Josif and Hutter, Frank},
  booktitle = {International Conference on Learning Representations},
  year    = {2022}
}
```

And, if PFN Studio specifically was useful, a link to this repo is the most helpful citation we could ask for.

[NOTICE](NOTICE) lists upstream OSS projects that this repo's starters, studies, and scorers were derived from, with their licenses and citations.

---

## Hosted studio + commercial product

[pfnstudio.com](https://pfnstudio.com) is the closed-source SaaS that builds on this repo: visual designer, GPU run orchestration, project sharing, marketplace publishing, team features. The studio is in private alpha — drop us a line at [hello@profitops.ai](mailto:hello@profitops.ai) or [book a call](https://calendly.com/profitops) if you want early access. Either way, you don't need the cloud to use this repo.

— [ProfitOps.ai](https://profitops.ai)
