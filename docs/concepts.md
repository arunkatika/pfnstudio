# Concepts

PFN Studio has five first-class artifact types. Each is a structured file in your repo. Each has a JSON Schema in `schemas/`. Each can be added, edited, or reviewed via a normal PR.

## Prior

A synthetic data generator. The defining mechanic of a prior-fitted network: instead of training on scraped real data, you sample tasks from a prior and train the model to do posterior inference on those tasks.

A prior is a directory:

```
priors/<id>/
├── prior.yaml      spec: parameters, ranges, output shapes
├── prior.py        implementation: a sample() function
└── prior.md        why this prior exists, identifiability notes, known limitations
```

The yaml is a strict contract — JSON-Schema-validated. The Python file is convention; we recommend a top-level `sample(**params, seed)` returning a dataclass with named outputs that match `outputs.variables` in the spec.

## Model

An architecture, defined as a composition of blocks. The block registry lives in `priorstudio.core` (v0.2). Models are config-driven so the same architecture is reproducible across runs.

```yaml
blocks:
  - type: temporal_encoder    # registered block
    config: { d_model: 256, n_layers: 4 }
  - type: causal_attention    # registered block
    config: { n_heads: 8 }
```

Custom blocks: register via `@register_block("my_block")` in your project's Python code. Once registered they're usable in any model config.

## Eval

A benchmark configuration. Pins a dataset, the metrics, and known baselines. Results from a run get written back into the eval's results history (v0.2).

The point: when someone reports "we beat baseline X on Sachs," the eval YAML defines exactly what "Sachs" and "X" mean.

## Run

An experiment manifest. A run pins:

- A specific prior version
- A specific model version
- One or more eval versions
- Hyperparameters
- Compute target (local / Vast / Modal / etc.)
- Tracking destination (W&B project, HF repo)

Reproducibility property: given a run yaml, anyone with the same compute can reproduce the experiment.

## Initiative

A research workstream. Markdown with frontmatter that links into the roadmap. Status fields: `proposed`, `in_progress`, `blocked`, `done`, `abandoned`.

The point: keep the *why* of a piece of work alive next to the artifacts it touches, not in a separate Notion doc that drifts out of sync.

## Axis (v0.7, planned)

A *steerable property* of a prior — e.g. `monotonicity`, `lag_scale`, `feedback_allowed`. Axes are plug-ins that narrow an existing prior without rewriting it; the trained model learns to honor them at inference, so domain experts can inject knowledge (via UI chips or a glossary) without retraining.

```
axes/<name>/
├── axis.yaml      type, value space, default (always includes `unknown`)
└── axis.py        sampler hook: (scm, tag_value) -> scm
```

`prior.yaml` declares which axes it consumes. A fraction of training samples always use `tag=unknown` for every axis, so the unconditional path remains bit-identical to a pre-axis baseline.

Two diagnostic metrics live alongside the axis:

- **Honoring** — does the trained model actually change its prediction when the axis flips? Measured by held-context tag-flip divergence.
- **Coverage** — fraction of axis-value combinations seen during training. Low coverage is a deployment risk.

See [ROADMAP.md](../ROADMAP.md#v07--axis-aware-priors-promptable-inference) for the full v0.7 plan.

## How they connect

```
ROADMAP.md
    └── initiatives/0001-foo.md  ── status, version_target
                │
                │ (links to)
                ▼
        priors/foo/prior.yaml + prior.py
        models/bar.yaml
        evals/baz.yaml
                │
                │ (composed in)
                ▼
        runs/run_v0_1.yaml ── results filled in after execution
```
