# {{project_name}}

> {{one_line_description}}

A prior-fitted foundation model project, scaffolded with [PFN Studio](https://github.com/{{org}}/priorstudio).

## Layout

```
.
├── ROADMAP.md            versioned plan (v0.1 → v1.0)
├── initiatives/          one .md per workstream
├── priors/               YAML spec + Python impl per prior
├── models/               architecture configs
├── evals/                benchmark configs
├── literature/           BibTeX + per-paper summaries
└── runs/                 experiment manifests
```

## Workflow

1. **Add a prior.** Each prior is a directory under `priors/` with a `prior.yaml` (spec) and `prior.py` (implementation). See `priors/example_linear_scm/`.
2. **Add a model.** YAML config under `models/` describing the block composition.
3. **Add an eval.** YAML config under `evals/` pinning a dataset, metrics, and baselines.
4. **Define a run.** YAML manifest under `runs/` linking a prior + model + eval + hyperparams + compute target.
5. **Track it as an initiative.** Add a markdown file under `initiatives/` and link from `ROADMAP.md`.

## Validate

```bash
pfnstudio validate
pfnstudio lint              # cross-reference checks
```

## Run

```bash
pfnstudio run runs/example_run.yaml
```
