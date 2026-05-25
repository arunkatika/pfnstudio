# PFN Studio Roadmap

## Current capabilities (v0.5)

What works today, end-to-end:

**Authoring & artifacts**
- Five first-class artifact types: Prior, Model, Eval, Run, Initiative — each schema-validated, editable via UI or as files in the canonical FM project layout.
- Priors UI: parameter table, code editor, citations, version diff.
- Roadmap markdown editor (with preview) and Initiatives list.
- 13 seeded reference priors across regression, classification, time series, probabilistic, and causal-discovery categories — forkable into any project.

**Composition**
- Block registry with built-in PyTorch blocks: `transformer_encoder`, `causal_attention_pool`, `tabular_embedder`, `discovery_head`, `estimation_head`, `scalar_head`. Custom blocks via `@register_block`.

**Training & compute**
- Local PFN training loop (in-process, requires `pfnstudio-core[torch]`).
- Compute adapters: `local` works; `vast`, `modal`, `runpod`, `hf_spaces` scaffolded.
- Tracking adapters: local (`results.json` + `metrics.jsonl`), W&B, MLflow, HuggingFace model upload.

**Platform**
- Multi-tenant API (NestJS + Prisma + Postgres) with JWT + API-token auth and project-scoped guards.
- Angular product app: login/register, project list, project nav.
- Export endpoint: project → tar.gz of canonical FM file layout.
- CLI: `pfnstudio init/validate/lint/list-artifacts/run/version` with cross-reference checks.
- Share links per trained run, Docker Compose for Postgres, seed script (demo workspace).
- CI on every PR: pytest + template validation + ruff.

What does **not** exist yet (motivates the roadmap below):
- Models / Evals / Runs / Literature editor UIs (API works; editors stubbed).
- CLI ↔ API integration (`pfnstudio login/pull/push`).
- Run submission from the web UI; live run status.
- Axis-aware priors / promptable inference (see v0.7).
- Automated prior fitting (see v0.8).

---

## v0.1 — Scaffolding ✅
- [x] Repo structure, license, docs skeleton
- [x] FM project template with examples for every artifact type
- [x] JSON Schemas for prior / model / eval / run / initiative
- [x] CONTRIBUTING.md with PR conventions

## v0.2 — CLI ✅
- [x] `pfnstudio init/validate/lint/list-artifacts/run/version`
- [x] Cross-reference checks (run→prior/model/eval, dir/id match, bibkeys)

## v0.3 — Static Studio ✅
- [x] Static-site generator (pfnstudio-studio package)
- [x] `studio build` and `studio serve`

## v0.4 — Compute & tracking ✅
- [x] core package: Prior/Model/Eval/Run, registry, loaders
- [x] Built-in PyTorch blocks
- [x] Local PFN training loop
- [x] Compute adapters: local works, vast/modal/runpod/hf_spaces scaffolded
- [x] Tracking adapters: local + W&B + MLflow
- [x] Tests + GitHub Actions CI

## v0.5 — Web app + database ⚙️
- [x] Prisma schema for all entities (User, Org, OrgMember, Project, Prior, Model, Eval, Run, Initiative, LiteratureEntry, ApiToken)
- [x] NestJS API with JWT + API token auth, multi-tenant guards (ProjectAccessGuard + role-based)
- [x] CRUD endpoints for all artifact types
- [x] Export endpoint (project → tar.gz of canonical FM file layout)
- [x] Angular shell with login/register, project list, project nav
- [x] Priors fully wired UI (list / create / edit / delete with parameter table, code editor, citations)
- [x] Roadmap markdown editor with preview
- [x] Initiatives list + create
- [x] Docker Compose for Postgres
- [x] Seed script (demo workspace + demo project)
- [ ] Models / Evals / Runs / Literature UIs (API works; editors stubbed)
- [ ] CLI ↔ API integration (`pfnstudio login`, `pfnstudio pull`, `pfnstudio push`)
- [ ] Full SSH/scp orchestration for Vast adapter
- [ ] PyPI release for python packages

## v0.6 — Polish
- [ ] Models / Evals / Runs / Literature editor UIs
- [ ] Diff view between artifact versions
- [ ] Run submission from the web UI (kicks off compute adapter on backend worker)
- [ ] Run live status (websocket / SSE)
- [ ] Comments / @mentions on artifacts
- [ ] First external project shipped on it

## Toward v1.0 (no commitment yet)

v1.0 means schemas frozen and the platform is stable. The workstreams below are scoped but not version-pinned — they'll slot into a v0.6.x or v0.7 cut once their MVP is ready, rather than blocking on a single big release.

### Foundation hardening
- [ ] Schema versioning + migration story
- [ ] Plugin system for custom artifact types
- [ ] At least 3 external PFN projects in production on it
- [ ] Hosted version at priorstudio.dev
- [ ] Documentation site

### Promptable inference (axis-aware priors)

A prior should expose *steerable axes* that the trained model honors at inference time, so domain experts can inject knowledge without retraining. Same model weights, different answer per constraint set. Existing samplers are unchanged; axes are *plugins* that narrow the prior.

- [ ] Axis specification format (`axes/<name>.py` + `axes/<name>.yaml`) — name, type (categorical / range / boolean), value space, default `unknown`, sampler hook.
- [ ] Axis registry + loader; priors declare which axes they consume in `prior.yaml`.
- [ ] **Axes editor in Edit Prior** — collapsed *Advanced* section below Citations, with sampler-hook code editor, value-space form, sample-mass sliders, "Preview 8 datasets" generator, "Test honoring" button. New Prior gets axes after Edit Prior is validated.
- [ ] First two reference axes shipped: `monotonicity`, `lag_scale`.
- [ ] Training-loop changes: each batch carries a sampled tag; tag embedding injected via cross-attention; loss is NLL conditional on tag.
- [ ] **Unknown-mass invariant.** A fixed fraction of training samples (default 30%) use `tag=unknown` for every axis, preserving bit-identical unconditional behaviour against the pre-axis baseline.
- [ ] **Honoring metric** in evals: holding context constant, flip a tag's value, measure prediction divergence in the expected direction. Surfaces broken axes before full training runs.
- [ ] **Coverage dashboard:** fraction of axis-value combinations seen at training time; flags under-sampled combos as deployment risk.
- [ ] Detector head (read-only inference of identifiable axes from context) trained as an auxiliary task; output: per-axis value + confidence.
- [ ] Inference API extension: `predict(context, tag=None)` — `None` is the new unconditional path.
- [ ] Axis vocabulary frozen as a v1 foundation contract (additions become quarterly releases).

### Learnable prior (auto-fit)

Make prior hyperparameters a search space and let the outer loop optimise them against held-out benchmarks (including customer-private data).

- [ ] Declare prior knobs as a typed search space inside `prior.yaml` (`search_space:` block) — categorical, integer, log-uniform, etc.
- [ ] **Auto-Fit** Studio UI: pick benchmarks, pick search-space scope, view trial table (per-trial knob deltas + metrics), live score curve, ETA, pause/resume.
- [ ] Three optimiser backends, shipped in this order:
  - [ ] Bayesian optimisation (works to ~20 knobs).
  - [ ] CMA-ES / evolutionary search (~100 knobs, embarrassingly parallel on Vast.ai).
  - [ ] Implicit differentiation through the inner training loop (true gradient, research-grade).
- [ ] Per-customer "Plant" benchmark slot — fit the prior to a customer's historical data *inside their tenant*; the data never leaves.
- [ ] "Promote to production" button: a winning trial becomes a new `prior` version pinned by a `run`.
- [ ] Cost guardrails: per-trial training budget, total wall-clock budget, early stopping on plateau.

### Glossary layer + vertical productisation

The thin translation seam that lets one foundation model serve many domains without retraining.

- [ ] **Glossary YAML** schema: site-specific terms → universal axis values (e.g. `"dryer lag": { axis: lag_scale, value: [3h, 5h] }`).
- [ ] Glossary editor in Studio with axis autocomplete and a "test phrase" sandbox.
- [ ] Vertical bundle export: foundation weights + glossary + benchmark pack as one shippable artifact.
- [ ] NL → axes parser (small LLM, optional) for free-form analyst input; routes to the same canonical axes for auditability.
- [ ] Reference glossaries for two verticals (pulp & paper, pharma) shipped as examples.
