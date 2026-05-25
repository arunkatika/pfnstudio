# Importing priors from an existing Python package

How to take a prior class that lives in your own (possibly private) Python package — e.g. a research codebase like TCPFN — and bring it into PFN Studio so it shows up in the web UI, samples on click, and can be forked, evaluated, and shared.

The flow has two halves:

1. **Authoring** — run a CLI command in your package's venv to generate a PFN Studio-format directory.
2. **Pushing** — upload that directory (with a wheel of your package) to a PFN Studio project.

The web UI then handles the rest: dependency isolation per prior, sampling, visualisation.

---

## Prerequisites

- A running PFN Studio stack (`./run.sh start`).
- Your package importable in a local Python venv (`pip install -e .` or similar).
- A wheel-buildable `pyproject.toml` in your package — used to ship private deps offline.

---

## Step 1 — Issue an API token

The CLI needs a personal token to push. Tokens are separate from your browser login.

1. Open http://localhost:4200 → log in.
2. Click **API tokens** in the header.
3. Name the token (e.g. `tcpfn-cli`), set an expiry, click **Issue token**.
4. Copy the `ps_…` value **immediately** — it's only shown once.

```bash
export PFNSTUDIO_TOKEN=ps_xxxxxxxxxxxxxxxxxxxxxxxx
```

To revoke a leaked token, delete it from the same page.

---

## Step 2 — Wrap each prior class

In the venv that has your package importable, run:

```bash
pfnstudio author wrap \
  --class your_pkg.module:YourPriorClass \
  --slug your-slug \
  --out ./port-priors \
  --kind temporal_causal \
  -r your-package -r torch
```

Flags:
- `--class`     `module:ClassName` — the importable path to your class.
- `--slug`      kebab-case identifier used as the prior's URL slug and registry key.
- `--out`       project root that will host `priors/<slug>/`. Reuse the same `--out` across multiple `wrap` calls so all your priors land in one project.
- `--kind`      one of: `scm`, `tabular`, `temporal`, `temporal_causal`, `graph`, `custom`. Drives per-kind UI hints in the inspector.
- `-r`          repeat for each `requirements.txt` line. Use `your-package` so the cached install knows it needs your package; the actual binary will come from the wheel you upload in step 3.
- `--sample-method` (optional) override if your class's batch method isn't called `sample_batch`.

This generates:

```
port-priors/priors/<slug>/
├── prior.py            # auto-generated wrapper around your class
├── prior.yaml          # parameters introspected from __init__
├── prior.md            # rationale stub
└── requirements.txt
```

The generator is best-effort. **You should hand-edit `prior.yaml` to fill in `outputs.variables`** — the introspector can't read return shapes from your `sample_batch`. Each entry should look like:

```yaml
- name: x
  type: tensor
  shape: "[T, V]"          # quoted! shape is a free-form string, not a list
  description: "Time series of V variables over T steps."
```

The visualisation picker uses this to decide between sparkline / multi-line / heatmap.

You should also sanity-check the generated `prior.py` — particularly the `sample_batch` call:

- If your class takes `seed` in `__init__`, the wrapper will forward it. Good.
- If your `sample_batch` uses kwargs other than `batch_size` (e.g. `n_context_units`, `n_query_units`), the generator will pick those up but with default values. **Tune the defaults** for the task — the generator emits a `# FIXME` comment where it can.

Repeat `pfnstudio author wrap` for every prior class you want to migrate.

---

## Step 3 — Build a wheel of your package

Your private package isn't on PyPI, so the API can't `pip install` it. Bundle it as a wheel:

```bash
python -m build --wheel --outdir /tmp/wheels
# /tmp/wheels/your_package-1.0.0-py3-none-any.whl
```

Drop the wheel into each prior's `wheels/` subdirectory:

```bash
for slug in your-slug-1 your-slug-2 ...; do
  mkdir -p ./port-priors/priors/$slug/wheels
  cp /tmp/wheels/your_package-*.whl ./port-priors/priors/$slug/wheels/
done
```

The CLI uploads everything in `wheels/` alongside the prior. The API caches the install per `(requirements.txt, wheel SHAs)` tuple, so all priors that share the same wheel + requirements share one cache entry.

> **Watch out:** if you change your package and rebuild the wheel, the SHA changes → the cache invalidates → next click re-installs (~5 min for torch-heavy stacks). Don't re-build casually.

> **Also:** declare *all* eagerly-imported third-party packages in your `pyproject.toml` `dependencies`. `pip install <wheel>` resolves the deps at install time using `--find-links` + PyPI as the resolver sources; missing deps surface only later when the sampler tries to import. We hit this with `pandas`/`scipy`/`joblib`/`networkx` in TCPFN.

---

## Step 4 — Create a target project

If you don't already have a project to push into, create one in the web UI:

1. http://localhost:4200/projects → **New project**.
2. Name it (e.g. "My TCPFN") — the slug is auto-derived (e.g. `my-tcpfn`).
3. Copy the slug for step 5.

(There's no CLI command to create a project — projects belong to organizations and are gated by membership, so they're created in the UI.)

---

## Step 5 — Push

From the directory that contains `port-priors/`:

```bash
pfnstudio push ./port-priors --project-id my-tcpfn
```

You should see one line per prior:
- `✓ prior <slug>` — newly created.
- `⤳ prior <slug> already exists — skipping body, will refresh wheels` — body is preserved, but local wheels are re-uploaded so dep-cache invalidation works.
- `+ wheel <name>.whl` — wheel uploaded.

Existing priors are skipped by default (their body isn't overwritten). Pass `--no-skip-existing` to fail loudly on conflict if that's the behaviour you want.

You can also reach this command from the UI: project → **Priors** tab → **Import** button shows a click-to-copy snippet pre-filled with the project slug.

---

## Step 6 — Click the prior

Navigate to your project's **Priors** tab and click any prior. The first click does heavy work:

1. Hashes `(requirements.txt, sorted wheel SHAs)` → dep-cache key.
2. `pip install --target <dep-cache>/<hash> --find-links <wheel-dir> -r requirements.txt <wheel…>` — single resolver call; local wheels feed the resolver, requirements.txt can reference anything available in either source. **Allow ~5 min** for torch-heavy stacks on first install.
3. Spawns `pfnstudio sample` with `PYTHONPATH` prepended to the cache.
4. Visualises the result using the schema-driven picker:
   - 1D `[N]` → sparkline.
   - 2D `[T, V]` (long & narrow) → multi-line series.
   - 2D otherwise → heatmap.
   - 3D → first-slice heatmap.
   - scalar → big stat card.
   - dict / unknown → JSON fallback.

Subsequent clicks hit the cache and are instant.

---

## Common gotchas

- **Token shown once.** Lose it → revoke and re-issue, don't try to recover.
- **Slug, not name.** The CLI's `--project-id` accepts either the UUID or the URL slug. Use the slug — it's stable across resets, the UUID isn't.
- **Wheel filename = package name + version.** Re-uploading a wheel with the same filename overwrites the disk file *and* replaces the DB row, so the cache key flips. By design.
- **Optional / native deps.** Packages that try to import optional native libs at module load (triton, momentfm, etc.) fail on macOS arm64. Wrap those imports in `try / except ImportError` in your source — don't add them to `pyproject.toml` `dependencies`.
- **`shape:` must be a string, not a list.** `prior.yaml` validates against a Pydantic model with `shape: str | None`. Use `shape: "[T, V]"`, not `shape: [T, V]`.
- **Python version.** PFN Studio runs an internal `.deps-venv` on Python 3.11 (configured via `PFNSTUDIO_PIP` and `PFNSTUDIO_CLI` in `.env`). Your authoring venv can be any version — only the *generated* code needs to be importable on 3.11.

---

## End-to-end example: TCPFN → PFN Studio

```bash
# In tcpfn's venv (where `import tcpfn` works)
cd /path/to/tcpfn

pfnstudio author wrap \
  --class tcpfn.discovery.discovery_prior:CausalDiscoveryPrior \
  --slug tcpfn-discovery --out ./port-priors --kind temporal_causal \
  -r tcpfn -r torch

pfnstudio author wrap \
  --class tcpfn.temporal.causalfm_prior:CausalFMPrior \
  --slug tcpfn-causalfm --out ./port-priors --kind temporal_causal \
  -r tcpfn -r torch

# (hand-edit each port-priors/priors/<slug>/prior.yaml — outputs.variables)

# Build wheel and stage
python -m build --wheel --outdir /tmp/wheels
for slug in tcpfn-discovery tcpfn-causalfm; do
  mkdir -p port-priors/priors/$slug/wheels
  cp /tmp/wheels/tcpfn-*.whl port-priors/priors/$slug/wheels/
done

# Create "my-tcpfn" project in the web UI, then:
export PFNSTUDIO_TOKEN=ps_…
pfnstudio push ./port-priors --project-id my-tcpfn
```

Open http://localhost:4200/projects/my-tcpfn/priors and click `tcpfn-discovery`.

---

## See also

- [USAGE.md](../USAGE.md) — `./run.sh` ops handbook.
- [getting-started.md](getting-started.md) — authoring priors from scratch (no existing package).
- [architecture.md](architecture.md) — how the API, CLI, and web UI fit together.
