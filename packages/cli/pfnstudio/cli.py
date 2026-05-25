"""PFN Studio CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .compute import ADAPTERS, get_adapter
from .lint import lint_project
from .scaffold import scaffold_project
from .tracking import select_trackers
from .validate import validate_project

app = typer.Typer(
    name="priorstudio",
    help="Build, validate, and run prior-fitted foundation model projects.",
    no_args_is_help=True,
)

studio_app = typer.Typer(
    name="studio", help="Static-site Studio: render the project as a browsable site."
)
app.add_typer(studio_app, name="studio")

author_app = typer.Typer(
    name="author", help="Local prior-authoring helpers (no API access required)."
)
app.add_typer(author_app, name="author")

console = Console()


@app.command()
def version() -> None:
    """Print the CLI version."""
    console.print(f"priorstudio {__version__}")


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name (kebab-case)."),
    path: Path = typer.Option(Path.cwd(), help="Parent directory."),
    description: str = typer.Option("A prior-fitted foundation model project."),
    org: str = typer.Option("your-org", help="GitHub org/user for repo URL substitutions."),
) -> None:
    """Scaffold a new FM project from the template."""
    target = path / name
    if target.exists():
        console.print(f"[red]Refusing to overwrite existing path:[/red] {target}")
        raise typer.Exit(code=1)
    scaffold_project(target=target, project_name=name, description=description, org=org)
    console.print(f"[green]Created[/green] {target}")


@app.command()
def validate(project: Path = typer.Argument(Path.cwd())) -> None:
    """Validate every artifact against its JSON Schema."""
    errors = validate_project(project)
    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        raise typer.Exit(code=1)
    console.print("[green]✓ All artifacts valid.[/green]")


@app.command()
def lint(project: Path = typer.Argument(Path.cwd())) -> None:
    """Cross-reference lint: do runs reference priors/models/evals that exist?"""
    errors = lint_project(project)
    if errors:
        for err in errors:
            console.print(f"[red]✗[/red] {err}")
        raise typer.Exit(code=1)
    console.print("[green]✓ All cross-references valid.[/green]")


@app.command()
def list_artifacts(project: Path = typer.Argument(Path.cwd())) -> None:
    """List every artifact in a project."""
    table = Table(title=str(project))
    table.add_column("kind")
    table.add_column("id")
    table.add_column("version")
    table.add_column("path")
    for prior_yaml in (project / "priors").rglob("prior.yaml"):
        import yaml as _y

        spec = _y.safe_load(prior_yaml.read_text()) or {}
        table.add_row(
            "prior",
            str(spec.get("id", "?")),
            str(spec.get("version", "?")),
            str(prior_yaml.relative_to(project)),
        )
    for model_yaml in (project / "models").glob("*.yaml"):
        import yaml as _y

        spec = _y.safe_load(model_yaml.read_text()) or {}
        table.add_row(
            "model",
            str(spec.get("id", "?")),
            str(spec.get("version", "?")),
            str(model_yaml.relative_to(project)),
        )
    for eval_yaml in (project / "evals").glob("*.yaml"):
        import yaml as _y

        spec = _y.safe_load(eval_yaml.read_text()) or {}
        table.add_row(
            "eval",
            str(spec.get("id", "?")),
            str(spec.get("version", "?")),
            str(eval_yaml.relative_to(project)),
        )
    for run_yaml in (project / "runs").glob("*.yaml"):
        import yaml as _y

        spec = _y.safe_load(run_yaml.read_text()) or {}
        table.add_row("run", str(spec.get("id", "?")), "—", str(run_yaml.relative_to(project)))
    console.print(table)


@app.command()
def sample(
    project: Path = typer.Argument(..., help="Project root containing priors/ and prior code."),
    prior: str = typer.Argument(..., help="Prior slug (matches priors/<slug>/ directory)."),
    count: int = typer.Option(9, help="How many tasks to sample."),
    seed: int = typer.Option(42, help="Base seed; per-sample seeds = base + i."),
    overrides_json: str = typer.Option("{}", help="JSON dict of parameter overrides."),
) -> None:
    """Sample N tasks from a prior and emit JSON to stdout.

    Used by the API's Visual Prior Inspector. Output is a single JSON object:
        {
          "prior_id": ...,
          "version": ...,
          "kind": ...,
          "outputs": [...],
          "samples": [ { "seed": 42, "X": [...], "y": [...], ... }, ... ]
        }
    Numpy arrays are converted to nested lists. Non-serialisable values become null.
    """
    import sys

    project = project.resolve()
    sys.path.insert(0, str(project))

    from pfnstudio_core.loaders import load_prior
    from pfnstudio_core.registry import discover_in_project, get_prior, list_priors

    # rich's Console.print() doesn't accept err=; use a separate stderr Console
    # for diagnostics so they don't pollute the JSON stdout the API parses.
    err_console = Console(stderr=True)

    prior_dir = project / "priors" / prior
    prior_yaml = prior_dir / "prior.yaml"
    if not prior_yaml.exists():
        err_console.print(f"[red]not found:[/red] {prior_yaml}")
        raise typer.Exit(code=1)

    spec = load_prior(prior_yaml)
    discover_in_project(project)
    try:
        prior_cls = get_prior(spec.id)
    except KeyError as e:
        # Fork tolerance: a forked prior gets a new slug but inherits its
        # parent's @register_prior("...") decorator unchanged. When the API
        # exports a single-prior project to a temp dir for sampling, the
        # registry then has exactly one entry — whatever the decorator names
        # it. Use that one regardless of the slug the YAML thinks it has.
        registered = list_priors()
        if len(registered) == 1:
            prior_cls = get_prior(registered[0])
        else:
            err_console.print(f"[red]registry miss:[/red] {e}")
            err_console.print(f"  registered priors: {registered}")
            raise typer.Exit(code=1) from e

    prior_obj = prior_cls()
    prior_obj.spec = spec

    overrides = json.loads(overrides_json)

    def _to_jsonable(v):
        try:
            import numpy as np

            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.floating, np.integer)):
                return v.item()
        except ImportError:
            pass
        try:
            import torch

            if isinstance(v, torch.Tensor):
                return v.detach().cpu().tolist()
        except ImportError:
            pass
        if isinstance(v, dict):
            return {k: _to_jsonable(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        return None  # silently drop non-serialisable

    samples = []
    for i in range(count):
        s = prior_obj.sample(seed=seed + i, **overrides)
        samples.append({"seed": seed + i, **{k: _to_jsonable(v) for k, v in s.items()}})

    payload = {
        "prior_id": spec.id,
        "name": spec.name,
        "version": spec.version,
        "kind": spec.kind,
        "outputs": [o.model_dump() for o in spec.outputs.variables],
        "parameters": {k: p.model_dump() for k, p in spec.parameters.items()},
        "overrides": overrides,
        "samples": samples,
    }
    print(json.dumps(payload))


@app.command()
def push(
    project_dir: Path = typer.Argument(
        ...,
        help="Local project directory in canonical PFN Studio layout (priors/, models/, evals/, runs/).",
    ),
    project_id: str = typer.Option(
        ..., "--project-id", help="Target project ID or slug in the PFN Studio API."
    ),
    api_url: str = typer.Option(
        "http://localhost:3000/api", "--api-url", envvar="PFNSTUDIO_API_URL"
    ),
    token: str = typer.Option(
        ...,
        "--token",
        envvar="PFNSTUDIO_TOKEN",
        help="API token (issue one at /api-tokens in the web UI).",
    ),
    only: str = typer.Option(
        "priors,models", help="Comma-separated kinds to push: priors,models,evals,runs."
    ),
    skip_existing: bool = typer.Option(
        True, help="Skip artifacts whose slug already exists in the target project."
    ),
) -> None:
    """Bulk-upload artifacts from a local project directory to PFN Studio.

    Example:
        priorstudio push my-tcpfn-priors --project-id tcpfn-port --token ps_xxx
    """
    import requests
    import yaml as _y

    project_dir = project_dir.resolve()
    if not project_dir.exists():
        console.print(f"[red]not found:[/red] {project_dir}")
        raise typer.Exit(code=1)

    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    base = api_url.rstrip("/")
    kinds = [k.strip() for k in only.split(",") if k.strip()]

    # Resolve the project (slug or id)
    res = requests.get(f"{base}/projects/{project_id}", headers=headers, timeout=10)
    if res.status_code == 401:
        _die_token()
    if res.status_code != 200:
        console.print(f"[red]project lookup failed:[/red] {res.status_code} {res.text[:300]}")
        raise typer.Exit(code=1)
    resolved_id = res.json()["id"]
    console.print(f"[blue]→[/blue] target project: {res.json()['name']} ({resolved_id})")

    pushed, skipped, failed = 0, 0, 0

    if "priors" in kinds:
        for prior_yaml in sorted((project_dir / "priors").rglob("prior.yaml")):
            spec = _y.safe_load(prior_yaml.read_text()) or {}
            prior_dir = prior_yaml.parent
            code_path = prior_dir / spec.get("implementation", "prior.py")
            rationale_path = prior_dir / "prior.md"
            requirements_path = prior_dir / "requirements.txt"
            wheel_dir = prior_dir / "wheels"

            body = {
                "name": spec.get("name") or spec["id"],
                "slug": spec["id"],
                "version": spec["version"],
                "kind": spec["kind"],
                "description": spec.get("description"),
                "rationale": rationale_path.read_text() if rationale_path.exists() else None,
                "parameters": spec.get("parameters") or {},
                "outputs": spec.get("outputs") or {"variables": []},
                "citations": spec.get("citations") or [],
                "code": code_path.read_text() if code_path.exists() else None,
                "requirements": requirements_path.read_text()
                if requirements_path.exists()
                else None,
                "pythonVersion": spec.get("python_version") or spec.get("pythonVersion"),
            }
            r = requests.post(
                f"{base}/projects/{resolved_id}/priors",
                headers=headers,
                json=body,
                timeout=30,
            )

            # Did we end up with a server-side prior to attach wheels to?
            target_prior_id: str | None = None
            if r.status_code in (200, 201):
                created = r.json()
                target_prior_id = created["id"]
                console.print(f"[green]✓[/green] prior {spec['id']}")
                pushed += 1
            elif r.status_code == 409 or "unique" in (r.text or "").lower():
                if skip_existing:
                    # Look up the existing prior so we can still attach/refresh wheels.
                    rg = requests.get(
                        f"{base}/projects/{resolved_id}/priors/{spec['id']}",
                        headers=headers,
                        timeout=10,
                    )
                    if rg.status_code == 200:
                        target_prior_id = rg.json()["id"]
                        console.print(
                            f"[yellow]⤳[/yellow] prior {spec['id']} already exists — skipping body, will refresh wheels"
                        )
                        skipped += 1
                    else:
                        console.print(
                            f"[yellow]⤳[/yellow] prior {spec['id']} already exists — could not look up id ({rg.status_code}); skipping wheels too"
                        )
                        skipped += 1
                else:
                    console.print(
                        f"[red]✗[/red] prior {spec['id']} conflict; pass --no-skip-existing to fail loudly"
                    )
                    failed += 1
            else:
                console.print(f"[red]✗[/red] prior {spec['id']}: {r.status_code} {r.text[:200]}")
                failed += 1

            # Upload any wheels in the local wheels/ dir to whichever prior we resolved.
            if target_prior_id:
                wheels = (
                    sorted(
                        [
                            p
                            for p in wheel_dir.glob("*")
                            if p.suffix in (".whl",) or p.name.endswith(".tar.gz")
                        ]
                    )
                    if wheel_dir.is_dir()
                    else []
                )
                for wheel_path in wheels:
                    with wheel_path.open("rb") as fh:
                        rw = requests.post(
                            f"{base}/projects/{resolved_id}/priors/{target_prior_id}/wheels",
                            headers={"Authorization": f"Token {token}"},
                            files={"file": (wheel_path.name, fh, "application/octet-stream")},
                            timeout=120,
                        )
                    if rw.status_code in (200, 201):
                        console.print(f"  [green]+[/green] wheel {wheel_path.name}")
                    else:
                        console.print(
                            f"  [red]✗[/red] wheel {wheel_path.name}: {rw.status_code} {rw.text[:160]}"
                        )

    if "models" in kinds:
        for model_yaml in sorted((project_dir / "models").glob("*.yaml")):
            spec = _y.safe_load(model_yaml.read_text()) or {}

            # Translate the canonical YAML keys (input_shape, output_heads,
            # snake_case) into the API's camelCase DTO. The CLI is the
            # YAML-side bridge — we don't want callers writing the API's
            # internal naming directly.
            body = {
                "name": spec.get("name") or spec["id"],
                "slug": spec["id"],
                "version": spec["version"],
                "description": spec.get("description"),
                "inputShape": spec.get("input_shape") or spec.get("inputShape"),
                "blocks": spec.get("blocks") or [],
                "outputHeads": spec.get("output_heads") or spec.get("outputHeads") or [],
                "citations": spec.get("citations") or [],
            }
            r = requests.post(
                f"{base}/projects/{resolved_id}/models",
                headers=headers,
                json=body,
                timeout=30,
            )
            if r.status_code in (200, 201):
                console.print(f"[green]✓[/green] model {spec['id']}")
                pushed += 1
            elif r.status_code == 409 or "unique" in (r.text or "").lower():
                if skip_existing:
                    console.print(f"[yellow]⤳[/yellow] model {spec['id']} already exists — skipped")
                    skipped += 1
                else:
                    console.print(
                        f"[red]✗[/red] model {spec['id']} conflict; pass --no-skip-existing to fail loudly"
                    )
                    failed += 1
            else:
                console.print(f"[red]✗[/red] model {spec['id']}: {r.status_code} {r.text[:200]}")
                failed += 1

    console.print()
    console.print(f"[bold]done:[/bold] {pushed} pushed, {skipped} skipped, {failed} failed.")
    if failed > 0:
        raise typer.Exit(code=1)


def _die_token() -> None:
    console.print(
        "[red]401 unauthorized.[/red] Issue an API token at the web UI's /api-tokens page, then export PFNSTUDIO_TOKEN=ps_xxx"
    )
    raise typer.Exit(code=1)


# ──────────────────────────────────────────────────────────────────────────
# `priorstudio author wrap` — generate a PFN Studio prior from any
# Python class with a `sample_batch(batch_size, seed, **kw)` method (or any
# entry-point you can describe).
#
# Runs locally in the user's own venv — that venv must have the source class
# importable (e.g. via `pip install -e .` or `pip install some_pkg`).
# ──────────────────────────────────────────────────────────────────────────
@author_app.command("wrap")
def author_wrap(
    target: str = typer.Option(
        ...,
        "--class",
        help="Importable class, e.g. tcpfn.discovery.discovery_prior:CausalDiscoveryPrior",
    ),
    slug: str = typer.Option(..., help="Slug for the new PFN Studio prior."),
    out: Path = typer.Option(
        Path.cwd(), help="Output project root (priors/<slug>/ will be created here)."
    ),
    requires: list[str] = typer.Option(
        [],
        "--requires",
        "-r",
        help="pip-installable lines for requirements.txt. Repeat the flag (-r tcpfn -r torch>=2.2).",
    ),
    sample_method: str = typer.Option(
        "sample_batch",
        help="Method name on the class that produces samples (default: sample_batch).",
    ),
    name: str = typer.Option(None, help="Display name (default: derived from class name)."),
    kind: str = typer.Option(
        "custom", help="Prior kind: scm, tabular, temporal, temporal_causal, graph, custom."
    ),
) -> None:
    """Generate a PFN Studio-format prior wrapping any Python class.

    Introspects the class constructor to populate parameters, emits a
    PFN Studio-canonical directory:

        priors/<slug>/
          ├── prior.py           — adapter calling the wrapped class
          ├── prior.yaml         — spec with parameters from __init__
          ├── prior.md           — placeholder rationale
          └── requirements.txt   — pip-installable deps

    After running this for each prior you want to migrate, push them with:
        priorstudio push <out-dir> --project-id <slug>
    """
    import importlib
    import inspect

    err_console = Console(stderr=True)

    if ":" not in target:
        err_console.print("[red]--class must look like 'package.module:ClassName'[/red]")
        raise typer.Exit(code=1)

    module_path, class_name = target.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        err_console.print(f"[red]Could not import {module_path}:[/red] {e}")
        err_console.print(
            "[yellow]Hint:[/yellow] are you in the venv that has this package installed?"
        )
        raise typer.Exit(code=1) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        err_console.print(f"[red]No class named {class_name} in {module_path}[/red]")
        raise typer.Exit(code=1)

    # Introspect __init__ params (skip self, and skip `seed` since the wrapper
    # already exposes seed as its own required argument — emitting it again
    # would cause a duplicate-kwarg SyntaxError).
    SKIP = {"self", "seed"}
    init_accepts_seed = False
    try:
        init_sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        err_console.print(
            f"[yellow]Warning:[/yellow] could not introspect {class_name}.__init__; emitting empty parameters."
        )
        init_params = []
    else:
        init_accepts_seed = "seed" in init_sig.parameters
        init_params = [
            (p.name, p)
            for p in init_sig.parameters.values()
            if p.name not in SKIP
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

    sample_method_name = sample_method
    has_sample_method = hasattr(cls, sample_method_name)
    if not has_sample_method:
        err_console.print(
            f"[yellow]Warning:[/yellow] {class_name} has no .{sample_method_name}(); the wrapper will be a stub you must edit."
        )

    # ── Generate prior.py ──
    prior_dir = out / "priors" / slug
    prior_dir.mkdir(parents=True, exist_ok=True)

    # Class name for the wrapper, e.g. WrappedCausalDiscoveryPrior
    wrapper_class = "Wrapped" + class_name + ("" if class_name.endswith("Prior") else "Prior")

    init_arg_lines = []
    init_kwargs_pass = []
    for pname, pinfo in init_params:
        default = pinfo.default
        if default is inspect.Parameter.empty:
            init_arg_lines.append(f"        {pname},")
            init_kwargs_pass.append(f"{pname}={pname}")
        else:
            # Python repr of the default. For non-trivial defaults, fall back to a placeholder.
            try:
                rep = repr(default)
                # Avoid emitting class instances or unrepresentable defaults verbatim
                if rep.startswith("<") or "object at 0x" in rep:
                    rep = "None  # FIXME: introspected default could not be serialised"
            except Exception:
                rep = "None  # FIXME: introspected default could not be serialised"
            init_arg_lines.append(f"        {pname}={rep},")
            init_kwargs_pass.append(f"{pname}={pname}")
    if init_accepts_seed:
        # Forward the wrapper's seed into the wrapped class's __init__ so
        # sampling is deterministic per seed even when sample_batch() itself
        # doesn't accept a seed argument.
        init_kwargs_pass.insert(0, "seed=seed")
    init_args_block = "\n".join(init_arg_lines)
    init_kwargs_block = ", ".join(init_kwargs_pass)

    # Build sample_batch call kwargs based on what the method actually accepts.
    sample_batch_kwargs: list[str] = []
    sample_batch_fixme = ""
    if has_sample_method:
        try:
            sb_sig = inspect.signature(getattr(cls, sample_method_name))
            sb_params = sb_sig.parameters
            if "batch_size" in sb_params:
                sample_batch_kwargs.append("batch_size=1")
            if "seed" in sb_params:
                sample_batch_kwargs.append("seed=seed")
            if "n_context_units" in sb_params:
                sample_batch_kwargs.append("n_context_units=10")
                sample_batch_fixme = "        # FIXME: n_context_units / n_query_units default to 10/5; tune for your task.\n"
            if "n_query_units" in sb_params:
                sample_batch_kwargs.append("n_query_units=5")
        except (TypeError, ValueError):
            sample_batch_kwargs = ["batch_size=1"]

    sb_call = ", ".join(sample_batch_kwargs)
    sample_call_block = (
        f"        instance = self._cls({init_kwargs_block})\n"
        f"{sample_batch_fixme}"
        f"        batch = instance.{sample_method_name}({sb_call})\n"
        f"        if isinstance(batch, dict):\n"
        f"            return {{k: _drop_batch_dim(v) for k, v in batch.items()}}\n"
        f"        return {{'sample': batch}}"
        if has_sample_method
        else "        raise NotImplementedError('sample method not found on the wrapped class — edit this file')"
    )

    prior_py = f'''"""Auto-generated wrapper for {module_path}:{class_name}.

Generated by `priorstudio author wrap`. Edit freely — this is just a starting
point. The PFN Studio sampler will install requirements.txt before importing
this file, so the wrapped class needs to be available via that.
"""

from __future__ import annotations
from typing import Any

from pfnstudio_core import Prior, register_prior

# Import the original class. This lookup must succeed in the cached install
# environment that PFN Studio creates from requirements.txt.
from {module_path} import {class_name} as _Wrapped


def _drop_batch_dim(v: Any) -> Any:
    """If v looks like a tensor/array with a leading batch dim of size 1, drop it."""
    shape = getattr(v, "shape", None)
    if shape is not None and len(shape) > 0 and shape[0] == 1:
        return v[0]
    return v


@register_prior("{slug}")
class {wrapper_class}(Prior):
    """Adapter exposing {class_name} through PFN Studio's Prior API."""

    _cls = _Wrapped

    def sample(
        self,
        *,
        seed: int,
{init_args_block}
        **_: Any,
    ) -> dict[str, Any]:
{sample_call_block}
'''

    (prior_dir / "prior.py").write_text(prior_py)

    # ── Generate prior.yaml ──
    yaml_params: dict = {}
    for pname, pinfo in init_params:
        default = pinfo.default
        ann = pinfo.annotation
        ptype = "float"  # safe default
        # Order matters: bool is a subclass of int in Python, so check bool first.
        if ann is bool:
            ptype = "bool"
        elif ann is int:
            ptype = "int"
        elif ann is float:
            ptype = "float"
        elif ann is str:
            ptype = "str"
        entry: dict = {"type": ptype, "description": f"From {class_name}.__init__"}
        if default is not inspect.Parameter.empty and isinstance(default, (int, float)):
            entry["default"] = default
        yaml_params[pname] = entry

    display_name = name or " ".join(
        word for word in [class_name.replace("Prior", ""), "Prior"] if word
    )

    yaml_body = f"""id: {slug}
name: {display_name}
version: 0.1.0
kind: {kind}
description: |
  Wrapped from {module_path}:{class_name}.
  Generated by `priorstudio author wrap`. Edit this file to fill in
  output variables and tighten parameter ranges.

parameters:
"""
    for pname, entry in yaml_params.items():
        yaml_body += f"  {pname}:\n"
        for k, v in entry.items():
            if isinstance(v, str):
                yaml_body += f'    {k}: "{v}"\n'
            else:
                yaml_body += f"    {k}: {v}\n"

    yaml_body += """
outputs:
  variables:
    # FIXME: fill in based on what the wrapped sample method returns.
    - name: data
      type: tensor
      description: Sample output from the wrapped class

implementation: prior.py
"""
    (prior_dir / "prior.yaml").write_text(yaml_body)

    # ── Generate prior.md ──
    md = f"""# {display_name}

Wrapper around `{module_path}:{class_name}`.

## TODO before publishing

- [ ] Verify the `sample()` method correctly unwraps the batch returned by the original class
- [ ] Fill in `outputs.variables` in prior.yaml with actual shapes and descriptions
- [ ] Tighten parameter ranges in prior.yaml (e.g., add `range: [lo, hi]` for sliders)
- [ ] Test locally with `priorstudio sample <project-dir> {slug} --count 3`
- [ ] Push with `priorstudio push <project-dir> --project-id <project>`

## Why this prior exists

(Edit this section to describe the pedagogical or research purpose.)
"""
    (prior_dir / "prior.md").write_text(md)

    # ── requirements.txt ──
    if requires:
        (prior_dir / "requirements.txt").write_text("\n".join(requires) + "\n")

    console.print(f"[green]✓[/green] Wrote {prior_dir}/")
    console.print(
        f"  Files: prior.py, prior.yaml, prior.md{', requirements.txt' if requires else ''}"
    )
    console.print()
    console.print("[bold]Next:[/bold]")
    console.print(f"  1. Edit [cyan]{prior_dir}/prior.yaml[/cyan] — fill in outputs[].variables")
    console.print(f"  2. Edit [cyan]{prior_dir}/prior.py[/cyan] — verify the sample() unwrap logic")
    console.print(f"  3. Test: [cyan]priorstudio sample {out} {slug} --count 3[/cyan]")
    console.print(f"  4. Push: [cyan]priorstudio push {out} --project-id <slug>[/cyan]")


@app.command()
def run(
    manifest: Path = typer.Argument(..., help="Path to a run YAML."),
    project: Path = typer.Option(None, help="Project root (default: parent of runs/)."),
    target: str = typer.Option(
        None, help=f"Override compute target. Available: {sorted(ADAPTERS)}"
    ),
) -> None:
    """Execute a run via the configured (or overridden) compute adapter."""
    manifest = manifest.resolve()
    if project is None:
        if manifest.parent.name == "runs":
            project = manifest.parent.parent
        else:
            project = Path.cwd()
    project = project.resolve()

    import yaml as _y

    spec = _y.safe_load(manifest.read_text())
    chosen_target = target or (spec.get("compute") or {}).get("target", "local")
    console.print(f"[blue]→[/blue] dispatching {spec.get('id')} via [bold]{chosen_target}[/bold]")

    adapter = get_adapter(chosen_target)

    from pfnstudio_core.run import RunSpec

    run_spec = RunSpec.model_validate(spec)
    trackers = select_trackers(run_spec.tracking)
    for t in trackers:
        t.start(run_id=run_spec.id, config=spec, project_root=project)

    try:
        results = adapter.submit(run_yaml=manifest, project_root=project)
    finally:
        for t in trackers:
            t.finish(results if "results" in dir() else {})

    console.print_json(json.dumps(results, default=str))
    if results.get("status") == "error":
        raise typer.Exit(code=1)


@app.command()
def predict(
    manifest: Path = typer.Argument(..., help="Path to a run YAML."),
    checkpoint: Path = typer.Option(
        ..., help="Path to the checkpoint directory (contains model.pt + topology.json)."
    ),
    input_json: Path = typer.Option(
        None,
        "--input",
        help="Path to a JSON file with the inference payload. Reads stdin if omitted.",
    ),
    project: Path = typer.Option(None, help="Project root (default: parent of runs/)."),
) -> None:
    """Load a trained checkpoint and run inference.

    The input JSON shape depends on the prior. For a regression prior
    (linear-regression-style PFN) it should be::

        {
          "context": {"x": [[1.0],[2.0]], "y": [2.0, 4.1]},
          "query":   {"x": [[3.0],[4.0]]}
        }

    Output JSON shape::

        {"predictions": [<y for each query x>]}

    Output is emitted on stdout so the API can capture it. Errors are
    surfaced as ``{"error": "..."}`` with a non-zero exit code.
    """
    import json as _json
    import sys as _sys

    manifest = manifest.resolve()
    if project is None:
        project = manifest.parent.parent if manifest.parent.name == "runs" else Path.cwd()
    project = project.resolve()

    payload_text = input_json.read_text() if input_json is not None else _sys.stdin.read()
    try:
        payload = _json.loads(payload_text)
    except _json.JSONDecodeError as e:
        console.print_json(_json.dumps({"error": f"invalid input JSON: {e}"}))
        raise typer.Exit(code=1) from e

    # Lazy-import the inference module so the CLI's other commands
    # (init / validate / lint) don't pay torch's import cost.
    try:
        from pfnstudio_core.training.predict import run_inference
    except ImportError as e:
        console.print_json(_json.dumps({"error": f"predict module unavailable: {e}"}))
        raise typer.Exit(code=1) from e

    # Optional `tag` field for promptable priors. Extracted from the
    # top-level payload and passed as a separate argument so the rest
    # of the payload (context/query/X) keeps its existing shape.
    # Brains trained without promptable_training ignore the tag (no
    # tag_embedder → no tag pathway).
    tag = payload.pop("tag", None) if isinstance(payload, dict) else None
    # Optional `detect` flag — when True and the checkpoint ships a
    # companion detector, the response includes a `detected_tag`
    # field with the detector's per-axis proposal.
    detect_flag = bool(payload.pop("detect", False)) if isinstance(payload, dict) else False

    try:
        out = run_inference(
            manifest_path=manifest,
            project_root=project,
            checkpoint_dir=checkpoint.resolve(),
            payload=payload,
            tag=tag,
            detect=detect_flag,
        )
    except Exception as e:
        console.print_json(
            _json.dumps(
                {
                    "error": f"{type(e).__name__}: {e}",
                }
            )
        )
        raise typer.Exit(code=1) from e

    # Emit ONLY the JSON output on stdout so the API can parse it
    # cleanly. Anything else (progress, logs) should go on stderr.
    _sys.stdout.write(_json.dumps(out) + "\n")
    _sys.stdout.flush()


@studio_app.command("build")
def studio_build(
    project: Path = typer.Argument(Path.cwd()),
    out: Path = typer.Option(Path("_studio"), help="Output directory."),
) -> None:
    """Render a static site for the project."""
    try:
        from pfnstudio_studio.build import build_site
    except ImportError as e:
        console.print(f"[red]pfnstudio-studio not installed:[/red] {e}")
        console.print("Install: pip install pfnstudio-studio")
        raise typer.Exit(code=1) from e
    out_dir = build_site(project_root=project, out_dir=out)
    console.print(f"[green]Built site at[/green] {out_dir}")


@studio_app.command("serve")
def studio_serve(
    project: Path = typer.Argument(Path.cwd()),
    port: int = typer.Option(8000),
) -> None:
    """Build the static site and serve it locally."""
    try:
        from pfnstudio_studio.build import build_site
        from pfnstudio_studio.serve import serve_dir
    except ImportError as e:
        console.print(f"[red]pfnstudio-studio not installed:[/red] {e}")
        raise typer.Exit(code=1) from e
    out_dir = build_site(project_root=project, out_dir=Path("_studio"))
    serve_dir(out_dir, port=port)


if __name__ == "__main__":
    app()
