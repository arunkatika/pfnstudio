"""Render an FM project as a static site."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import markdown as md
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_STATIC = Path(__file__).resolve().parent / "static"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATES),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["markdown"] = lambda text: md.markdown(
        text or "", extensions=["fenced_code", "tables"]
    )
    return env


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _read_initiative(path: Path) -> dict:
    text = path.read_text()
    fm: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm = yaml.safe_load(text[3:end]) or {}
            body = text[end + 3 :].lstrip()
    return {"frontmatter": fm, "body": body, "path": path}


def _gather(project_root: Path) -> dict[str, Any]:
    root = project_root
    priors = []
    for prior_yaml in sorted((root / "priors").rglob("prior.yaml")):
        spec = _read_yaml(prior_yaml)
        prior_md = prior_yaml.parent / "prior.md"
        spec["_doc"] = prior_md.read_text() if prior_md.exists() else ""
        spec["_dir"] = prior_yaml.parent.name
        priors.append(spec)

    models = [_read_yaml(p) for p in sorted((root / "models").glob("*.yaml"))]
    evals = [_read_yaml(p) for p in sorted((root / "evals").glob("*.yaml"))]
    runs = [_read_yaml(p) for p in sorted((root / "runs").glob("*.yaml"))]

    initiatives = [_read_initiative(p) for p in sorted((root / "initiatives").glob("*.md"))]

    roadmap = (root / "ROADMAP.md").read_text() if (root / "ROADMAP.md").exists() else ""
    readme = (root / "README.md").read_text() if (root / "README.md").exists() else ""

    lit_summaries = []
    summaries_dir = root / "literature" / "summaries"
    if summaries_dir.exists():
        for md_path in sorted(summaries_dir.glob("*.md")):
            lit_summaries.append(_read_initiative(md_path))

    bib = (
        (root / "literature" / "references.bib").read_text()
        if (root / "literature" / "references.bib").exists()
        else ""
    )

    return {
        "project_name": root.name,
        "readme": readme,
        "roadmap": roadmap,
        "initiatives": initiatives,
        "priors": priors,
        "models": models,
        "evals": evals,
        "runs": runs,
        "lit_summaries": lit_summaries,
        "bib": bib,
    }


def build_site(project_root: Path, out_dir: Path) -> Path:
    project_root = project_root.resolve()
    out_dir = (project_root / out_dir).resolve() if not out_dir.is_absolute() else out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if _STATIC.exists():
        shutil.copytree(_STATIC, out_dir / "static")

    data = _gather(project_root)
    env = _env()

    pages = {
        "index.html": "index.html.j2",
        "roadmap.html": "roadmap.html.j2",
        "initiatives.html": "initiatives.html.j2",
        "priors.html": "priors.html.j2",
        "models.html": "models.html.j2",
        "evals.html": "evals.html.j2",
        "runs.html": "runs.html.j2",
        "literature.html": "literature.html.j2",
    }
    for filename, template_name in pages.items():
        template = env.get_template(template_name)
        (out_dir / filename).write_text(template.render(**data))

    return out_dir
