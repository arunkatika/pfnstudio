"""Validate FM project artifacts against JSON schemas."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from ._paths import schemas_root


def _coerce(value: Any) -> Any:
    """YAML parses bare ISO dates as datetime.date, but our schemas declare them as
    string-with-format-date. Coerce date/datetime to ISO strings recursively before
    validation so users can write either quoted or unquoted dates."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return _coerce(yaml.safe_load(f))


def _load_schema(name: str) -> dict:
    return json.loads((schemas_root() / f"{name}.schema.json").read_text())


def _validate(instance: dict, schema: dict, source: Path) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = []
    for err in validator.iter_errors(instance):
        path = ".".join(str(p) for p in err.absolute_path) or "(root)"
        errors.append(f"{source}: {path}: {err.message}")
    return errors


def _validate_initiative_frontmatter(md_path: Path, schema: dict) -> list[str]:
    text = md_path.read_text()
    if not text.startswith("---"):
        return [f"{md_path}: missing YAML frontmatter"]
    end = text.find("---", 3)
    if end == -1:
        return [f"{md_path}: unterminated YAML frontmatter"]
    fm = _coerce(yaml.safe_load(text[3:end]))
    if not isinstance(fm, dict):
        return [f"{md_path}: frontmatter is not a mapping"]
    return _validate(fm, schema, md_path)


def validate_project(project: Path) -> list[str]:
    errors: list[str] = []

    prior_schema = _load_schema("prior")
    for prior_yaml in (project / "priors").rglob("prior.yaml"):
        errors.extend(_validate(_load_yaml(prior_yaml), prior_schema, prior_yaml))

    model_schema = _load_schema("model")
    for model_yaml in (project / "models").glob("*.yaml"):
        errors.extend(_validate(_load_yaml(model_yaml), model_schema, model_yaml))

    eval_schema = _load_schema("eval")
    for eval_yaml in (project / "evals").glob("*.yaml"):
        errors.extend(_validate(_load_yaml(eval_yaml), eval_schema, eval_yaml))

    run_schema = _load_schema("run")
    for run_yaml in (project / "runs").glob("*.yaml"):
        errors.extend(_validate(_load_yaml(run_yaml), run_schema, run_yaml))

    initiative_schema = _load_schema("initiative")
    for md in (project / "initiatives").glob("*.md"):
        errors.extend(_validate_initiative_frontmatter(md, initiative_schema))

    return errors
