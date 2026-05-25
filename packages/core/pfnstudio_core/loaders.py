"""Load YAML artifacts into typed pydantic objects."""

from __future__ import annotations

from pathlib import Path

import yaml

from .eval import EvalSpec
from .model import ModelSpec
from .prior import PriorSpec
from .run import RunSpec


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_prior(path: Path) -> PriorSpec:
    return PriorSpec.model_validate(_read_yaml(path))


def load_model(path: Path) -> ModelSpec:
    return ModelSpec.model_validate(_read_yaml(path))


def load_eval(path: Path) -> EvalSpec:
    return EvalSpec.model_validate(_read_yaml(path))


def load_run(path: Path) -> RunSpec:
    return RunSpec.model_validate(_read_yaml(path))


def load_project(project_root: Path) -> dict[str, list]:
    """Discover every artifact under a project root."""
    root = Path(project_root)
    return {
        "priors": [load_prior(p) for p in (root / "priors").rglob("prior.yaml")],
        "models": [load_model(p) for p in (root / "models").glob("*.yaml")],
        "evals": [load_eval(p) for p in (root / "evals").glob("*.yaml")],
        "runs": [load_run(p) for p in (root / "runs").glob("*.yaml")],
    }
