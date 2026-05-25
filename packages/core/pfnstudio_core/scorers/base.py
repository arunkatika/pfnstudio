"""DatasetScorer abstract base — load dataset → run model → return metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..datasets import RegistryDatasetLoader
from ..eval import EvalSpec
from ..run import RunSpec


@dataclass
class ScorerResult:
    """Outcome of running a scorer on a trained model.

    `metrics` is the dict of metric_name → value the scorer computed
    (matching the names declared on `eval_spec.metrics`).
    `meta` carries any additional context the runner / UI may surface,
    e.g. number of series scored, baseline values, sample size.
    `skipped` set to True means the scorer couldn't run (e.g. dataset
    not available, missing feature dimension) and the runner should
    surface that reason rather than treat absence as 0-valued metrics.
    """

    metrics: dict[str, float]
    meta: dict[str, Any]
    skipped: bool = False
    skip_reason: str | None = None


class DatasetScorer(ABC):
    """Compute eval metrics from a registry-loaded dataset and trained model.

    Lifecycle: instantiated once at import time (BUILTIN_SCORERS is a
    module-level dict), called per (model, eval_spec) pair by the
    local adapter after training. Implementations should be stateless
    — keep any cached work inside `score()` scope.
    """

    @abstractmethod
    def score(
        self,
        *,
        model: Any,
        eval_spec: EvalSpec,
        loader: RegistryDatasetLoader,
        run_spec: RunSpec,
    ) -> ScorerResult: ...
