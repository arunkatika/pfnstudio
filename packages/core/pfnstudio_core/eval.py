"""Eval abstraction. An Eval is a benchmark configuration plus a scoring function."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class DatasetRef(BaseModel):
    name: str
    source: str | None = None
    version: str | None = None
    split: str | None = None


class MetricSpec(BaseModel):
    name: str
    higher_is_better: bool | None = None
    description: str | None = None


class BaselineEntry(BaseModel):
    name: str
    score: float | None = None
    source: str | None = None


class EvalSpec(BaseModel):
    id: str
    name: str
    version: str
    task: str
    description: str | None = None
    dataset: DatasetRef
    metrics: list[MetricSpec]
    baselines: list[BaselineEntry] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class Eval(ABC):
    """Subclass and decorate with @register_eval(\"<id>\") to bind to an EvalSpec."""

    spec: EvalSpec

    @abstractmethod
    def score(self, predictions: Any, ground_truth: Any) -> dict[str, float]:
        """Return a dict keyed by metric name, matching `spec.metrics[*].name`."""
