"""Eval spec — the declarative benchmark definition (dataset + metrics +
baselines). The *scoring* half lives in a DatasetScorer (scorers/base.py),
bound to an eval by slug via @register_scorer. One "eval" = one spec; one
"scorer" = the code that fills it. There is no second scoring contract."""

from __future__ import annotations

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
