"""Run manifest + executor protocol."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class PriorRef(BaseModel):
    id: str
    version: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class ModelRef(BaseModel):
    id: str
    version: str


class EvalRef(BaseModel):
    id: str
    version: str


class ComputeSpec(BaseModel):
    target: str = "local"
    gpu: str | None = None
    num_gpus: int = 1


class TrackingSpec(BaseModel):
    wandb_project: str | None = None
    wandb_run_id: str | None = None
    hf_repo: str | None = None
    mlflow_experiment: str | None = None


class RunSpec(BaseModel):
    id: str
    description: str | None = None
    prior: PriorRef
    model: ModelRef
    evals: list[EvalRef]
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    compute: ComputeSpec = Field(default_factory=ComputeSpec)
    tracking: TrackingSpec = Field(default_factory=TrackingSpec)
    results: dict[str, Any] = Field(default_factory=dict)


class ComputeAdapter(Protocol):
    """Submits a Run to a compute backend and returns the populated results."""

    name: str

    def submit(self, run: RunSpec, project_root: Any) -> dict[str, Any]: ...


class Run:
    """Wrapper that holds a RunSpec; executors mutate `spec.results` after completion."""

    def __init__(self, spec: RunSpec):
        self.spec = spec
