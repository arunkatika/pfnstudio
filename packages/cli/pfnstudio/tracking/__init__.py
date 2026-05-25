"""Tracking adapters. Each writes run metadata + metrics to an external backend."""

from .base import Tracker
from .local import LocalTracker
from .mlflow import MLflowTracker
from .wandb import WandbTracker

TRACKERS: dict[str, type[Tracker]] = {
    "local": LocalTracker,
    "wandb": WandbTracker,
    "mlflow": MLflowTracker,
}


def select_trackers(run_tracking) -> list[Tracker]:
    """Choose trackers based on what fields are populated on RunSpec.tracking."""
    chosen: list[Tracker] = [LocalTracker()]
    if getattr(run_tracking, "wandb_project", None):
        chosen.append(WandbTracker())
    if getattr(run_tracking, "mlflow_experiment", None):
        chosen.append(MLflowTracker())
    return chosen


__all__ = ["TRACKERS", "Tracker", "select_trackers"]
