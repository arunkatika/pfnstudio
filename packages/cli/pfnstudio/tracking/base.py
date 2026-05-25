"""Tracker protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    name: str = "abstract"

    @abstractmethod
    def start(self, run_id: str, config: dict[str, Any], project_root: Path) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    @abstractmethod
    def finish(self, results: dict[str, Any]) -> None: ...
