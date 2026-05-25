"""Local tracker — writes runs/<run_id>/results.json. Always enabled."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Tracker


class LocalTracker(Tracker):
    name = "local"

    def __init__(self) -> None:
        self._run_dir: Path | None = None
        self._metrics: list[dict[str, Any]] = []

    def start(self, run_id: str, config: dict[str, Any], project_root: Path) -> None:
        self._run_dir = project_root / "runs" / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        record = {"step": step, **metrics}
        self._metrics.append(record)
        if self._run_dir:
            (self._run_dir / "metrics.jsonl").open("a").write(json.dumps(record) + "\n")

    def finish(self, results: dict[str, Any]) -> None:
        if self._run_dir:
            (self._run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
