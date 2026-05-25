"""Compute adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ComputeAdapter(ABC):
    """Submits a Run to a compute backend, returns the populated results dict."""

    name: str = "abstract"

    @abstractmethod
    def submit(self, run_yaml: Path, project_root: Path) -> dict[str, Any]:
        """Run the experiment defined by run_yaml and return results."""
