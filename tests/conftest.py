"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "core"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "cli"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "studio"))
