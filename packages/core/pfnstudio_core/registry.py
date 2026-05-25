"""Decorator-based registries for priors, architecture blocks, and evals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_PRIORS: dict[str, type] = {}
_BLOCKS: dict[str, type] = {}
_EVALS: dict[str, type] = {}


def _register(table: dict[str, type], name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in table and table[name] is not cls:
            raise ValueError(f"Already registered under '{name}': {table[name]!r}")
        table[name] = cls
        return cls

    return decorator


def register_prior(name: str) -> Callable[[type[T]], type[T]]:
    """Register a Prior subclass so YAML specs can reference it by `id`."""
    return _register(_PRIORS, name)


def register_block(name: str) -> Callable[[type[T]], type[T]]:
    """Register an architecture block so model YAML can reference it by `type`."""
    return _register(_BLOCKS, name)


def register_eval(name: str) -> Callable[[type[T]], type[T]]:
    """Register an Eval subclass so YAML specs can reference it by `id`."""
    return _register(_EVALS, name)


def get_prior(name: str) -> type:
    # Loose match: tolerate `-` ↔ `_` swaps. The DB slugifier converts `_` to
    # `-` (URL-safe), but Python decorators register with idiomatic snake_case.
    # Without this, a prior registered as "linear_regression" can't be found
    # under its YAML id "linear-regression".
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        if candidate in _PRIORS:
            return _PRIORS[candidate]
    raise KeyError(f"No prior registered under '{name}'. Available: {sorted(_PRIORS)}")


def get_block(name: str) -> type:
    if name not in _BLOCKS:
        raise KeyError(f"No block registered under '{name}'. Available: {sorted(_BLOCKS)}")
    return _BLOCKS[name]


def get_eval(name: str) -> type:
    if name not in _EVALS:
        raise KeyError(f"No eval registered under '{name}'. Available: {sorted(_EVALS)}")
    return _EVALS[name]


def list_priors() -> list[str]:
    return sorted(_PRIORS)


def list_blocks() -> list[str]:
    return sorted(_BLOCKS)


def list_evals() -> list[str]:
    return sorted(_EVALS)


def _clear_for_tests() -> None:
    _PRIORS.clear()
    _BLOCKS.clear()
    _EVALS.clear()


def discover_in_project(project_root: Any) -> None:
    """Import every Python module under priors/ and evals/ so decorators register.

    project_root: pathlib.Path — typed loosely to avoid an import cycle.
    """
    import importlib.util
    from pathlib import Path

    root = Path(project_root)
    for sub in ("priors", "evals", "models"):
        d = root / sub
        if not d.exists():
            continue
        for py in d.rglob("*.py"):
            spec = importlib.util.spec_from_file_location(
                f"_ps_dyn_{py.stem}_{abs(hash(str(py)))}", py
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
