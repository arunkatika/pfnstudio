"""Axis abstraction for promptable priors.

An Axis declares a *steerable property* of a prior — a knob the trained
brain learns to honor at inference time. Examples: monotonicity (edge
signs in an SCM), lag_scale (delay distribution), feedback_allowed
(DAG vs cyclic), sparsity (mean parents per node).

The axis declares the contract (name, kind, value space, default).
Each Prior that opts into an axis implements its own application
logic — the same axis can mean different things to different priors
(e.g. monotonicity on an SCM constrains edge signs; on a regression
prior it constrains weight signs). The base class doesn't try to
unify those interpretations.

Back-compat invariants this module enforces:
- The reserved sentinel ``UNKNOWN`` means "skip this axis at sample
  time". A prior given ``tag={axis: UNKNOWN}`` for every axis is
  bit-identical to its pre-axis behavior. This is what guarantees
  adding axes can't regress existing benchmarks.
- A prior with no declared axes ignores any ``tag=...`` argument
  passed to ``sample()`` — old call sites keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AxisKind = Literal["categorical", "range", "boolean"]

# Reserved sentinel — when a tag has this for an axis, the prior must
# skip the axis hook and behave as if the axis weren't declared.
# Stored as a string so it round-trips through JSON / YAML cleanly.
UNKNOWN = "__unknown__"


@dataclass(frozen=True)
class Axis:
    name: str
    kind: AxisKind
    # For categorical: list of label strings. For range: [min, max].
    # For boolean: ignored.
    values: tuple[Any, ...] = field(default_factory=tuple)
    description: str = ""
    # Fraction of training samples that get UNKNOWN for this axis.
    # Default 0.3 — preserves substantial unconditional mass so the
    # brain doesn't lose its pre-axis baseline.
    unknown_mass: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.unknown_mass <= 1.0:
            raise ValueError(
                f"axis {self.name!r}: unknown_mass must be in [0, 1], got {self.unknown_mass}"
            )
        if self.kind == "categorical" and not self.values:
            raise ValueError(f"axis {self.name!r}: categorical axes need at least one value")
        if self.kind == "range" and len(self.values) != 2:
            raise ValueError(
                f"axis {self.name!r}: range axes need exactly 2 values (min, max), got {len(self.values)}"
            )

    def sample_value(self, rng: Any) -> Any:
        """Sample one value for this axis, honoring the unknown_mass invariant."""
        if rng.random() < self.unknown_mass:
            return UNKNOWN
        if self.kind == "categorical":
            return rng.choice(list(self.values))
        if self.kind == "boolean":
            return bool(rng.integers(0, 2))
        if self.kind == "range":
            lo, hi = float(self.values[0]), float(self.values[1])
            return float(rng.uniform(lo, hi))
        raise ValueError(f"unknown axis kind: {self.kind!r}")


# Module-level registry. Resolves axis names → Axis instances at
# loader time. Built-in axes register themselves on import; custom
# axes register via @register_axis.
_AXES: dict[str, Axis] = {}


def register_axis(axis: Axis) -> Axis:
    """Register an Axis. Idempotent: re-registering the *same* axis is
    a no-op (useful when modules get imported twice); re-registering
    a *different* axis under the same name raises."""
    existing = _AXES.get(axis.name)
    if existing is not None and existing != axis:
        raise ValueError(f"axis {axis.name!r} already registered with a different definition")
    _AXES[axis.name] = axis
    return axis


def get_axis(name: str) -> Axis:
    """Look up a registered axis by name. Raises KeyError if not registered."""
    if name not in _AXES:
        raise KeyError(f"axis {name!r} not registered. Known axes: {sorted(_AXES)}")
    return _AXES[name]


def list_axes() -> list[str]:
    return sorted(_AXES)


def is_unknown(value: Any) -> bool:
    """Whether a tag value is the UNKNOWN sentinel. Use this in axis
    application code rather than comparing to the string literal."""
    return value == UNKNOWN
