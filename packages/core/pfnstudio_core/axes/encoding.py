"""Tag encoding — fixed-length vector representation of an axis tag.

The encoding rules:

- **Categorical axes** → one-hot over the declared values, plus one
  extra slot for the UNKNOWN sentinel (always last). So a categorical
  axis with K values contributes K+1 dimensions.
- **Boolean axes** → 2 dims (false, true) plus 1 for UNKNOWN. Same
  shape as a 2-value categorical.
- **Range axes** → 2 dims: ``[is_unknown_flag, normalised_value]``.
  When tagged with a real value v in [min, max], normalised_value is
  ``(v - min) / (max - min)`` clipped to [0, 1] and is_unknown_flag
  is 0. When UNKNOWN, normalised_value is 0 and is_unknown_flag is 1.

The encoding is deterministic and stateless — the same axes list and
the same tag always produce the same vector. ``tag_dim(axes)`` returns
the total length so model constructors can size their tag embedder
once at init time without poking at sample tags.

Back-compat invariant: ``encode_tag({}, axes)`` and ``encode_tag(None,
axes)`` are byte-identical to ``encode_tag({a: UNKNOWN for a in axes},
axes)``. Used by callers who need a "no constraints" tag vector
without enumerating all axes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import UNKNOWN, Axis, is_unknown


def _axis_dim(axis: Axis) -> int:
    if axis.kind == "categorical":
        return len(axis.values) + 1  # values + UNKNOWN slot
    if axis.kind == "boolean":
        return 3  # false, true, UNKNOWN
    if axis.kind == "range":
        return 2  # is_unknown_flag, normalised_value
    raise ValueError(f"unknown axis kind: {axis.kind!r}")


def tag_dim(axes: list[Axis]) -> int:
    """Total length of the tag vector produced by ``encode_tag`` for
    the given axes (in the same order)."""
    return sum(_axis_dim(a) for a in axes)


def _encode_categorical(axis: Axis, value: Any) -> np.ndarray:
    """One-hot over values + final UNKNOWN slot."""
    dim = _axis_dim(axis)
    out = np.zeros(dim, dtype=np.float32)
    if value is None or is_unknown(value):
        out[-1] = 1.0
        return out
    if value not in axis.values:
        raise ValueError(f"axis {axis.name!r}: value {value!r} not in {axis.values}")
    idx = axis.values.index(value)
    out[idx] = 1.0
    return out


def _encode_boolean(axis: Axis, value: Any) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    if value is None or is_unknown(value):
        out[2] = 1.0
        return out
    out[int(bool(value))] = 1.0
    return out


def _encode_range(axis: Axis, value: Any) -> np.ndarray:
    out = np.zeros(2, dtype=np.float32)
    if value is None or is_unknown(value):
        out[0] = 1.0  # is_unknown_flag
        return out
    lo, hi = float(axis.values[0]), float(axis.values[1])
    span = hi - lo
    if span == 0:
        out[1] = 0.0
    else:
        normalised = (float(value) - lo) / span
        out[1] = float(np.clip(normalised, 0.0, 1.0))
    return out


def encode_tag(tag: dict[str, Any] | None, axes: list[Axis]) -> np.ndarray:
    """Encode a tag dict into a fixed-length float32 vector for the given axes.

    Missing axis entries are treated as UNKNOWN — same as explicitly
    passing UNKNOWN. The encoded vector's length matches ``tag_dim(axes)``.
    """
    tag = tag or {}
    chunks: list[np.ndarray] = []
    for axis in axes:
        value = tag.get(axis.name, UNKNOWN)
        if axis.kind == "categorical":
            chunks.append(_encode_categorical(axis, value))
        elif axis.kind == "boolean":
            chunks.append(_encode_boolean(axis, value))
        elif axis.kind == "range":
            chunks.append(_encode_range(axis, value))
        else:
            raise ValueError(f"unknown axis kind: {axis.kind!r}")
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def sample_tag(axes: list[Axis], rng: np.random.Generator) -> dict[str, Any]:
    """Sample one tag dict for all axes, honoring each axis's
    ``unknown_mass``. Returns axis_name → sampled value (or UNKNOWN).

    Used by the training loop to draw a fresh tag per batch."""
    return {axis.name: axis.sample_value(rng) for axis in axes}
