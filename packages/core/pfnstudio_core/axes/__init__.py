"""Axes package — public surface for promptable-prior axes.

Importing this module also imports `builtins`, which eagerly
registers the built-in axes (monotonicity, …) so they're available
without an extra import in user code."""

from . import builtins as _builtins  # noqa: F401  (side-effect: registers built-ins)
from .base import (
    UNKNOWN,
    Axis,
    AxisKind,
    get_axis,
    is_unknown,
    list_axes,
    register_axis,
)
from .detector import AxisDetector, detect, train_detector
from .encoding import encode_tag, sample_tag, tag_dim

__all__ = [
    "UNKNOWN",
    "Axis",
    "AxisDetector",
    "AxisKind",
    "detect",
    "encode_tag",
    "get_axis",
    "is_unknown",
    "list_axes",
    "register_axis",
    "sample_tag",
    "tag_dim",
    "train_detector",
]
