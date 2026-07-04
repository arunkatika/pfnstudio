"""PFN Studio core abstractions."""

from . import axes as _axes  # noqa: F401  # registers built-in axes on import
from . import blocks as _blocks  # noqa: F401  # registers built-in blocks on import
from .axes import (
    UNKNOWN,
    Axis,
    AxisDetector,
    detect,
    encode_tag,
    get_axis,
    is_unknown,
    list_axes,
    register_axis,
    sample_tag,
    tag_dim,
    train_detector,
)
from .datasets import DatasetUnavailable, RegistryDatasetLoader
from .eval import EvalSpec
from .model import Model, ModelSpec
from .prior import Prior, PriorSpec
from .registry import (
    get_block,
    get_prior,
    get_scorer,
    list_blocks,
    list_priors,
    list_scorers,
    register_block,
    register_prior,
    register_scorer,
)
from .run import Run, RunSpec

__version__ = "0.8.0"

__all__ = [
    "UNKNOWN",
    "Axis",
    "AxisDetector",
    "DatasetUnavailable",
    "EvalSpec",
    "Model",
    "ModelSpec",
    "Prior",
    "PriorSpec",
    "RegistryDatasetLoader",
    "Run",
    "RunSpec",
    "detect",
    "encode_tag",
    "get_axis",
    "get_block",
    "get_prior",
    "get_scorer",
    "is_unknown",
    "list_axes",
    "list_blocks",
    "list_priors",
    "list_scorers",
    "register_axis",
    "register_block",
    "register_prior",
    "register_scorer",
    "sample_tag",
    "tag_dim",
    "train_detector",
]
