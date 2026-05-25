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
from .eval import Eval, EvalSpec
from .model import Model, ModelSpec
from .prior import Prior, PriorSpec
from .registry import (
    get_block,
    get_eval,
    get_prior,
    list_blocks,
    list_evals,
    list_priors,
    register_block,
    register_eval,
    register_prior,
)
from .run import Run, RunSpec

__version__ = "0.4.0"

__all__ = [
    "UNKNOWN",
    "Axis",
    "AxisDetector",
    "DatasetUnavailable",
    "Eval",
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
    "get_eval",
    "get_prior",
    "is_unknown",
    "list_axes",
    "list_blocks",
    "list_evals",
    "list_priors",
    "register_axis",
    "register_block",
    "register_eval",
    "register_prior",
    "sample_tag",
    "tag_dim",
    "train_detector",
]
