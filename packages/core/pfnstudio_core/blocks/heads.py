"""Output heads. Each head takes a pooled representation and emits task-specific output."""

from __future__ import annotations

from typing import Any

from ..registry import register_block


@register_block("discovery_head")
class DiscoveryHead:
    """Predicts an adjacency matrix over D variables. Output: (B, D, D) logits."""

    def __init__(self, d_model: int = 256, num_variables: int = 11):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("discovery_head requires torch.") from e

        self.num_variables = num_variables
        self.proj = nn.Linear(d_model, num_variables * num_variables)

    def __call__(self, x: Any) -> Any:
        return self.proj(x).reshape(-1, self.num_variables, self.num_variables)


@register_block("estimation_head")
class EstimationHead:
    """Predicts a scalar treatment effect per (treatment, outcome) pair."""

    def __init__(self, d_model: int = 256, num_pairs: int = 1):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("estimation_head requires torch.") from e

        self.proj = nn.Linear(d_model, num_pairs)

    def __call__(self, x: Any) -> Any:
        return self.proj(x)


@register_block("scalar_head")
class ScalarHead:
    """Generic scalar regression head."""

    def __init__(self, d_model: int = 256, d_out: int = 1):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("scalar_head requires torch.") from e

        self.proj = nn.Linear(d_model, d_out)

    def __call__(self, x: Any) -> Any:
        return self.proj(x)
