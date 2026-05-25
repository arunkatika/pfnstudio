"""Tabular embedder — projects (B, N, D_in) into (B, N, d_model)."""

from __future__ import annotations

from typing import Any

from ..registry import register_block


@register_block("tabular_embedder")
class TabularEmbedder:
    def __init__(self, d_model: int = 256, d_in: int | None = None):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("tabular_embedder requires torch.") from e

        self.d_model = d_model
        self.d_in = d_in
        self._linear = nn.LazyLinear(d_model) if d_in is None else nn.Linear(d_in, d_model)

    def __call__(self, x: Any) -> Any:
        return self._linear(x)
