"""Example custom architecture block — lives in the PROJECT, not core.

This is the point of the template: a researcher can define their own block as
project code and use it in a model, exactly like priors/evals live in the
project. It is discovered at run time (discover_in_project imports blocks/*.py)
and referenced from a model by its registered `type`.

A gated residual: x + sigmoid(gate(x)) * mlp(x). A drop-in refinement you can
insert between transformer layers to add a learned, per-feature gated update
without changing the sequence shape.

Anatomy of a block (copy this shape for your own):
  - A plain class decorated with @register_block("<type>"); "<type>" is what a
    model's blocks[].type references. ABSOLUTE imports only — this file is
    loaded via exec_module with no package context.
  - __init__(self, <config kwargs>, **_): build torch submodules as attributes
    (they're auto-collected for training). Accept **_ so unknown config is
    ignored. Import torch INSIDE __init__, never at module top, so discovery
    never fails when torch is absent.
  - __call__(self, x): a (B, N, d_model) tensor -> (B, N, d_model) tensor.
"""

from __future__ import annotations

from typing import Any

from pfnstudio_core.registry import register_block


@register_block("gated_residual")
class GatedResidual:
    """x + sigmoid(gate(x)) * mlp(x) — shape-preserving gated residual update."""

    def __init__(self, d_model: int = 256, hidden_mult: int = 2, **_: Any) -> None:
        import torch.nn as nn

        self.d_model = d_model
        hidden = d_model * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def __call__(self, x: Any) -> Any:
        return x + self.gate(x) * self.mlp(x)
