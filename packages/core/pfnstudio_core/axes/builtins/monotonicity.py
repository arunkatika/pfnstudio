"""Built-in `monotonicity` axis.

Declares the contract; per-prior application lives in each Prior
subclass (an SCM's monotonicity constrains edge signs, a regression
prior's monotonicity constrains weight signs, etc.).
"""

from __future__ import annotations

from ..base import Axis, register_axis

monotonicity = register_axis(
    Axis(
        name="monotonicity",
        kind="categorical",
        values=("positive", "negative", "mixed"),
        description=(
            "Sign discipline of the prior's causal relationships. "
            "positive = X up implies Y up everywhere; negative = "
            "X up implies Y down; mixed = signs may vary. Unknown "
            "(default) leaves the prior unconstrained."
        ),
        unknown_mass=0.3,
    )
)
