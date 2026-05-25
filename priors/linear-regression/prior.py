"""Linear regression prior — y = a*x + b + Gaussian noise."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("linear_regression")
class LinearRegressionPrior(Prior):
    def sample(
        self,
        *,
        seed,
        num_points=100,
        coefficient_range=2.0,
        intercept_range=1.0,
        noise_scale=0.1,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = rng.uniform(-coefficient_range, coefficient_range)
        b = rng.uniform(-intercept_range, intercept_range)
        x = rng.uniform(-2, 2, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0, noise_scale, size=num_points)).astype(np.float32)
        return {"X": x.reshape(-1, 1), "y": y, "a_true": float(a), "b_true": float(b)}
