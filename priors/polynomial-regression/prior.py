"""Polynomial regression prior — random degree-D polynomial + noise."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("polynomial_regression")
class PolynomialRegressionPrior(Prior):
    def sample(
        self, *, seed, num_points=100, max_degree=3, coef_scale=1.0, noise_scale=0.2, **_
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        d = int(rng.integers(1, max_degree + 1))
        coefs = rng.normal(0, coef_scale, size=d + 1).astype(np.float32)
        x = rng.uniform(-1.5, 1.5, size=num_points).astype(np.float32)
        y = np.polyval(coefs[::-1], x) + rng.normal(0, noise_scale, size=num_points).astype(
            np.float32
        )
        return {"X": x.reshape(-1, 1), "y": y.astype(np.float32), "degree_true": d}
