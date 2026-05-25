"""GP-RBF regression prior — y ~ GP(0, k_RBF(x, x'))."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("gp_regression")
class GpRegressionPrior(Prior):
    def sample(
        self,
        *,
        seed,
        num_points=80,
        ls_range=(0.2, 1.5),
        var_range=(0.5, 2.0),
        noise_scale=0.1,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        ls = float(rng.uniform(*ls_range))
        var = float(rng.uniform(*var_range))
        x = np.sort(rng.uniform(-3, 3, size=num_points)).astype(np.float32)
        d2 = (x[:, None] - x[None, :]) ** 2
        K = var * np.exp(-0.5 * d2 / (ls**2)) + 1e-6 * np.eye(num_points, dtype=np.float32)
        f = rng.multivariate_normal(np.zeros(num_points), K).astype(np.float32)
        y = (f + rng.normal(0, noise_scale, size=num_points)).astype(np.float32)
        return {"X": x.reshape(-1, 1), "y": y, "ls_true": ls, "var_true": var}
