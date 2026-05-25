"""AR(2) forecasting prior with stationarity rejection sampling."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("ar2_process")
class AR2Prior(Prior):
    def sample(self, *, seed, num_points=100, noise_scale=0.3, **_) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        while True:
            phi1 = rng.uniform(-1.5, 1.5)
            phi2 = rng.uniform(-1.0, 1.0)
            if abs(phi2) < 1 and phi1 + phi2 < 1 and phi2 - phi1 < 1:
                break
        y = np.zeros(num_points, dtype=np.float32)
        eps = rng.normal(0, noise_scale, size=num_points).astype(np.float32)
        for t in range(2, num_points):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
        t = np.arange(num_points, dtype=np.float32).reshape(-1, 1)
        return {
            "t": t,
            "y": y,
            "X": y.reshape(-1, 1),
            "phi1_true": float(phi1),
            "phi2_true": float(phi2),
        }
