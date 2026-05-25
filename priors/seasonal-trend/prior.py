"""Seasonal + trend prior — random slope + periodic + noise."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("seasonal_trend")
class SeasonalTrendPrior(Prior):
    def sample(
        self,
        *,
        seed,
        num_points=120,
        period_range=(8, 30),
        trend_scale=0.05,
        season_amp_range=(0.5, 2.0),
        noise_scale=0.2,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        slope = float(rng.uniform(-trend_scale, trend_scale))
        period = float(rng.uniform(*period_range))
        amp = float(rng.uniform(*season_amp_range))
        phi = float(rng.uniform(0, 2 * np.pi))
        t = np.arange(num_points, dtype=np.float32)
        y = (
            slope * t
            + amp * np.sin(2 * np.pi * t / period + phi)
            + rng.normal(0, noise_scale, size=num_points)
        ).astype(np.float32)
        return {
            "t": t.reshape(-1, 1),
            "y": y,
            "X": y.reshape(-1, 1),
            "slope_true": slope,
            "period_true": period,
            "amp_true": amp,
        }
