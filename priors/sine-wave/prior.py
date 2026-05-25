"""Sine-wave forecasting prior."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("sine_wave")
class SineWavePrior(Prior):
    def sample(
        self,
        *,
        seed,
        num_points=100,
        amplitude_max=2.0,
        omega_range=(0.5, 2.0),
        noise_scale=0.1,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.uniform(0.5, amplitude_max))
        omega = float(rng.uniform(omega_range[0], omega_range[1]))
        phi = float(rng.uniform(0, 2 * np.pi))
        t = np.linspace(0, 10, num_points).astype(np.float32)
        y = (a * np.sin(omega * t + phi) + rng.normal(0, noise_scale, size=num_points)).astype(
            np.float32
        )
        return {
            "t": t.reshape(-1, 1),
            "y": y,
            "X": y.reshape(-1, 1),
            "a_true": a,
            "omega_true": omega,
            "phi_true": phi,
        }
