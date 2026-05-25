"""Two-moons classification prior — random interlocking half-moons."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("two_moons")
class TwoMoonsPrior(Prior):
    def sample(self, *, seed, num_points=100, noise_scale=0.15, radius=1.0, **_) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        n0 = num_points // 2
        n1 = num_points - n0
        t0 = rng.uniform(0, np.pi, size=n0)
        t1 = rng.uniform(0, np.pi, size=n1)
        x0 = np.stack([radius * np.cos(t0), radius * np.sin(t0)], axis=-1)
        x1 = np.stack([radius - radius * np.cos(t1), -radius * np.sin(t1) + 0.5], axis=-1)
        theta = rng.uniform(0, 2 * np.pi)
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
        )
        X = np.concatenate([x0, x1], axis=0).astype(np.float32) @ R.T
        X = X + rng.normal(0, noise_scale, size=X.shape).astype(np.float32)
        labels = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)
        perm = rng.permutation(num_points)
        return {"X": X[perm], "labels": labels[perm]}
