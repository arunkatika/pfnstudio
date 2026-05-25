"""Hierarchical normal prior — population μ_0, group means μ_g."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("hierarchical_normal")
class HierarchicalNormalPrior(Prior):
    def sample(
        self, *, seed, num_groups=8, per_group=12, population_std=1.0, group_std=0.5, **_
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        mu0 = float(rng.normal(0, 1))
        mu_g = rng.normal(mu0, population_std, size=num_groups).astype(np.float32)
        groups = np.repeat(np.arange(num_groups), per_group).astype(np.int64)
        y = rng.normal(mu_g[groups], group_std).astype(np.float32)
        X = np.stack(
            [
                groups.astype(np.float32),
                np.tile(np.arange(per_group), num_groups).astype(np.float32),
            ],
            axis=-1,
        )
        return {"X": X, "y": y, "mu0_true": mu0, "mu_g_true": mu_g}
