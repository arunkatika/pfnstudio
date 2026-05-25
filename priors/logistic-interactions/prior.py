"""Logistic with random pairwise interactions."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("logistic_interactions")
class LogisticInteractionsPrior(Prior):
    def sample(
        self, *, seed, num_points=200, dim=6, interaction_density=0.2, w_scale=1.0, **_
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        w = rng.normal(0, w_scale, size=dim).astype(np.float32)
        W = rng.normal(0, w_scale, size=(dim, dim)).astype(np.float32)
        mask = (rng.uniform(size=(dim, dim)) < interaction_density).astype(np.float32)
        W = np.triu(W * mask, k=1)
        X = rng.normal(0, 1, size=(num_points, dim)).astype(np.float32)
        logits = X @ w + (X @ W * X).sum(-1)
        p = 1.0 / (1.0 + np.exp(-logits))
        labels = (rng.uniform(size=num_points) < p).astype(np.float32)
        return {"X": X, "labels": labels}
