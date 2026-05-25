"""Linear-SCM discovery prior — recover an ER-DAG from observations."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("linear_scm")
class LinearScmPrior(Prior):
    def sample(
        self,
        *,
        seed,
        num_points=200,
        d=5,
        edge_prob=0.4,
        weight_range=(0.5, 2.0),
        noise_scale=0.3,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(d)
        A = np.zeros((d, d), dtype=np.float32)
        for i in range(d):
            for j in range(i + 1, d):
                u, v = order[i], order[j]
                if rng.uniform() < edge_prob:
                    w = rng.uniform(*weight_range) * rng.choice([-1.0, 1.0])
                    A[u, v] = w
        # Forward simulate in topological order.
        X = np.zeros((num_points, d), dtype=np.float32)
        for k in order:
            parents = np.where(A[:, k] != 0)[0]
            if len(parents):
                X[:, k] = X[:, parents] @ A[parents, k]
            X[:, k] += rng.normal(0, noise_scale, size=num_points).astype(np.float32)
        adj = (A != 0).astype(np.float32)
        return {"X": X, "A": adj}
