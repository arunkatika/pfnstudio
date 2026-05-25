"""Gaussian-mixture classification prior."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("gaussian_mixture_cls")
class GaussianMixtureClsPrior(Prior):
    def sample(
        self, *, seed, num_points=200, num_classes=2, dim=4, cluster_std=0.8, separation=3.0, **_
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        means = rng.uniform(-separation, separation, size=(num_classes, dim)).astype(np.float32)
        per_class = [num_points // num_classes] * num_classes
        per_class[-1] += num_points - sum(per_class)
        Xs, ys = [], []
        for k, n in enumerate(per_class):
            Xs.append(means[k] + rng.normal(0, cluster_std, size=(n, dim)).astype(np.float32))
            ys.append(np.full(n, k, dtype=np.float32))
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        perm = rng.permutation(num_points)
        # Binary outcome for the BCE head; multi-class would need a softmax head.
        labels = (y > 0).astype(np.float32) if num_classes == 2 else y
        return {"X": X[perm], "labels": labels[perm]}
