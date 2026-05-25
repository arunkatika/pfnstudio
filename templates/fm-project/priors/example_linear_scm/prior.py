"""Example linear SCM prior — replace with your project's real prior.

This file shows the registration pattern. The CLI's `discover_in_project` walks
priors/, evals/, models/ and imports every .py file, which triggers these
decorators and populates the runtime registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@dataclass
class LinearSCMSample:
    X: np.ndarray
    A: np.ndarray
    W: np.ndarray


@register_prior("example_linear_scm")
class ExampleLinearSCMPrior(Prior):
    def sample(
        self,
        *,
        seed: int,
        num_variables: int = 11,
        edge_density: float = 0.3,
        coefficient_range: float = 1.0,
        noise_scale: float = 0.5,
        num_samples: int = 1000,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        d = int(num_variables)

        A = (rng.uniform(size=(d, d)) < edge_density).astype(np.float32)
        A = np.triu(A, k=1)

        W = A * rng.uniform(-coefficient_range, coefficient_range, size=(d, d)).astype(np.float32)
        floor = 0.1
        small = (np.abs(W) < floor) & (A == 1)
        W[small] = floor

        noise = rng.normal(0.0, noise_scale, size=(int(num_samples), d)).astype(np.float32)
        X = np.zeros_like(noise)
        for j in range(d):
            X[:, j] = noise[:, j] + X @ W[:, j]

        return {"X": X, "A": A, "W": W}
