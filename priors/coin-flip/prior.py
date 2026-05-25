"""Beta-Bernoulli coin-flip prior."""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


@register_prior("coin_flip")
class CoinFlipPrior(Prior):
    def sample(self, *, seed, num_flips=50, alpha=2.0, beta=2.0, **_) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        p = float(rng.beta(alpha, beta))
        flips = (rng.uniform(size=num_flips) < p).astype(np.float32)
        return {"flips": flips.reshape(-1, 1), "p_true": p}
