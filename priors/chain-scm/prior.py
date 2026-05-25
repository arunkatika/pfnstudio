"""Causal-chain prior — random permutations wired as chains.

Promptable via the ``monotonicity`` axis: when tagged, every edge in
the sampled chain gets a constrained sign. Untagged (or tagged
``UNKNOWN``) the prior samples signs uniformly from {-1, +1} — the
pre-axis behavior, preserved bit-identically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, is_unknown, register_prior


@register_prior("chain_scm")
class ChainScmPrior(Prior):
    # Declares which axes this prior knows how to honor. Listed names
    # must be registered in the axis registry; chain-scm only honors
    # monotonicity in v1.
    axes = ["monotonicity"]

    def sample(
        self,
        *,
        seed,
        num_points=200,
        d=5,
        weight_range=(0.5, 2.0),
        noise_scale=0.3,
        tag: dict[str, Any] | None = None,
        **_,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(d)

        # Resolve monotonicity from the tag. UNKNOWN / absent / unknown
        # axis values fall through to the pre-axis sampling path, which
        # is exactly what the back-compat invariant requires.
        sign_choice = self._resolve_signs(tag, rng)

        A = np.zeros((d, d), dtype=np.float32)
        for i in range(d - 1):
            u, v = order[i], order[i + 1]
            A[u, v] = rng.uniform(*weight_range) * sign_choice()
        X = np.zeros((num_points, d), dtype=np.float32)
        for k in order:
            parents = np.where(A[:, k] != 0)[0]
            if len(parents):
                X[:, k] = X[:, parents] @ A[parents, k]
            X[:, k] += rng.normal(0, noise_scale, size=num_points).astype(np.float32)
        adj = (A != 0).astype(np.float32)
        return {"X": X, "A": adj}

    def _resolve_signs(self, tag: dict[str, Any] | None, rng: np.random.Generator):
        """Return a zero-arg callable yielding the sign for each new edge.

        Pre-axis behavior: ``rng.choice([-1.0, 1.0])`` on every call.
        With ``monotonicity=positive`` every call returns ``+1.0``;
        with ``negative`` every call returns ``-1.0``; ``mixed``
        explicitly random (semantically distinct from UNKNOWN but
        sample-identical to the pre-axis baseline).
        """
        mono = (tag or {}).get("monotonicity") if tag else None
        if mono is None or is_unknown(mono) or mono == "mixed":
            return lambda: float(rng.choice([-1.0, 1.0]))
        if mono == "positive":
            return lambda: 1.0
        if mono == "negative":
            return lambda: -1.0
        raise ValueError(
            f"chain_scm: unsupported monotonicity value {mono!r}. "
            "Expected one of: positive, negative, mixed, UNKNOWN."
        )
