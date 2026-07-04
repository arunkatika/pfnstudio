"""Bayesian linear regression prior, packed for in-context inference.

Each sample is a single regression task drawn from the prior

    y = a * x + b + ε,    a, b ~ N(0, weight_std),   ε ~ N(0, noise_scale).

The sample is *packed* as a sequence the transformer can attend across:

    sequence  = [ (x_ctx[0],  y_ctx[0],  1.0),   ...   <- context tokens
                  (x_ctx[k-1],y_ctx[k-1],1.0),
                  (x_qry[0],  0,         0.0),   ...   <- query tokens
                  (x_qry[m-1],0,         0.0) ]

`n_ctx` marks the boundary. `pfnstudio_core`'s default training step
slices the model's logits at `n_ctx` so the loss is computed only at
query positions — context tokens are present only so the transformer
can route information from them.

This is the textbook PFN setup. The model never sees `(a, b)` and learns
to do regression purely from the context (x, y) pairs in a single
forward pass.

Cited:
  Müller et al., "Transformers Can Do Bayesian Inference", ICLR 2022.
  (https://arxiv.org/abs/2112.10510)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior

# Fraction of each task's points used as context; the rest are queries.
CTX_FRAC = 0.75


@register_prior("bayesian_linear")
class BayesianLinearPrior(Prior):
    def sample(
        self,
        *,
        seed: int,
        num_points: int = 64,
        weight_std: float = 1.0,
        noise_scale: float = 0.1,
        x_range: float = 2.0,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal(0.0, weight_std))
        b = float(rng.normal(0.0, weight_std))

        x = rng.uniform(-x_range, x_range, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, noise_scale, size=num_points)).astype(np.float32)

        # Random context/query split per task.
        perm = rng.permutation(num_points)
        x_perm = x[perm]
        y_perm = y[perm]

        n_ctx = int(num_points * CTX_FRAC)
        x_ctx, x_q = x_perm[:n_ctx], x_perm[n_ctx:]
        y_ctx, y_q = y_perm[:n_ctx], y_perm[n_ctx:]

        ctx_tok = np.stack(
            [x_ctx, y_ctx, np.ones_like(x_ctx, dtype=np.float32)], axis=1
        )  # (n_ctx, 3)
        q_tok = np.stack(
            [x_q, np.zeros_like(x_q, dtype=np.float32), np.zeros_like(x_q, dtype=np.float32)],
            axis=1,
        )  # (n_q, 3)
        seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)

        return {
            "X": seq,  # (N, 3) — packed (x, y_or_0, is_ctx) tokens
            "y": y_q,  # (n_query,) — targets for query positions only
            "n_ctx": n_ctx,  # boundary the default step uses to slice logits
            "a_true": a,
            "b_true": b,
        }
