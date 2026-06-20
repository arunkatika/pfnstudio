"""MLP-SCM prior for causal sensitivity analysis (Javurek et al., 2026).

Faithful reimplementation of the synthetic SCM family from
"Amortizing Causal Sensitivity Analysis via Prior Data-Fitted Networks"
(arXiv:2605.10590). Reimplemented from the paper's description; no
upstream code is copied.

v0.1 scope: in-context CATE recovery under unobserved confounding.
v0.2 will replace the CATE target with the Lagrangian-labeled sensitivity
bound under the Marginal Sensitivity Model.

Each call to sample() draws one DGP: random MLP weights for the
propensity and outcome equations, a per-task u_strength controlling the
magnitude of unobserved confounding, and num_samples points. The points
are packed as a (context, query) sequence the transformer attends across:

    token = (x_1, ..., x_d, t, y_or_0, is_context_flag)

Context tokens carry the observed (T, Y) pair. Query tokens have T and Y
masked to 0 and is_context_flag=0. The target at each query position is
the latent CATE for that X — the model has to recover it from the
context alone, never seeing U.

Cited:
  Javurek, Frauen, Brockschmidt, Schweisthal, Feuerriegel.
  "Amortizing Causal Sensitivity Analysis via Prior Data-Fitted Networks."
  arXiv:2605.10590
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior, register_prior


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _draw_mlp(rng: np.random.Generator, in_dim: int, hidden: int, out_dim: int):
    """Draw a 2-layer MLP with tanh nonlinearity. Returns a closure f(x)."""
    W1 = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, hidden)).astype(np.float32)
    b1 = rng.normal(0.0, 0.1, size=hidden).astype(np.float32)
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, out_dim)).astype(np.float32)
    b2 = rng.normal(0.0, 0.1, size=out_dim).astype(np.float32)

    def fwd(x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ W1 + b1)
        return (h @ W2 + b2).squeeze(-1) if out_dim == 1 else h @ W2 + b2

    return fwd


@register_prior("mlp_sensitivity_scm")
class MlpSensitivitySCMPrior(Prior):
    def sample(
        self,
        *,
        seed: int,
        num_features: int = 10,
        num_samples: int = 256,
        ctx_frac: float = 0.75,
        outcome_hidden: int = 32,
        propensity_hidden: int = 16,
        u_strength: float | None = None,
        noise_scale: float = 0.2,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)

        # ── 1. Draw the SCM (one fresh MLP-SCM per task) ─────────────
        propensity_mlp = _draw_mlp(rng, num_features, propensity_hidden, 1)
        # Outcome MLP takes (X, T, U) — input dim = d + 2 (U is scalar).
        outcome_mlp = _draw_mlp(rng, num_features + 2, outcome_hidden, 1)

        # Per-task confounding magnitude. Drawn uniformly so the model
        # sees a range of Γ-regimes during training (this is what makes
        # the trained PFN generalize across sensitivity levels).
        u_scale = float(rng.uniform(0.5, 2.0)) if u_strength is None else float(u_strength)

        # ── 2. Sample points ─────────────────────────────────────────
        X = rng.normal(0.0, 1.0, size=(num_samples, num_features)).astype(np.float32)
        U = rng.normal(0.0, 1.0, size=num_samples).astype(np.float32)

        prop = _sigmoid(propensity_mlp(X))
        T = (rng.uniform(0.0, 1.0, size=num_samples) < prop).astype(np.float32)
        eps = rng.normal(0.0, noise_scale, size=num_samples).astype(np.float32)

        # Potential outcomes: compute both, observe one. U enters Y but
        # not T — the textbook unobserved-confounding setup.
        T0_col = np.zeros((num_samples, 1), dtype=np.float32)
        T1_col = np.ones((num_samples, 1), dtype=np.float32)
        U_col = (u_scale * U).reshape(-1, 1)

        y0 = outcome_mlp(np.concatenate([X, T0_col, U_col], axis=1)) + eps
        y1 = outcome_mlp(np.concatenate([X, T1_col, U_col], axis=1)) + eps
        Y = T * y1 + (1.0 - T) * y0

        # Latent CATE — E_U[Y1 - Y0 | X]. We integrate over U by Monte
        # Carlo with a moderate sample count; the prior is N(0,1), and
        # the outcome MLP is smooth, so this converges quickly.
        u_mc = rng.normal(0.0, 1.0, size=(64,)).astype(np.float32)
        u_mc_scaled = (u_scale * u_mc).reshape(-1, 1, 1)  # (mc, 1, 1)
        # Broadcast: (mc, n, d+2) — replicate X for each u draw, append t, u.
        Xb = np.broadcast_to(X[None, :, :], (64, num_samples, num_features))
        T0b = np.broadcast_to(T0_col[None, :, :], (64, num_samples, 1))
        T1b = np.broadcast_to(T1_col[None, :, :], (64, num_samples, 1))
        Ub = np.broadcast_to(u_mc_scaled, (64, num_samples, 1))
        y0_mc = outcome_mlp(np.concatenate([Xb, T0b, Ub], axis=-1))
        y1_mc = outcome_mlp(np.concatenate([Xb, T1b, Ub], axis=-1))
        cate_true = (y1_mc - y0_mc).mean(axis=0).astype(np.float32)  # (n,)

        # ── 3. Pack as (context, query) tokens ───────────────────────
        # Token width = d + 3: (x_1..x_d, t, y_or_0, is_ctx).
        perm = rng.permutation(num_samples)
        X_p = X[perm]
        T_p = T[perm]
        Y_p = Y[perm]
        cate_p = cate_true[perm]

        n_ctx = int(num_samples * ctx_frac)

        ctx_tokens = np.concatenate(
            [
                X_p[:n_ctx],
                T_p[:n_ctx, None],
                Y_p[:n_ctx, None],
                np.ones((n_ctx, 1), dtype=np.float32),
            ],
            axis=1,
        )
        n_q = num_samples - n_ctx
        q_tokens = np.concatenate(
            [
                X_p[n_ctx:],
                np.zeros((n_q, 1), dtype=np.float32),  # T masked
                np.zeros((n_q, 1), dtype=np.float32),  # Y masked
                np.zeros((n_q, 1), dtype=np.float32),  # is_ctx = 0
            ],
            axis=1,
        )

        seq = np.concatenate([ctx_tokens, q_tokens], axis=0).astype(np.float32)
        y_target = cate_p[n_ctx:].astype(np.float32)

        return {
            "X": seq,  # (N, d+3)
            "y": y_target,  # (n_query,) — CATE at queries
            "n_ctx": n_ctx,
            "cate_true": cate_p.astype(np.float32),  # full task CATE for eval
            "u_strength": np.float32(u_scale),
        }
