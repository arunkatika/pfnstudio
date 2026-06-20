"""Do-PFN SCM prior — in-context interventional outcome prediction.

Faithful reimplementation of the synthetic SCM family from
"Do-PFN: In-Context Learning for Causal Effect Estimation"
(Robertson et al., NeurIPS 2025; arXiv:2506.06039). Reimplemented from
the paper's description; no upstream code is copied (the upstream repo
at github.com/jr2021/Do-PFN has no license declared).

Each call to sample() draws one DGP:

  1. Sample an SCM:
     - Random covariance Σ for the d-dim covariate X (block correlation).
     - Random 2-layer MLP for the propensity π(X, U) → sigmoid → Bernoulli T.
     - Random 2-layer MLP for the outcome Y(X, T, U) + ε.
     - Per-task u_strength controlling the magnitude of unobserved confounding.
  2. Sample num_samples points from the SCM (observational distribution).
  3. Pack as a (context, query) sequence:
     - Context tokens carry (X, T_obs, Y_obs).
     - Query tokens carry (X, t_intervention, y_masked=0).
  4. The training target at each query is the INTERVENTIONAL outcome
     Y_{do(T = t_intervention)} drawn cleanly from the SCM. The model's
     job is to predict that from the observational context alone — the
     core Do-PFN claim.

Why this is different from `mlp_sensitivity_scm`
-------------------------------------------------

The causal-sensitivity-pfn prior trains the model to predict CATE
(treatment effect = Y1 - Y0). This Do-PFN prior trains the model to
predict the per-query INTERVENTIONAL outcome (Y under do(T=t) for the
specific t requested by the query). CATE is derivable at inference time
by querying both do(T=0) and do(T=1) and subtracting; CID is the
fundamental output.

Cited:
  Robertson, J., Reuter, A., Guo, S., Hollmann, N., Hutter, F.,
  Schölkopf, B. (2025). Do-PFN: In-Context Learning for Causal Effect
  Estimation. NeurIPS 2025. arXiv:2506.06039
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


def _draw_block_covariance(
    rng: np.random.Generator,
    d: int,
    correlation: float,
) -> np.ndarray:
    """Draw a block-diagonal correlation matrix for X.

    Random partition of the d features into blocks; within a block,
    pairwise correlation = ``correlation``. Cheap, valid PSD, and
    matches the paper's stated distribution over covariate
    correlation regimes without committing to one specific structure.
    """
    if correlation <= 1e-6:
        return np.eye(d, dtype=np.float32)
    # Random block partition: at least 1 block, at most d.
    n_blocks = int(rng.integers(1, max(2, d // 2 + 1)))
    sizes = rng.multinomial(d, np.ones(n_blocks) / n_blocks)
    cov = np.eye(d, dtype=np.float32)
    cursor = 0
    for sz in sizes:
        if sz <= 1:
            cursor += int(sz)
            continue
        block = np.full((sz, sz), correlation, dtype=np.float32)
        np.fill_diagonal(block, 1.0)
        cov[cursor : cursor + sz, cursor : cursor + sz] = block
        cursor += int(sz)
    return cov


@register_prior("do_pfn_scm")
class DoPfnSCMPrior(Prior):
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
        cov_correlation: float | None = None,
        noise_scale: float = 0.2,
        intervention_t: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)

        # ── 1. Draw the SCM ────────────────────────────────────────
        u_scale = float(rng.uniform(0.5, 2.0)) if u_strength is None else float(u_strength)
        rho = float(rng.uniform(0.0, 0.5)) if cov_correlation is None else float(cov_correlation)

        cov = _draw_block_covariance(rng, num_features, rho)
        # Cholesky for fast correlated samples; falls back to eye on
        # numerical issues (shouldn't happen with valid block matrix).
        try:
            L = np.linalg.cholesky(cov).astype(np.float32)
        except np.linalg.LinAlgError:
            L = np.eye(num_features, dtype=np.float32)

        # Propensity depends on (X, U) — confounded design. Outcome
        # depends on (X, T, U). Both MLPs are 2-layer with tanh.
        propensity_mlp = _draw_mlp(rng, num_features + 1, propensity_hidden, 1)
        outcome_mlp = _draw_mlp(rng, num_features + 2, outcome_hidden, 1)

        # ── 2. Sample points from the observational distribution ──
        Z = rng.normal(0.0, 1.0, size=(num_samples, num_features)).astype(np.float32)
        X = Z @ L.T  # X ~ N(0, Σ)
        U = rng.normal(0.0, 1.0, size=num_samples).astype(np.float32)
        U_col = (u_scale * U).reshape(-1, 1)

        prop = _sigmoid(propensity_mlp(np.concatenate([X, U_col], axis=1)))
        T_obs = (rng.uniform(0.0, 1.0, size=num_samples) < prop).astype(np.float32)
        eps = rng.normal(0.0, noise_scale, size=num_samples).astype(np.float32)

        T0_col = np.zeros((num_samples, 1), dtype=np.float32)
        T1_col = np.ones((num_samples, 1), dtype=np.float32)

        y0 = outcome_mlp(np.concatenate([X, T0_col, U_col], axis=1)) + eps
        y1 = outcome_mlp(np.concatenate([X, T1_col, U_col], axis=1)) + eps
        Y_obs = T_obs * y1 + (1.0 - T_obs) * y0

        # ── 3. Pick the per-query intervention t ───────────────────
        # Per-query coin flip is the paper's training convention — the
        # model learns to predict the interventional distribution under
        # whichever t the query asks for. A fixed override is supported
        # for paired-query evaluation (e.g. evaluate both do(0) and do(1)
        # on the same X for CATE).
        if intervention_t is None:
            t_query = (rng.uniform(0.0, 1.0, size=num_samples) < 0.5).astype(np.float32)
        else:
            t_query = np.full(num_samples, float(intervention_t), dtype=np.float32)

        # Training target: interventional outcome under the chosen t.
        # Note that this uses the SAME outcome MLP and SAME U as the
        # observational draw — the only thing that changes is the
        # treatment. That's the "do" operation in the SCM.
        T_int_col = t_query.reshape(-1, 1)
        y_int = outcome_mlp(np.concatenate([X, T_int_col, U_col], axis=1)) + eps

        # Monte-Carlo CATE for the auxiliary cate_true field (useful
        # for eval, NOT the training target).
        u_mc = rng.normal(0.0, 1.0, size=(64,)).astype(np.float32)
        u_mc_scaled = (u_scale * u_mc).reshape(-1, 1, 1)
        Xb = np.broadcast_to(X[None, :, :], (64, num_samples, num_features))
        T0b = np.broadcast_to(T0_col[None, :, :], (64, num_samples, 1))
        T1b = np.broadcast_to(T1_col[None, :, :], (64, num_samples, 1))
        Ub = np.broadcast_to(u_mc_scaled, (64, num_samples, 1))
        y0_mc = outcome_mlp(np.concatenate([Xb, T0b, Ub], axis=-1))
        y1_mc = outcome_mlp(np.concatenate([Xb, T1b, Ub], axis=-1))
        cate_true = (y1_mc - y0_mc).mean(axis=0).astype(np.float32)

        # ── 4. Pack as (context, query) tokens ─────────────────────
        # Token width = d + 3:
        #   ctx: (x_1..x_d, t_obs, y_obs, is_ctx=1)
        #   qry: (x_1..x_d, t_intervention, y_or_zero=0, is_ctx=0)
        # Random ctx/qry shuffle per task so the model can't memorise
        # positional cues.
        perm = rng.permutation(num_samples)
        X_p = X[perm]
        T_obs_p = T_obs[perm]
        Y_obs_p = Y_obs[perm]
        t_query_p = t_query[perm]
        y_int_p = y_int[perm]
        cate_p = cate_true[perm]

        n_ctx = int(num_samples * ctx_frac)

        ctx_tokens = np.concatenate(
            [
                X_p[:n_ctx],
                T_obs_p[:n_ctx, None],
                Y_obs_p[:n_ctx, None],
                np.ones((n_ctx, 1), dtype=np.float32),
            ],
            axis=1,
        )
        n_q = num_samples - n_ctx
        q_tokens = np.concatenate(
            [
                X_p[n_ctx:],
                t_query_p[n_ctx:, None],  # intervention t the query asks about
                np.zeros((n_q, 1), dtype=np.float32),  # y_obs masked
                np.zeros((n_q, 1), dtype=np.float32),  # is_ctx = 0
            ],
            axis=1,
        )

        seq = np.concatenate([ctx_tokens, q_tokens], axis=0).astype(np.float32)
        y_target = y_int_p[n_ctx:].astype(np.float32)

        return {
            "X": seq,  # (N, d+3)
            "y": y_target,  # (n_query,) interventional Y
            "n_ctx": n_ctx,
            "cate_true": cate_p.astype(np.float32),  # full-task CATE for eval
            "u_strength": np.float32(u_scale),
            "cov_correlation": np.float32(rho),
        }
