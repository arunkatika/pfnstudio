"""Closed-form Bayesian baseline scorer for the bayesian_linear prior.

For a Bayesian linear regression with prior ``w ~ N(0, σ_w²·I)`` and
likelihood ``y | x, w ~ N(w·[x, 1], σ_n²)``, the posterior over w
given a context of (x, y) observations is analytical:

    Σ_n = (XᵀX / σ_n² + I / σ_w²)^-1
    μ_n = Σ_n · Xᵀy / σ_n²

The Bayes-optimal point predictor at a new x_q is then ``x_q^T μ_n``.
A correctly-trained PFN should match this prediction within numerical
noise — that's the canonical claim of Müller 2022.

This scorer:
  1. Samples N fresh tasks from the bayesian_linear prior (same
     packed-token shape the trainer used).
  2. Runs the model on the full packed sequence and reads predictions
     at query positions.
  3. Computes the closed-form posterior mean using only the context's
     (x, y) pairs and the prior's documented σ_w, σ_n parameters.
  4. Emits:
       - ``rmse``                — PFN MSE against the latent
                                    (noise-free) y_q values.
       - ``rmse_vs_posterior``   — RMSE between PFN predictions and the
                                    analytic posterior mean. **A
                                    correctly trained PFN drives this
                                    to zero**, which is the verdict the
                                    reproducibility badge looks at.

Slug: ``closed_form_baseline``. Bound to PFNS_TEMPLATE in
``apps/web/src/app/projects/templates.ts``.
"""

from __future__ import annotations

from typing import Any

from .base import DatasetScorer, ScorerResult

NUM_TASKS = 50
POINTS_PER_TASK = 100
BASE_SEED = 20_000


def _posterior_mean_prediction(
    x_ctx: Any,
    y_ctx: Any,
    x_qry: Any,
    *,
    weight_std: float,
    noise_scale: float,
) -> Any:
    """Closed-form Bayes-optimal prediction at query x's.

    Designs the augmented matrix [x, 1] so the bias term is part of
    the Bayesian update. Returns predictions at x_qry as a 1-D array.
    """
    import numpy as np

    n = int(x_ctx.shape[0])
    Xc = np.stack([x_ctx, np.ones(n, dtype=np.float32)], axis=1)  # (n, 2)
    Xq = np.stack([x_qry, np.ones(int(x_qry.shape[0]), dtype=np.float32)], axis=1)  # (m, 2)

    sigma_w2 = float(weight_std) ** 2
    sigma_n2 = float(noise_scale) ** 2

    # Σ_n = (XᵀX / σ_n² + I / σ_w²)^-1
    XtX = Xc.T @ Xc  # (2, 2)
    precision = XtX / sigma_n2 + np.eye(2, dtype=np.float32) / sigma_w2
    cov = np.linalg.inv(precision)
    mean_w = (cov @ Xc.T @ y_ctx.astype(np.float32)) / sigma_n2  # (2,)
    return (Xq @ mean_w).astype(np.float32)  # (m,)


class ClosedFormBaseline(DatasetScorer):
    """PFN vs analytic Bayesian posterior on a univariate linear prior."""

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        try:
            import numpy as np
            import torch
        except ImportError as e:
            return ScorerResult(
                metrics={},
                meta={"dependency_missing": str(e)},
                skipped=True,
                skip_reason=f"missing dependency: {e}",
            )

        from ..registry import get_prior

        try:
            prior_cls = get_prior(run_spec.prior.id)
        except KeyError:
            return ScorerResult(
                metrics={},
                meta={"prior_id": run_spec.prior.id},
                skipped=True,
                skip_reason=f"prior '{run_spec.prior.id}' not registered in this project",
            )

        prior = prior_cls()

        # Read the prior's σ_w / σ_n from the run's overrides; fall
        # back to the prior's documented defaults if unset.
        overrides = dict(getattr(run_spec.prior, "overrides", {}) or {})
        weight_std = float(overrides.get("weight_std", 1.0))
        noise_scale = float(overrides.get("noise_scale", 0.1))

        sse_latent = 0.0
        sse_posterior = 0.0
        total = 0
        recorded_ctx_fraction: float | None = None

        for k in range(NUM_TASKS):
            task = prior.sample(
                seed=BASE_SEED + k,
                num_points=POINTS_PER_TASK,
                **{k: v for k, v in overrides.items() if k != "num_points"},
            )
            seq = task.get("X")
            y_q = task.get("y")
            n_ctx = task.get("n_ctx")
            a_true = task.get("a_true")
            b_true = task.get("b_true")

            if seq is None or y_q is None or n_ctx is None:
                return ScorerResult(
                    metrics={},
                    meta={"prior_keys": sorted(list(task.keys()))},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit the packed-token shape this scorer needs "
                        "(X, y, n_ctx). bayesian_linear @ 0.2.0+ should produce these."
                    ),
                )

            seq_arr = np.asarray(seq, dtype=np.float32)
            if seq_arr.ndim != 2 or seq_arr.shape[1] < 3:
                return ScorerResult(
                    metrics={},
                    meta={"X_shape": list(seq_arr.shape)},
                    skipped=True,
                    skip_reason=(
                        "Closed-form scorer needs packed tokens of width ≥ 3 "
                        "(x, y_or_zero, is_context_flag). Got width "
                        f"{seq_arr.shape[1] if seq_arr.ndim == 2 else '?'}."
                    ),
                )

            n_ctx_i = int(n_ctx)
            if recorded_ctx_fraction is None:
                recorded_ctx_fraction = n_ctx_i / float(POINTS_PER_TASK)

            x_ctx = seq_arr[:n_ctx_i, 0]
            y_ctx = seq_arr[:n_ctx_i, 1]
            x_qry = seq_arr[n_ctx_i:, 0]
            y_q_arr = np.asarray(y_q, dtype=np.float32)

            with torch.no_grad():
                out = torch.from_numpy(seq_arr).unsqueeze(0)
                for _, mod in model.modules:
                    out = mod(out)
                preds = out[0, n_ctx_i:, 0].cpu().numpy()

            # Latent (noise-free) y_q for the rmse metric — what the
            # paper calls "y_true". We have a_true + b_true on the task
            # dict, so reconstruct them directly.
            if a_true is not None and b_true is not None:
                y_latent = (float(a_true) * x_qry + float(b_true)).astype(np.float32)
            else:
                y_latent = y_q_arr  # fall back to noisy y if latent unavailable

            # Analytic posterior mean prediction on the same query x's.
            y_post = _posterior_mean_prediction(
                x_ctx,
                y_ctx,
                x_qry,
                weight_std=weight_std,
                noise_scale=noise_scale,
            )

            sse_latent += float(np.sum((preds - y_latent) ** 2))
            sse_posterior += float(np.sum((preds - y_post) ** 2))
            total += int(y_q_arr.shape[0])

        if total == 0:
            return ScorerResult(
                metrics={},
                meta={},
                skipped=True,
                skip_reason="No query points scored.",
            )

        # RMSE against latent y_q (with the prior's noise floor implicit
        # — the irreducible error is √(sigma_n² + x_q² Σ_n x_q)).
        rmse = float(np.sqrt(sse_latent / total))
        # RMSE against the analytic posterior mean — this is the value
        # the verdict badge looks at. A correctly trained PFN drives
        # this near zero.
        rmse_vs_posterior = float(np.sqrt(sse_posterior / total))

        return ScorerResult(
            metrics={
                "rmse": rmse,
                "rmse_vs_posterior": rmse_vs_posterior,
            },
            meta={
                "tasks": NUM_TASKS,
                "points_per_task": POINTS_PER_TASK,
                "context_fraction": recorded_ctx_fraction or 0.0,
                "base_seed": BASE_SEED,
                "weight_std": weight_std,
                "noise_scale": noise_scale,
                "note": (
                    "RMSE against the closed-form Bayesian posterior mean "
                    "(computed from the prior's σ_w / σ_n). A correctly trained "
                    "PFN should drive rmse_vs_posterior near zero — the canonical "
                    "Müller 2022 verdict."
                ),
            },
        )
