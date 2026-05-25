"""In-context regression scorer: PFN vs mean baseline vs closed-form OLS.

The model under test is a transformer trained to do in-context Bayesian
regression — given a packed (context, query) sequence, predict the y's
at the query positions from the (x, y) pairs at the context positions.
This scorer evaluates how well the trained model approximates the
Bayesian-optimal solution for a *univariate* Gaussian-linear prior:

  1. Sample N fresh tasks from the same prior the model was trained on
     (drawn via `pfnstudio_core.registry.get_prior(run_spec.prior.id)`).
     Each task carries an `n_ctx` boundary marking the context / query
     split inside the packed sequence — same convention `_default_step`
     uses to slice logits during training.
  2. Run the model on each packed sequence and read predictions at the
     query positions.
  3. Compare against two baselines on the same query positions:
       - **mean baseline** — predict the context-y mean. A useless model
         scores at this level.
       - **OLS** — closed-form least-squares fit on the context's (x, y)
         pairs. For a univariate Gaussian-linear prior this is the
         Bayesian-optimal predictor; a correctly trained PFN should
         approach it.
  4. Emit MSE values + ratios as metrics, plus meta about the sample.

Used by `studies/linear-regression-bayes/`. Skips cleanly (with a
descriptive reason) if the prior's samples don't carry `n_ctx` or `X`
isn't univariate (column 0 is treated as x for the OLS fit).

This is a *synthetic* scorer — `loader` is ignored. The dataset for the
eval is fresh draws from the run's own prior, not a registry table.
"""

from __future__ import annotations

from .base import DatasetScorer, ScorerResult

NUM_TASKS = 50
POINTS_PER_TASK = 80
BASE_SEED = 10_000


class InContextRegressionVsOLS(DatasetScorer):
    """Compare a trained in-context regression PFN to mean + OLS baselines."""

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
        # Seed schedule: BASE_SEED + k so the scorer is reproducible across
        # runs and disjoint from the training seeds (which use seed + step
        # starting at run_spec.hyperparams.seed, typically 42).
        sse_pfn = 0.0
        sse_mean = 0.0
        sse_ols = 0.0
        total = 0
        recorded_ctx_fraction: float | None = None

        for k in range(NUM_TASKS):
            task = prior.sample(seed=BASE_SEED + k, num_points=POINTS_PER_TASK)
            seq = task.get("X")
            y_q = task.get("y")
            n_ctx = task.get("n_ctx")

            if seq is None or y_q is None or n_ctx is None:
                return ScorerResult(
                    metrics={},
                    meta={"prior_keys": sorted(list(task.keys()))},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit the packed in-context shape — need X, y, n_ctx. "
                        "This scorer matches priors that pack (context, query) into one "
                        "sequence (e.g. bayesian_linear)."
                    ),
                )

            seq_arr = np.asarray(seq, dtype=np.float32)
            if seq_arr.ndim != 2 or seq_arr.shape[1] < 1:
                return ScorerResult(
                    metrics={},
                    meta={"X_shape": list(seq_arr.shape)},
                    skipped=True,
                    skip_reason="Prior's X must be 2-D with at least 1 feature.",
                )

            n_ctx_i = int(n_ctx)
            if recorded_ctx_fraction is None:
                recorded_ctx_fraction = n_ctx_i / float(POINTS_PER_TASK)

            x_ctx_col = seq_arr[:n_ctx_i, 0]
            y_ctx = seq_arr[:n_ctx_i, 1] if seq_arr.shape[1] >= 2 else None
            x_q_col = seq_arr[n_ctx_i:, 0]
            y_q_arr = np.asarray(y_q, dtype=np.float32)

            with torch.no_grad():
                out = torch.from_numpy(seq_arr).unsqueeze(0)
                for _, mod in model.modules:
                    out = mod(out)
                preds = out[0, n_ctx_i:, 0].cpu().numpy()

            sse_pfn += float(np.sum((preds - y_q_arr) ** 2))

            # Mean baseline + OLS on the context — both require y_ctx
            # available, which it is for the packed in-context shape.
            if y_ctx is not None:
                mean_pred = np.full_like(y_q_arr, float(y_ctx.mean()))
                a_ols, b_ols = np.polyfit(x_ctx_col, y_ctx, 1)
                ols_pred = (a_ols * x_q_col + b_ols).astype(np.float32)
                sse_mean += float(np.sum((mean_pred - y_q_arr) ** 2))
                sse_ols += float(np.sum((ols_pred - y_q_arr) ** 2))

            total += int(y_q_arr.shape[0])

        if total == 0:
            return ScorerResult(
                metrics={},
                meta={},
                skipped=True,
                skip_reason="No query points scored.",
            )

        pfn_mse = sse_pfn / total
        mean_mse = sse_mean / total
        ols_mse = sse_ols / total

        # Ratios are the headline numbers we surface in the README; the
        # UI typically renders them as "100x better than mean" etc.
        ratio_vs_mean = (mean_mse / pfn_mse) if pfn_mse > 0 else 0.0
        ratio_vs_ols = (pfn_mse / ols_mse) if ols_mse > 0 else 0.0

        return ScorerResult(
            metrics={
                "pfn_mse": pfn_mse,
                "mean_baseline_mse": mean_mse,
                "ols_mse": ols_mse,
                "ratio_vs_mean": ratio_vs_mean,
                "ratio_vs_ols": ratio_vs_ols,
            },
            meta={
                "tasks": NUM_TASKS,
                "points_per_task": POINTS_PER_TASK,
                "context_fraction": recorded_ctx_fraction or 0.0,
                "base_seed": BASE_SEED,
                "note": (
                    "PFN vs mean vs OLS on tasks drawn fresh from the training prior. "
                    "OLS is the Bayesian-optimal predictor for a univariate Gaussian-linear "
                    "prior; a correctly trained in-context PFN should approach it."
                ),
            },
        )
