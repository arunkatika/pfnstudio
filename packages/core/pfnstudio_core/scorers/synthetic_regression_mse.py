"""Synthetic regression scorer: PFN MSE vs mean-baseline on fresh tasks.

Drop-in eval for any in-context regression brain — samples fresh tasks
from the run's own prior, runs the trained model, and reports the PFN's
forecast MSE against a "predict the context mean" baseline.

Used by wizard-built brains whose declared eval slug doesn't match a
prior-specific scorer (`closed_form_baseline`, `m4_monthly_mse`, etc.)
The wizard generates this scorer's slug — `synthetic_regression_mse` —
on `buildEval` for any regression speciality so the Evals card on the
run-detail page has metrics to surface.

Why a mean baseline (not OLS)? OLS only makes sense when the prior is
univariate Gaussian-linear; this scorer has to work across every
regression prior we ship (AR(2), pfns_reference, lc_pfn, ifbo,
tabpfn_ts, …). The context-mean baseline is the cheapest non-trivial
predictor: any in-context learner that has actually learned the prior
should beat it by a wide margin. Studies that want the tighter OLS
comparison opt into `in_context_regression_ols` explicitly.

This is a *synthetic* scorer — `loader` is ignored. The dataset is
fresh draws from the run's own prior, not a registry table.
"""

from __future__ import annotations

from .base import DatasetScorer, ScorerResult

NUM_TASKS = 32
BASE_SEED = 30_000


class SyntheticRegressionMSE(DatasetScorer):
    """MSE on fresh tasks from the run's prior, vs context-mean baseline."""

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
        from ..training.loop import _split_encoder_heads  # type: ignore

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
        # Reuse prior parameter overrides from the run config so the eval
        # samples land in the same distribution the trainer saw — same
        # num_points, noise_scale, etc. The scorer otherwise might pull
        # the prior's own defaults which can mismatch (e.g. num_points
        # could be 64 in training but 100 default in the prior).
        prior_overrides = dict(getattr(run_spec.prior, "overrides", {}) or {})

        # Inference uses the encoder+heads split for safety: with the
        # multi-head fan-out the trainer enables, naively iterating
        # model.modules would chain head #2 off head #1's output and
        # crash. Pick the first head whose output is scalar per token.
        modules = list(getattr(model, "modules", []))
        encoder_blocks, head_blocks = _split_encoder_heads(modules)

        sse_pfn = 0.0
        sse_mean = 0.0
        total = 0
        sample_n_ctx = None
        sample_points = None

        for k in range(NUM_TASKS):
            task = prior.sample(seed=BASE_SEED + k, **prior_overrides)
            seq = task.get("X")
            y_q = task.get("y")
            n_ctx = task.get("n_ctx")

            if seq is None or y_q is None:
                return ScorerResult(
                    metrics={},
                    meta={"prior_keys": sorted(list(task.keys()))},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit the X + y shape this scorer needs. "
                        "Wizard-built regression priors should emit both — if this "
                        "skipped, the prior is non-standard and needs a dedicated scorer."
                    ),
                )

            seq_arr = np.asarray(seq, dtype=np.float32)
            y_q_arr = np.asarray(y_q, dtype=np.float32)
            if seq_arr.ndim < 2:
                return ScorerResult(
                    metrics={},
                    meta={"X_shape": list(seq_arr.shape)},
                    skipped=True,
                    skip_reason="Prior's X must have at least 2 dimensions (N, features).",
                )

            with torch.no_grad():
                x = torch.from_numpy(seq_arr).unsqueeze(0)
                # Encoder once.
                for mod in encoder_blocks:
                    x = mod(x)
                # Heads in parallel; pick the first that produces (B, N, 1)
                # (squeezable to the y_q target shape). Mirrors the
                # trainer's regression branch.
                preds_arr = None
                head_outputs = [head(x) for head in head_blocks] if head_blocks else [x]
                for ho in head_outputs:
                    pred = ho
                    if n_ctx is not None:
                        pred = pred[:, int(n_ctx) :, :]
                    if pred.dim() == 3 and pred.shape[-1] == 1:
                        pred = pred.squeeze(-1)
                    if pred.shape[-1] == y_q_arr.shape[-1]:
                        preds_arr = pred[0].cpu().numpy()
                        break
                if preds_arr is None:
                    return ScorerResult(
                        metrics={},
                        meta={"head_count": len(head_blocks)},
                        skipped=True,
                        skip_reason=(
                            "No head produced an output matching the y target shape. "
                            "Check that the model has a scalar_head with d_out=1."
                        ),
                    )

            sse_pfn += float(np.sum((preds_arr - y_q_arr) ** 2))

            # Context-mean baseline: predict the mean of the context y's.
            # Read y from the prior task's full sequence when packed;
            # otherwise the prior probably gives `y` aligned with `X`
            # over all positions, so use the first n_ctx (or first half
            # if there's no n_ctx) as the "context" for the baseline.
            if n_ctx is not None and seq_arr.shape[1] >= 2:
                # Packed-token convention: column 1 of context rows is y.
                y_ctx = seq_arr[: int(n_ctx), 1]
            elif n_ctx is None:
                # Non-packed: full y is available as the task's y, but
                # split has been done — fall back to first half of y_q.
                # This is a defensive path; non-packed priors usually
                # ship the full y in the task dict.
                y_ctx = y_q_arr[: max(1, len(y_q_arr) // 2)]
            else:
                y_ctx = y_q_arr[: max(1, int(n_ctx))]
            mean_pred = np.full_like(y_q_arr, float(y_ctx.mean()))
            sse_mean += float(np.sum((mean_pred - y_q_arr) ** 2))

            total += int(y_q_arr.shape[0])
            if sample_n_ctx is None and n_ctx is not None:
                sample_n_ctx = int(n_ctx)
            if sample_points is None:
                sample_points = int(seq_arr.shape[0])

        if total == 0:
            return ScorerResult(
                metrics={},
                meta={},
                skipped=True,
                skip_reason="No query points scored.",
            )

        mse = sse_pfn / total
        mean_mse = sse_mean / total
        # Ratio > 1 means the brain beats the baseline.
        ratio_vs_mean = (mean_mse / mse) if mse > 0 else 0.0

        return ScorerResult(
            metrics={
                "mse": mse,
                "mean_baseline_mse": mean_mse,
                "ratio_vs_mean": ratio_vs_mean,
            },
            meta={
                "tasks": NUM_TASKS,
                "points_per_task": sample_points or 0,
                "n_ctx_sample": sample_n_ctx,
                "base_seed": BASE_SEED,
                "note": (
                    "PFN MSE vs context-mean baseline on fresh tasks from the trained "
                    "prior. ratio_vs_mean > 1 means the brain is genuinely using context; "
                    "≈ 1 means it's no better than predicting the mean."
                ),
            },
        )
