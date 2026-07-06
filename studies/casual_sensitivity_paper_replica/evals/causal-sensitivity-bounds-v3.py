from __future__ import annotations

from pfnstudio_core.registry import register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult


@register_scorer("causal-sensitivity-bounds-v3")
class CausalSensitivityBoundsV3Scorer(DatasetScorer):
    """Scores a two-headed (theta_lower, theta_upper) bounds model against
    fresh draws from causal_sensitivity_packed.

    Ground truth: the prior packs context rows (is_context=1) on top of
    query rows (is_context=0). Query rows carry theta_star (the Lagrangian
    bound label) and bound_type (0=upper, 1=lower) as diagnostic outputs.
    We split query rows by bound_type and score each head against its
    matching subset.
    """

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        import numpy as np
        import torch

        from pfnstudio_core.priors import get_prior

        prior = get_prior(run_spec.prior.id)

        n_eval_tasks = 16
        lower_sq_errs: list[float] = []
        upper_sq_errs: list[float] = []
        width_abs_errs: list[float] = []
        non_crossing = 0
        total_pairs = 0

        for i in range(n_eval_tasks):
            seed = 10_000 + i
            sample = prior.sample(seed=seed)

            X = np.asarray(sample["X"], dtype=np.float32)
            n_ctx = int(np.asarray(sample["n_ctx"]).reshape(()))
            theta_star = np.asarray(sample["theta_star"], dtype=np.float32).reshape(-1)
            bound_type = np.asarray(sample["bound_type"], dtype=np.float32).reshape(-1)
            query_id = np.asarray(sample.get("query_id", np.arange(len(theta_star))))
            query_id = query_id.reshape(-1)

            if X.shape[0] <= n_ctx:
                continue

            inp = torch.from_numpy(X).unsqueeze(0)

            out = inp
            for _, mod in model.modules:
                out = mod(out)

            preds = out[0, n_ctx:, :]
            if preds.shape[-1] < 2:
                # Model only has one usable output column; skip bound-pair
                # diagnostics but still score whichever column exists as
                # if it were the upper bound, so the eval still reports
                # something rather than silently skipping.
                pred_single = preds[:, 0].detach().cpu().numpy()
                upper_mask = bound_type == 0
                lower_mask = bound_type == 1
                if upper_mask.any():
                    upper_sq_errs.extend(
                        ((pred_single[upper_mask] - theta_star[upper_mask]) ** 2).tolist()
                    )
                if lower_mask.any():
                    lower_sq_errs.extend(
                        ((pred_single[lower_mask] - theta_star[lower_mask]) ** 2).tolist()
                    )
                continue

            pred_lower = preds[:, 0].detach().cpu().numpy()
            pred_upper = preds[:, 1].detach().cpu().numpy()

            upper_mask = bound_type == 0
            lower_mask = bound_type == 1

            if upper_mask.any():
                upper_sq_errs.extend(
                    ((pred_upper[upper_mask] - theta_star[upper_mask]) ** 2).tolist()
                )
            if lower_mask.any():
                lower_sq_errs.extend(
                    ((pred_lower[lower_mask] - theta_star[lower_mask]) ** 2).tolist()
                )

            # Pair up lower/upper predictions per query_id to check
            # non-crossing + width error.
            common_ids = np.intersect1d(query_id[lower_mask], query_id[upper_mask])
            for qid in common_ids:
                l_idx = np.where((query_id == qid) & lower_mask)[0]
                u_idx = np.where((query_id == qid) & upper_mask)[0]
                if len(l_idx) == 0 or len(u_idx) == 0:
                    continue
                l_pred = float(pred_lower[l_idx[0]])
                u_pred = float(pred_upper[u_idx[0]])
                l_true = float(theta_star[l_idx[0]])
                u_true = float(theta_star[u_idx[0]])

                total_pairs += 1
                if l_pred <= u_pred:
                    non_crossing += 1

                width_pred = u_pred - l_pred
                width_true = u_true - l_true
                width_abs_errs.append(abs(width_pred - width_true))

        if not lower_sq_errs and not upper_sq_errs:
            return ScorerResult(
                metrics={},
                meta={"n_eval_tasks": n_eval_tasks},
                skipped=True,
                skip_reason="No query rows scored — check prior output keys (X, n_ctx, theta_star, bound_type) or model output width.",
            )

        lower_rmse = float(np.sqrt(np.mean(lower_sq_errs))) if lower_sq_errs else float("nan")
        upper_rmse = float(np.sqrt(np.mean(upper_sq_errs))) if upper_sq_errs else float("nan")
        coverage = float(non_crossing / total_pairs) if total_pairs > 0 else float("nan")
        interval_width_mae = float(np.mean(width_abs_errs)) if width_abs_errs else float("nan")

        return ScorerResult(
            metrics={
                "lower_rmse": lower_rmse,
                "upper_rmse": upper_rmse,
                "coverage": coverage,
                "interval_width_mae": interval_width_mae,
            },
            meta={
                "n_eval_tasks": n_eval_tasks,
                "n_lower_points": len(lower_sq_errs),
                "n_upper_points": len(upper_sq_errs),
                "n_pairs": total_pairs,
            },
        )