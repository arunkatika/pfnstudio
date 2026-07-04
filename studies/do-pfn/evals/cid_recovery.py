"""Do-PFN CID + CATE recovery scorer — ships in the template, not core.

This is the executable half of `evals/cid_recovery.yaml`. It is registered by
slug via `@register_scorer("cid_recovery")` and discovered at run time by
`pfnstudio_core.registry.discover_in_project` (which imports `evals/*.py`), so
the Do-PFN template owns its own paper-specific scoring — exactly like its
`prior.py`. Nothing paper-specific lives in pfnstudio-core.

What it computes (all numbers are measured, none are placeholders):

  For NUM_TASKS fresh SCMs drawn from the run's own prior (`do_pfn_scm`):
    • Query the trained model twice on the SAME context — once with every
      query intervention fixed to do(T=0), once to do(T=1). The prior is
      built for this: fixing `intervention_t` changes only the query
      tokens, leaving the context (and the latent CATE) identical, so the
      two runs are a clean paired comparison.
    • CID   — predicted interventional outcome vs the SCM's oracle Y_int,
              pooled over both do(0) and do(1) query positions.
    • CATE  — (pred_do1 − pred_do0) vs the prior's latent `cate_true`.
    • naive — a per-task T-learner ridge (fit E[Y|X] on the observed T=1
              and T=0 context arms separately, take the difference on the
              query X). Biased under the unobserved confounder U — the
              baseline Do-PFN claims to beat.
    • oracle — the latent CATE the prior emits; its MSE against itself is 0
              by construction (the achievable floor).

Metrics returned match `cid_recovery.yaml`:
  cid_mse, cate_mse, naive_cate_mse, oracle_cate_mse,
  ratio_vs_naive_cate (naive/pfn, >1 = model beats naive),
  ratio_vs_oracle_cate (pfn/oracle; the oracle floor is ≈0 here, so this
    reports the absolute CATE MSE — 0 = perfect recovery — rather than
    dividing by zero; see `meta['oracle_floor']`).
"""

from __future__ import annotations

# Absolute imports — this module is loaded by discover_in_project via
# exec_module (no package context), so relative imports would fail.
from pfnstudio_core.registry import register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult

# 50 fresh DGPs per the eval spec. Seeds are large + fixed so scoring is
# reproducible and disjoint from the training seed schedule.
NUM_TASKS = 50
BASE_SEED = 900_000
RIDGE_LAMBDA = 1.0
_EPS = 1e-9


def _ridge_fit(x, y, lam, np):
    """Closed-form ridge with a bias term. x: (n, d) → weights (d+1,)."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    a = xb.T @ xb + lam * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y)


def _ridge_pred(w, x, np):
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    return xb @ w


@register_scorer("cid_recovery")
class DoPfnCidRecoveryScorer(DatasetScorer):
    """Measure interventional-outcome (CID) and CATE recovery vs baselines."""

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        try:
            import numpy as np
            import torch
        except ImportError as e:
            return ScorerResult(
                metrics={}, meta={"dependency_missing": str(e)},
                skipped=True, skip_reason=f"missing dependency: {e}",
            )

        from pfnstudio_core.registry import get_prior

        try:
            prior_cls = get_prior(run_spec.prior.id)
        except KeyError:
            return ScorerResult(
                metrics={}, meta={"prior_id": run_spec.prior.id},
                skipped=True,
                skip_reason=f"prior '{run_spec.prior.id}' not registered in this project",
            )
        prior = prior_cls()

        def run_model(seq: np.ndarray, n_ctx: int) -> np.ndarray:
            with torch.no_grad():
                out = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
                for _, mod in model.modules:
                    out = mod(out)
                return out[0, n_ctx:, 0].cpu().numpy().astype(np.float64)

        sse_cid = 0.0          # predicted Y_int vs oracle Y_int
        n_cid = 0
        sse_cate_pfn = 0.0     # (pred1-pred0) vs cate_true
        sse_cate_naive = 0.0   # T-learner ridge vs cate_true
        n_cate = 0
        naive_tasks = 0

        for k in range(NUM_TASKS):
            seed = BASE_SEED + k
            # Paired queries: identical context + latent CATE, only the query
            # intervention differs. Fixing intervention_t makes this exact.
            t0 = prior.sample(seed=seed, intervention_t=0.0)
            t1 = prior.sample(seed=seed, intervention_t=1.0)

            seq0 = t0.get("X")
            seq1 = t1.get("X")
            n_ctx = t0.get("n_ctx")
            cate_true = t0.get("cate_true")
            if seq0 is None or seq1 is None or n_ctx is None or cate_true is None:
                return ScorerResult(
                    metrics={}, meta={"prior_keys": sorted(t0.keys())},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit the Do-PFN shape (need X, n_ctx, "
                        "cate_true, and intervention_t support)."
                    ),
                )

            seq0 = np.asarray(seq0, dtype=np.float64)
            seq1 = np.asarray(seq1, dtype=np.float64)
            n_ctx = int(n_ctx)
            d = seq0.shape[1] - 3  # token width = d + (t, y, is_ctx)
            if d < 1:
                return ScorerResult(
                    metrics={}, meta={"token_width": seq0.shape[1]},
                    skipped=True, skip_reason="Token width < 4 — not the Do-PFN packing.",
                )

            y0_oracle = np.asarray(t0["y"], dtype=np.float64)  # Y_{do(0)} at queries
            y1_oracle = np.asarray(t1["y"], dtype=np.float64)  # Y_{do(1)} at queries
            cate_q = np.asarray(cate_true, dtype=np.float64)[n_ctx:]  # latent CATE, query rows

            pred0 = run_model(seq0, n_ctx)
            pred1 = run_model(seq1, n_ctx)

            # CID: predicted interventional outcome vs oracle, both arms pooled.
            sse_cid += float(np.sum((pred0 - y0_oracle) ** 2) + np.sum((pred1 - y1_oracle) ** 2))
            n_cid += int(y0_oracle.shape[0] + y1_oracle.shape[0])

            # CATE from the paired queries.
            cate_pfn = pred1 - pred0
            sse_cate_pfn += float(np.sum((cate_pfn - cate_q) ** 2))
            n_cate += int(cate_q.shape[0])

            # Naive T-learner ridge on the OBSERVED context (confounded).
            x_ctx = seq0[:n_ctx, :d]
            t_ctx = seq0[:n_ctx, d]
            y_ctx = seq0[:n_ctx, d + 1]
            x_q = seq0[n_ctx:, :d]
            m1 = t_ctx > 0.5
            m0 = ~m1
            if int(m1.sum()) >= d + 1 and int(m0.sum()) >= d + 1:
                w1 = _ridge_fit(x_ctx[m1], y_ctx[m1], RIDGE_LAMBDA, np)
                w0 = _ridge_fit(x_ctx[m0], y_ctx[m0], RIDGE_LAMBDA, np)
                cate_naive = _ridge_pred(w1, x_q, np) - _ridge_pred(w0, x_q, np)
                sse_cate_naive += float(np.sum((cate_naive - cate_q) ** 2))
                naive_tasks += 1

        if n_cate == 0:
            return ScorerResult(
                metrics={}, meta={}, skipped=True,
                skip_reason="No query positions scored.",
            )

        cid_mse = sse_cid / max(n_cid, 1)
        cate_mse = sse_cate_pfn / n_cate
        # naive averaged over the query rows it actually covered.
        naive_cate_mse = (
            sse_cate_naive / (naive_tasks * (n_cate / NUM_TASKS))
            if naive_tasks
            else float("nan")
        )
        oracle_cate_mse = 0.0  # oracle predicts latent cate_true → floor is 0

        metrics = {
            "cid_mse": cid_mse,
            "cate_mse": cate_mse,
            "naive_cate_mse": naive_cate_mse,
            "oracle_cate_mse": oracle_cate_mse,
            # >1 means the model's CATE beats the confounded observational ridge.
            "ratio_vs_naive_cate": (naive_cate_mse / cate_mse) if cate_mse > _EPS else float("inf"),
            # Oracle floor ≈ 0, so this degenerates to the absolute CATE MSE
            # (0 = perfect causal recovery) rather than a divide-by-zero.
            "ratio_vs_oracle_cate": cate_mse,
        }
        meta = {
            "num_tasks": NUM_TASKS,
            "naive_tasks_scored": naive_tasks,
            "query_positions_per_task": n_cate // NUM_TASKS,
            "oracle_floor": "latent cate_true; MSE against itself is 0 by construction",
        }
        return ScorerResult(metrics=metrics, meta=meta)
