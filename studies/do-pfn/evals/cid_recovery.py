"""Do-PFN CID + CATE recovery scorer — ships in the template, not core.

This is the executable half of `evals/cid_recovery.yaml`. It is registered by
slug via `@register_scorer("cid_recovery")` and discovered at run time by
`pfnstudio_core.registry.discover_in_project` (which imports `evals/*.py`), so
the Do-PFN template owns its own paper-specific scoring — exactly like its
`prior.py`. Nothing paper-specific lives in pfnstudio-core.

Adapted to the current random-DAG `do_pfn_scm_studio` prior, whose packed
token layout is `[t, x_1..x_d, y]` (width d+2): column 0 is the binary
treatment, columns 1..d the covariates, column d+1 the outcome (real on
context rows, NaN on query rows — the in-context learning marker). There is no
separate is_context column and no `intervention_t` knob, so the do(0)/do(1)
contrast is formed by editing the query treatment column directly — the same
way you'd query the trained model for a CATE at inference.

What it computes (all numbers are measured, none are placeholders):

  For NUM_TASKS fresh SCMs drawn from the run's own prior:
    • CID   — the model's predicted interventional outcome (query rows carry
              the SCM's own do(T=t) intervention) vs the prior's oracle Y_int.
    • CATE  — query the model twice on the SAME context, forcing every query
              treatment to 0 then to 1, and take (pred_do1 − pred_do0). Scored
              against the prior's latent `cate_true`.
    • naive — a per-task T-learner ridge (fit E[Y|X] on the observed T=1 and
              T=0 context arms separately, difference on the query X). Biased
              under the unobserved confounder — the baseline Do-PFN beats.
    • oracle — the latent CATE the prior emits; MSE against itself is 0 by
              construction (the achievable floor).

The model's final block is a distributional (bar) head, so its raw output is
per-bucket logits; the block's generic `to_prediction` reduces them to the
distribution mean before scoring (a plain scalar_head has no such method and
passes through unchanged).

Metrics returned match `cid_recovery.yaml`:
  cid_mse, cate_mse, naive_cate_mse, oracle_cate_mse,
  ratio_vs_naive_cate (naive/pfn, >1 = model beats naive),
  ratio_vs_oracle_cate (pfn/oracle; oracle floor ≈ 0, so this reports the
    absolute CATE MSE — 0 = perfect recovery — see meta['oracle_floor']),
  ate_error (|mean model CATE − mean latent CATE|).
"""

from __future__ import annotations

# Absolute imports — this module is loaded by discover_in_project via
# exec_module (no package context), so relative imports would fail.
from pfnstudio_core.registry import register_scorer
from pfnstudio_core.scorers.base import DatasetScorer, ScorerResult

# 50 fresh DGPs per the eval spec. Points-per-task kept small so scoring is
# quick even for the 12-layer axial model; seeds are large + fixed so scoring
# is reproducible and disjoint from the training seed schedule.
NUM_TASKS = 50
SCORER_POINTS = 256
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
        from pfnstudio_core.training.loop import _split_encoder_heads

        try:
            prior_cls = get_prior(run_spec.prior.id)
        except KeyError:
            return ScorerResult(
                metrics={}, meta={"prior_id": run_spec.prior.id},
                skipped=True,
                skip_reason=f"prior '{run_spec.prior.id}' not registered in this project",
            )
        prior = prior_cls()

        modules = list(model.modules)
        encoder, heads = _split_encoder_heads(modules)
        # The distributional head's logits→mean reducer, if any.
        reduce_fn = None
        for _, mod in modules:
            fn = getattr(mod, "to_prediction", None)
            if callable(fn):
                reduce_fn = fn
                break

        def run_model(seq: np.ndarray, n_ctx: int) -> np.ndarray:
            """Forward exactly as the trainer does — n_ctx-aware blocks get
            single_eval_pos, tuple outputs are unwrapped — then reduce the
            head output to a scalar per query position."""
            with torch.no_grad():
                enc = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
                for mod in encoder:
                    if n_ctx > 0 and getattr(mod, "needs_single_eval_pos", False):
                        r = mod(enc, single_eval_pos=n_ctx)
                    else:
                        r = mod(enc)
                    enc = r[0] if isinstance(r, tuple) else r
                out = heads[0](enc) if heads else enc
                if reduce_fn is not None:
                    out = reduce_fn(out)
                return out[0, n_ctx:, 0].cpu().numpy().astype(np.float64)

        sse_cid = 0.0          # predicted Y_int vs oracle Y_int
        n_cid = 0
        sse_cate_pfn = 0.0     # (pred1-pred0) vs cate_true
        sse_cate_naive = 0.0   # T-learner ridge vs cate_true
        n_cate = 0
        naive_rows = 0
        sum_cate_pfn = 0.0
        sum_cate_true = 0.0

        for k in range(NUM_TASKS):
            task = prior.sample(seed=BASE_SEED + k, num_samples=SCORER_POINTS)
            seq = task.get("X")
            y_q = task.get("y")
            n_ctx = task.get("n_ctx")
            cate_true = task.get("cate_true")
            if seq is None or y_q is None or n_ctx is None or cate_true is None:
                return ScorerResult(
                    metrics={}, meta={"prior_keys": sorted(task.keys())},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit the Do-PFN shape (need X, y, n_ctx, cate_true)."
                    ),
                )

            seq = np.asarray(seq, dtype=np.float64)
            n_ctx = int(n_ctx)
            width = seq.shape[1]
            d = width - 2  # [t, x_1..x_d, y]
            if seq.ndim != 2 or d < 1:
                return ScorerResult(
                    metrics={}, meta={"token_width": int(width)},
                    skipped=True, skip_reason="Token width < 3 — not the Do-PFN packing.",
                )

            y_q = np.asarray(y_q, dtype=np.float64)
            cate_q = np.asarray(cate_true, dtype=np.float64)[n_ctx:]

            # CID: the model on the query as generated (its own do(T=t_int)).
            pred_orig = run_model(seq, n_ctx)
            sse_cid += float(np.sum((pred_orig - y_q) ** 2))
            n_cid += int(y_q.shape[0])

            # CATE: same context, force every query treatment to 0 then 1.
            seq0 = seq.copy(); seq0[n_ctx:, 0] = 0.0
            seq1 = seq.copy(); seq1[n_ctx:, 0] = 1.0
            cate_pfn = run_model(seq1, n_ctx) - run_model(seq0, n_ctx)
            sse_cate_pfn += float(np.sum((cate_pfn - cate_q) ** 2))
            n_cate += int(cate_q.shape[0])
            sum_cate_pfn += float(np.sum(cate_pfn))
            sum_cate_true += float(np.sum(cate_q))

            # Naive T-learner ridge on the OBSERVED (confounded) context.
            x_ctx = seq[:n_ctx, 1 : width - 1]
            t_ctx = seq[:n_ctx, 0]
            y_ctx = seq[:n_ctx, width - 1]
            x_q = seq[n_ctx:, 1 : width - 1]
            m1 = t_ctx > 0.5
            m0 = ~m1
            if int(m1.sum()) >= d + 1 and int(m0.sum()) >= d + 1:
                w1 = _ridge_fit(x_ctx[m1], y_ctx[m1], RIDGE_LAMBDA, np)
                w0 = _ridge_fit(x_ctx[m0], y_ctx[m0], RIDGE_LAMBDA, np)
                cate_naive = _ridge_pred(w1, x_q, np) - _ridge_pred(w0, x_q, np)
                sse_cate_naive += float(np.sum((cate_naive - cate_q) ** 2))
                naive_rows += int(cate_q.shape[0])

        if n_cate == 0:
            return ScorerResult(
                metrics={}, meta={}, skipped=True,
                skip_reason="No query positions scored.",
            )

        cid_mse = sse_cid / max(n_cid, 1)
        cate_mse = sse_cate_pfn / n_cate
        naive_cate_mse = (sse_cate_naive / naive_rows) if naive_rows else float("nan")
        oracle_cate_mse = 0.0  # oracle predicts latent cate_true → floor is 0
        ate_error = abs(sum_cate_pfn / n_cate - sum_cate_true / n_cate)

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
            "ate_error": ate_error,
        }
        meta = {
            "num_tasks": NUM_TASKS,
            "points_per_task": SCORER_POINTS,
            "naive_rows_scored": naive_rows,
            "cate_method": "paired do(0)/do(1) by forcing the query treatment column",
            "oracle_floor": "latent cate_true; MSE against itself is 0 by construction",
        }
        return ScorerResult(metrics=metrics, meta=meta)
