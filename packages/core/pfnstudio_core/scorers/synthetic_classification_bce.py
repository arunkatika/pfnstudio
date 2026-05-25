"""Synthetic classification scorer: PFN BCE / accuracy vs majority baseline.

Drop-in eval for any in-context binary classification brain — samples
fresh tasks from the run's own prior, runs the trained model, and
reports the PFN's BCE + accuracy against a "predict the majority class"
baseline.

Wired by the wizard for any classification speciality (its priors emit
`labels` in addition to `X`). Pairs with `synthetic_regression_mse` so
every wizard-built brain has at least one runnable scorer regardless of
which speciality the user picks.

This is a *synthetic* scorer — `loader` is ignored. The dataset is
fresh draws from the run's own prior, not a registry table.
"""

from __future__ import annotations

import math

from .base import DatasetScorer, ScorerResult

NUM_TASKS = 32
BASE_SEED = 40_000
# Probability clamp for BCE stability — the logsig path is exact but
# downstream code reading `bce` shouldn't have to deal with inf when a
# fresh-from-init model emits extreme logits.
EPS = 1e-7


class SyntheticClassificationBCE(DatasetScorer):
    """BCE + accuracy on fresh tasks from the run's prior, vs majority baseline."""

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
        prior_overrides = dict(getattr(run_spec.prior, "overrides", {}) or {})

        # Same encoder+heads split as the trainer + regression scorer —
        # safe with multi-head models.
        modules = list(getattr(model, "modules", []))
        encoder_blocks, head_blocks = _split_encoder_heads(modules)

        total_bce = 0.0
        total_majority_bce = 0.0
        total_correct = 0
        total_majority_correct = 0
        total = 0
        majority_class_sum = 0.0  # sum of label means across tasks
        sample_n_ctx = None
        sample_points = None
        # For AUC: collect predicted prob + true label across all tasks
        # so we compute one ROC across the whole eval, not per-task.
        all_probs: list[float] = []
        all_labels: list[int] = []

        for k in range(NUM_TASKS):
            task = prior.sample(seed=BASE_SEED + k, **prior_overrides)
            seq = task.get("X")
            labels_q = task.get("labels")
            n_ctx = task.get("n_ctx")

            if seq is None or labels_q is None:
                return ScorerResult(
                    metrics={},
                    meta={"prior_keys": sorted(list(task.keys()))},
                    skipped=True,
                    skip_reason=(
                        "Prior didn't emit X + labels. Wizard-built classification "
                        "priors should emit both; if this skipped, the prior is "
                        "non-standard and needs a dedicated scorer."
                    ),
                )

            seq_arr = np.asarray(seq, dtype=np.float32)
            labels_q_arr = np.asarray(labels_q, dtype=np.float32)
            if seq_arr.ndim < 2:
                return ScorerResult(
                    metrics={},
                    meta={"X_shape": list(seq_arr.shape)},
                    skipped=True,
                    skip_reason="Prior's X must have at least 2 dimensions (N, features).",
                )

            with torch.no_grad():
                x = torch.from_numpy(seq_arr).unsqueeze(0)
                for mod in encoder_blocks:
                    x = mod(x)
                logits_arr = None
                head_outputs = [head(x) for head in head_blocks] if head_blocks else [x]
                for ho in head_outputs:
                    pred = ho
                    if n_ctx is not None:
                        pred = pred[:, int(n_ctx) :, :]
                    if pred.dim() == 3 and pred.shape[-1] == 1:
                        pred = pred.squeeze(-1)
                    if pred.shape[-1] == labels_q_arr.shape[-1]:
                        logits_arr = pred[0].cpu().numpy()
                        break
                if logits_arr is None:
                    return ScorerResult(
                        metrics={},
                        meta={"head_count": len(head_blocks)},
                        skipped=True,
                        skip_reason=(
                            "No head produced an output matching the labels target "
                            "shape. Check that the model has a scalar_head with d_out=1."
                        ),
                    )

            probs = 1.0 / (1.0 + np.exp(-logits_arr))
            probs = np.clip(probs, EPS, 1.0 - EPS)
            preds = (probs >= 0.5).astype(np.float32)

            # Per-token BCE
            bce_terms = -(labels_q_arr * np.log(probs) + (1.0 - labels_q_arr) * np.log(1.0 - probs))
            total_bce += float(np.sum(bce_terms))
            total_correct += int(np.sum(preds == labels_q_arr))

            # Majority baseline: read the context labels when possible.
            # For packed-token priors column [-2] (just before the
            # is_context flag) is the label. Fall back to the task's
            # held-out labels' own mean otherwise — gives the right
            # number when training and test labels share a distribution.
            if n_ctx is not None and seq_arr.shape[1] >= 3:
                ctx_labels = seq_arr[: int(n_ctx), -2]
            elif n_ctx is None and seq_arr.shape[1] >= 2:
                ctx_labels = seq_arr[:, -1]  # heuristic — last column
            else:
                ctx_labels = labels_q_arr
            majority_class = 1.0 if ctx_labels.mean() >= 0.5 else 0.0
            majority_class_sum += float(majority_class)
            majority_prob = float(np.clip(ctx_labels.mean(), EPS, 1.0 - EPS))
            majority_bce_terms = -(
                labels_q_arr * math.log(majority_prob)
                + (1.0 - labels_q_arr) * math.log(1.0 - majority_prob)
            )
            total_majority_bce += float(np.sum(majority_bce_terms))
            total_majority_correct += int(np.sum(majority_class == labels_q_arr))

            all_probs.extend(probs.tolist())
            all_labels.extend(labels_q_arr.astype(int).tolist())

            total += int(labels_q_arr.shape[0])
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

        bce = total_bce / total
        majority_bce = total_majority_bce / total
        accuracy = total_correct / total
        majority_accuracy = total_majority_correct / total

        # AUC computed across all (prob, label) pairs. Falls back to NaN
        # if the labels are single-class (AUC undefined) so the UI can
        # show a dash rather than a misleading number.
        try:
            auc = _roc_auc(all_probs, all_labels)
        except ValueError:
            auc = float("nan")

        return ScorerResult(
            metrics={
                "bce": bce,
                "accuracy": accuracy,
                "auc": auc,
                "majority_bce": majority_bce,
                "majority_accuracy": majority_accuracy,
            },
            meta={
                "tasks": NUM_TASKS,
                "points_per_task": sample_points or 0,
                "n_ctx_sample": sample_n_ctx,
                "base_seed": BASE_SEED,
                "majority_class_avg": majority_class_sum / max(1, NUM_TASKS),
                "note": (
                    "PFN BCE / accuracy vs majority-class baseline on fresh tasks "
                    "from the trained prior. Beating majority is the floor; "
                    "majority_bce ≈ -log(0.5) = 0.693 for balanced priors."
                ),
            },
        )


def _roc_auc(probs: list[float], labels: list[int]) -> float:
    """Compute ROC AUC without sklearn. Returns 0.5 when only one class
    present (AUC undefined; surfaced as NaN by the caller via the
    ValueError it raises). Uses the rank-sum formulation which is O(N
    log N) and stable on small samples.
    """
    if not probs:
        raise ValueError("empty input")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("single-class labels — AUC undefined")
    # Pair, sort by predicted prob, compute Mann–Whitney U.
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and probs[order[j + 1]] == probs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)
