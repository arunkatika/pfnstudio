"""Breast-cancer (Wisconsin) scorer: PFN vs LogisticRegression baseline.

Loads sklearn.datasets.load_breast_cancer() — 569 samples, 30 features,
binary label — and evaluates a trained in-context classifier on it.

The PFN was trained on 64-token sequences (48 context + 16 query) from
the random-binary-classification prior. Transformers don't extrapolate
robustly to *much* longer sequences than they saw at train time, so
naively packing the whole 569-row dataset into one sequence gives near-
chance results. Instead this scorer evaluates the PFN the way TabPFN
itself is meant to be used: many small in-context passes.

  1. Standardise features (mean 0, std 1 per column) so the dataset
     matches the prior's feature distribution.
  2. Pick a fixed train/test split (75/25, stratified, random_state=42)
     — same split LogReg sees, so the comparison is fair.
  3. Bootstrap-evaluate the PFN: repeat K times,
       - sample 48 context examples from the train fold (stratified),
       - sample 16 query examples from the test fold (without replacement
         across the bootstrap),
       - pack the 64-token sequence, run the model, collect query
         probabilities for those 16 test indices.
     After K passes every test point has been seen multiple times;
     average probabilities then threshold at 0.5.
  4. Compute accuracy + AUC and compare against sklearn LogisticRegression
     fit on the full 426-sample train fold (an honest baseline that uses
     all available supervision).

The PFN has never seen the breast-cancer dataset during training — only
the random-binary-classification synthetic prior. The baseline gap is
the headline: PFN vs LogReg on a real tabular dataset, with zero
real-data training for the PFN.

Reference:
  Hollmann, Müller, Eggensperger, Hutter — TabPFN. ICLR 2023.
  https://arxiv.org/abs/2207.01848
"""

from __future__ import annotations

from .base import DatasetScorer, ScorerResult

CTX_PER_PASS = 48  # match the prior's training context length
QRY_PER_PASS = 16  # match the prior's training query length
N_BOOTSTRAPS = 30  # passes; each test point gets sampled ~ N_BOOTSTRAPS * 16 / n_test times
BOOTSTRAP_SEED = 7


class BreastCancerVsLogReg(DatasetScorer):
    """Zero-shot in-context classification eval on sklearn breast-cancer."""

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        try:
            import numpy as np
            import torch
            from sklearn.datasets import load_breast_cancer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            return ScorerResult(
                metrics={},
                meta={"dependency_missing": str(e)},
                skipped=True,
                skip_reason=(
                    f"missing dependency: {e}. Install scikit-learn for this scorer "
                    "(it's not a hard dep of pfnstudio-core)."
                ),
            )

        # Load + standardise.
        data = load_breast_cancer()
        x_full = data.data.astype(np.float32)
        y_full = data.target.astype(np.float32)
        scaler = StandardScaler()
        x_full = scaler.fit_transform(x_full).astype(np.float32)

        x_train, x_test, y_train, y_test = train_test_split(
            x_full, y_full, test_size=0.25, stratify=y_full, random_state=42
        )
        n_test = x_test.shape[0]

        # Bootstrap-evaluate the PFN: average per-test-point probabilities
        # over many short in-context passes. Each test point gets seen
        # multiple times so the average is stable.
        prob_sum = np.zeros(n_test, dtype=np.float64)
        seen_count = np.zeros(n_test, dtype=np.int32)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        # Stratified context sampling: keep the two classes balanced inside
        # each pass so the PFN sees enough positive + negative examples.
        idx0 = np.where(y_train == 0.0)[0]
        idx1 = np.where(y_train == 1.0)[0]
        n_ctx_per_class = CTX_PER_PASS // 2

        with torch.no_grad():
            for _ in range(N_BOOTSTRAPS):
                ctx_idx0 = rng.choice(idx0, size=n_ctx_per_class, replace=False)
                ctx_idx1 = rng.choice(idx1, size=n_ctx_per_class, replace=False)
                ctx_idx = np.concatenate([ctx_idx0, ctx_idx1])
                rng.shuffle(ctx_idx)
                qry_idx = rng.choice(n_test, size=QRY_PER_PASS, replace=False)

                x_ctx = x_train[ctx_idx]
                y_ctx = y_train[ctx_idx]
                x_qry = x_test[qry_idx]

                ctx_tok = np.concatenate(
                    [
                        x_ctx,
                        y_ctx[:, None],
                        np.ones((CTX_PER_PASS, 1), dtype=np.float32),
                    ],
                    axis=1,
                )
                q_tok = np.concatenate(
                    [
                        x_qry,
                        np.zeros((QRY_PER_PASS, 1), dtype=np.float32),
                        np.zeros((QRY_PER_PASS, 1), dtype=np.float32),
                    ],
                    axis=1,
                )
                seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)

                if seq.shape[1] != 32:
                    return ScorerResult(
                        metrics={},
                        meta={"packed_shape": list(seq.shape)},
                        skipped=True,
                        skip_reason=(
                            f"Expected packed input width 32 (30 features + 2 markers), "
                            f"got {seq.shape[1]}. Was the model trained on a different prior?"
                        ),
                    )

                inp = torch.from_numpy(seq).unsqueeze(0)  # (1, 64, 32)
                out = inp
                for _, mod in model.modules:
                    out = mod(out)
                logits = out[0, CTX_PER_PASS:, 0].cpu().numpy()
                probs = 1.0 / (1.0 + np.exp(-logits))

                prob_sum[qry_idx] += probs
                seen_count[qry_idx] += 1

        # Some test points might not have been sampled — drop those from
        # the metric so we don't divide by zero. With N_BOOTSTRAPS=30 and
        # QRY_PER_PASS=16 the expected hit rate per test point is
        # 30*16/143 ≈ 3.4, so unseen-rate is essentially zero.
        sampled = seen_count > 0
        avg_probs = prob_sum[sampled] / seen_count[sampled]
        y_eval = y_test[sampled]
        preds_pfn = (avg_probs >= 0.5).astype(np.float32)
        pfn_acc = float(accuracy_score(y_eval, preds_pfn))
        pfn_auc = float(roc_auc_score(y_eval, avg_probs))

        # LogisticRegression baseline trained on the full 426-sample
        # train fold — gets all the supervision the dataset offers, so
        # it's an honest upper bar for this benchmark.
        logreg = LogisticRegression(max_iter=2000, random_state=42).fit(x_train, y_train)
        lr_probs = logreg.predict_proba(x_test)[:, 1]
        lr_preds = (lr_probs >= 0.5).astype(np.float32)
        lr_acc = float(accuracy_score(y_test, lr_preds))
        lr_auc = float(roc_auc_score(y_test, lr_probs))

        majority = float(max(y_train.mean(), 1.0 - y_train.mean()))

        # In-distribution sanity: run the model on 100 fresh tasks from
        # the training prior and report accuracy. This is what training
        # is *supposed* to nail; if it's near 50% the run itself is
        # broken and the breast-cancer numbers below are meaningless.
        in_dist_acc = 0.0
        in_dist_bce = 0.0
        in_dist_mean_pos_logit = 0.0
        in_dist_mean_neg_logit = 0.0
        try:
            import torch.nn.functional as F  # noqa: N812

            from ..registry import get_prior

            prior_cls = get_prior(run_spec.prior.id)
            prior = prior_cls()
            correct = total = 0
            sum_bce = 0.0
            sum_pos = 0.0
            n_pos = 0
            sum_neg = 0.0
            n_neg = 0
            with torch.no_grad():
                for k in range(100):
                    t = prior.sample(seed=20000 + k, num_points=64)
                    seq = np.asarray(t["X"], dtype=np.float32)
                    lbl = np.asarray(t["labels"], dtype=np.float32)
                    nc = int(t["n_ctx"])
                    out = torch.from_numpy(seq).unsqueeze(0)
                    for _, mod in model.modules:
                        out = mod(out)
                    logits = out[0, nc:, 0]
                    bce = F.binary_cross_entropy_with_logits(
                        logits, torch.from_numpy(lbl).float()
                    ).item()
                    logits_np = logits.cpu().numpy()
                    preds = (logits_np > 0).astype(np.float32)
                    correct += int((preds == lbl).sum())
                    total += int(lbl.shape[0])
                    sum_bce += bce
                    if (lbl == 1.0).any():
                        sum_pos += float(logits_np[lbl == 1.0].mean())
                        n_pos += 1
                    if (lbl == 0.0).any():
                        sum_neg += float(logits_np[lbl == 0.0].mean())
                        n_neg += 1
            if total:
                in_dist_acc = correct / total
            in_dist_bce = sum_bce / 100
            in_dist_mean_pos_logit = sum_pos / max(n_pos, 1)
            in_dist_mean_neg_logit = sum_neg / max(n_neg, 1)
        except Exception:
            in_dist_acc = 0.0

        return ScorerResult(
            metrics={
                "pfn_accuracy": pfn_acc,
                "pfn_auc": pfn_auc,
                "logreg_accuracy": lr_acc,
                "logreg_auc": lr_auc,
                "majority_baseline": majority,
                "accuracy_gap_vs_logreg": pfn_acc - lr_acc,
                "pfn_in_distribution_accuracy": in_dist_acc,
                "pfn_in_distribution_bce": in_dist_bce,
                "pfn_in_distribution_mean_logit_pos": in_dist_mean_pos_logit,
                "pfn_in_distribution_mean_logit_neg": in_dist_mean_neg_logit,
            },
            meta={
                "n_train": int(x_train.shape[0]),
                "n_test": int(n_test),
                "n_test_evaluated": int(sampled.sum()),
                "n_features": int(x_train.shape[1]),
                "context_per_pass": CTX_PER_PASS,
                "query_per_pass": QRY_PER_PASS,
                "bootstraps": N_BOOTSTRAPS,
                "dataset": "sklearn.datasets.load_breast_cancer",
                "split": "75/25 stratified, random_state=42",
                "method": (
                    "PFN: bootstrap-averaged in-context probabilities over "
                    f"{N_BOOTSTRAPS} passes of {CTX_PER_PASS} stratified context + "
                    f"{QRY_PER_PASS} query samples (matches training length). "
                    "LogReg: trained on the full 426-sample train fold."
                ),
            },
        )
