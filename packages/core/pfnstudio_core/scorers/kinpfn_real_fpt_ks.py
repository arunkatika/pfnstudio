"""KinPFN real-RNA FPT scorer — KS distance vs paper Table 7's 0.0632.

Loads the consolidated KinPFN test set (635 real RNA sequences × 1000
Kinfold first-passage-time simulations each, derived from
github.com/automl/KinPFN), then for each sampled sequence:

  1. Pick 100 random FPTs as observations (matches the paper's N=100
     context size, Table 7).
  2. Per-sequence normalize log(t) so values fall in the catalog's
     training range (the prior emits log_t in [-2, 6]). Real FPTs span
     up to ~1.7e8, so per-sequence normalization is necessary for the
     model to see in-distribution log_t at inference. The exact
     normalization is documented inline — it affects absolute KS
     numbers, so reproducibility comparisons against the paper are
     sensitive to it.
  3. Compute the self-referential context CDF F_ctx(t_(k)) = k/N at
     the sorted observation FPTs — the same shape the training prior
     emits (log_t, observed_cdf).
  4. Pack the context-query sequence (75 context + 25 query, matching
     the catalog's training token budget) and run the model.
  5. Compute the empirical CDF over ALL 1000 simulations (the much
     better ground-truth estimate). KS = max |F_pred − F_emp_full|
     over the query positions.

The catalog's training num_points is 100 (75 ctx + 25 query). Real
sequences have 1000 FPTs each, so we sub-sample 100 for the in-context
pass. Averaging over many sequences gives a stable mean-KS estimate.

Paper claim (KinPFN Table 7, N=100 context, real-world FPT test set):
KinPFN KS = 0.0632. Trained brains should land in that vicinity if
the catalog's prior + training + inference path actually reproduce
the paper.

Reference: Scheuer, Runge et al. — *KinPFN: A Foundation Model for
RNA Kinetics*. ICLR 2025.
"""

from __future__ import annotations

from .base import DatasetScorer, ScorerResult

# Catalog training format: 100 tokens, 75 context, 25 query.
NUM_TOKENS = 100
N_CTX_PER_PASS = 75
N_QRY_PER_PASS = 25
# Cap the number of sequences scored to keep wall-clock manageable.
# 635 sequences × ~10ms/forward ≈ 6s end-to-end — fine on CPU.
N_SEQUENCES = 635
BOOTSTRAP_SEED = 7
# Target training range for log_t. The catalog prior uses [-2, 6] for
# log_time_min / log_time_max. Per-sequence affine normalisation maps
# the observed FPTs into roughly this range, with the median FPT at
# the centre of the range.
TRAIN_LOG_MIN = -2.0
TRAIN_LOG_MAX = 6.0


class KinPFNRealFPTKS(DatasetScorer):
    """KS distance on real RNA FPT distributions vs paper Table 7."""

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

        try:
            df = loader.load_table(eval_spec.dataset.source or "", split=eval_spec.dataset.split)
        except Exception as e:
            return ScorerResult(
                metrics={},
                meta={"load_error": str(e)},
                skipped=True,
                skip_reason=(
                    f"could not load kinpfn-real-fpt parquet: {e}. "
                    "Download it from the registry first: it's the consolidated "
                    "KinPFN paper test set, ~3 MB."
                ),
            )

        if "fpt" not in df.columns or "sequence_id" not in df.columns:
            return ScorerResult(
                metrics={},
                meta={"columns": list(df.columns)},
                skipped=True,
                skip_reason=(
                    "Expected columns 'sequence_id' and 'fpt' in the kinpfn-real-fpt "
                    "parquet. Did the dataset schema change?"
                ),
            )

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        seq_ids = df["sequence_id"].unique()
        rng.shuffle(seq_ids)
        seq_ids = seq_ids[:N_SEQUENCES]

        per_seq_ks: list[float] = []
        sequences_skipped = 0

        with torch.no_grad():
            for seq_id in seq_ids:
                fpts = df.loc[df["sequence_id"] == seq_id, "fpt"].to_numpy(dtype=np.float64)
                # Drop zeros / non-positives (log undefined).
                fpts = fpts[fpts > 0.0]
                if len(fpts) < NUM_TOKENS + 50:
                    sequences_skipped += 1
                    continue

                # Sub-sample 100 observations for the in-context pass.
                idx = rng.choice(len(fpts), size=NUM_TOKENS, replace=False)
                obs = fpts[idx]
                # Sort for empirical CDF computation: rank-i of NUM_TOKENS
                # corresponds to empirical CDF = i/NUM_TOKENS.
                order = np.argsort(obs)
                obs_sorted = obs[order]
                # Per-sequence log-space normalisation. Centre the median
                # observed FPT at the middle of the training log range,
                # and scale so the inter-quartile range fits comfortably.
                # Per-sequence normalisation is necessary because real
                # FPTs span huge ranges (1e-2 to 1e8) while the catalog
                # prior emits log_t in [-2, 6].
                log_obs_sorted = np.log(obs_sorted)
                centre = float(np.median(log_obs_sorted))
                spread = (
                    float(np.percentile(log_obs_sorted, 75) - np.percentile(log_obs_sorted, 25))
                    + 1e-6
                )
                target_centre = (TRAIN_LOG_MIN + TRAIN_LOG_MAX) / 2.0
                target_half_iqr = (TRAIN_LOG_MAX - TRAIN_LOG_MIN) / 4.0
                log_obs_norm_sorted = (log_obs_sorted - centre) / spread * (
                    2.0 * target_half_iqr
                ) + target_centre
                log_obs_norm_sorted = np.clip(
                    log_obs_norm_sorted, TRAIN_LOG_MIN, TRAIN_LOG_MAX
                ).astype(np.float32)

                # Empirical CDF at each sorted observation, computed over
                # ALL NUM_TOKENS — so context CDF values can span the full
                # [0, 1] range (matching what the model saw in training,
                # where context CDFs came from an analytic Weibull mixture).
                cdf_emp_sorted = (np.arange(1, NUM_TOKENS + 1) / float(NUM_TOKENS)).astype(
                    np.float32
                )

                # Randomly assign 75 of the 100 to context, 25 to query.
                # Shuffling means context covers the FULL CDF range
                # (not just the lower tail) — necessary so the model
                # sees the in-distribution context shape it trained on.
                token_perm = rng.permutation(NUM_TOKENS)
                ctx_idx = token_perm[:N_CTX_PER_PASS]
                qry_idx = token_perm[N_CTX_PER_PASS:]
                log_ctx = log_obs_norm_sorted[ctx_idx]
                cdf_ctx = cdf_emp_sorted[ctx_idx]
                log_qry = log_obs_norm_sorted[qry_idx]

                ctx_tok = np.stack(
                    [log_ctx, cdf_ctx, np.ones(N_CTX_PER_PASS, dtype=np.float32)], axis=1
                )
                q_tok = np.stack(
                    [
                        log_qry,
                        np.zeros(N_QRY_PER_PASS, dtype=np.float32),
                        np.zeros(N_QRY_PER_PASS, dtype=np.float32),
                    ],
                    axis=1,
                )
                seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)

                inp = torch.from_numpy(seq).unsqueeze(0)
                out = inp
                for _, mod in model.modules:
                    out = mod(out)
                # Predicted CDF at query positions (the 25 tokens we
                # masked out during packing).
                f_pred = out[0, N_CTX_PER_PASS:, 0].cpu().numpy()
                f_pred = np.clip(f_pred, 0.0, 1.0)

                # Ground-truth empirical CDF at query t values using ALL
                # available simulations (better estimate than the
                # 100-sample CDF). The query points are obs_sorted[qry_idx];
                # their rank in the full distribution is searchsorted.
                fpts_sorted = np.sort(fpts)
                obs_query = obs_sorted[qry_idx]
                f_emp_full = np.searchsorted(fpts_sorted, obs_query, side="right") / float(
                    len(fpts_sorted)
                )

                ks = float(np.max(np.abs(f_pred - f_emp_full.astype(np.float32))))
                per_seq_ks.append(ks)

        if not per_seq_ks:
            return ScorerResult(
                metrics={},
                meta={"sequences_skipped": sequences_skipped},
                skipped=True,
                skip_reason="no sequences had enough FPTs to score.",
            )

        ks_arr = np.asarray(per_seq_ks, dtype=np.float32)
        return ScorerResult(
            metrics={
                "ks_distance": float(ks_arr.mean()),
                "ks_distance_median": float(np.median(ks_arr)),
                "ks_distance_p95": float(np.percentile(ks_arr, 95)),
            },
            meta={
                "sequences_scored": len(per_seq_ks),
                "sequences_skipped": sequences_skipped,
                "context_size": N_CTX_PER_PASS,
                "query_size": N_QRY_PER_PASS,
                "paper_claim_table7_n100": 0.0632,
                "note": (
                    f"Mean KS over {len(per_seq_ks)} real-RNA FPT distributions. "
                    "Paper claim (KinPFN Table 7, N=100 context, real-world FPTs): "
                    "KS = 0.0632. Per-sequence log-space normalisation mapped each "
                    "sequence's observed FPTs into the catalog's training log_t range "
                    "[-2, 6] before in-context inference."
                ),
            },
        )
