"""M4-monthly forecast scorer.

Loads the M4 competition's monthly training set, samples a small batch
of held-out series, holds out the last 12 months of each, builds the
same tabular feature representation TabPFN-TS's prior emits during
training (lagged y values + calendar position + normalised time), runs
the trained model on those feature vectors, and computes:

  - mse  — forecast MSE on the held-out 12-month tail
  - mase — Mean Absolute Scaled Error against a seasonal-naive
            baseline (predict y[t] = y[t-12])

The scorer is purposefully small (20 series sampled, 12-month horizon)
so it runs in a few seconds on CPU — it's a structural proof point,
not a paper-scale benchmark run.

Reference: Makridakis, Spiliotis, Assimakopoulos — The M4 Competition.
Int J Forecast 36, 2020.
"""

from __future__ import annotations

import math

from .base import DatasetScorer, ScorerResult

# Match the TabPFN-TS prior's feature shape so the model sees inputs of
# the same dimensionality it was trained on.
NUM_LAGS = 8
SEASONALITY = 12
FORECAST_HORIZON = 12
N_SERIES = 20  # small sample → fast eval on CPU
MIN_HISTORY = NUM_LAGS + 24  # need enough history for lags + a context window


class M4MonthlyForecast(DatasetScorer):
    """Forecast MSE on a held-out tail of M4 monthly series."""

    def score(self, *, model, eval_spec, loader, run_spec) -> ScorerResult:
        try:
            import numpy as np
            import pandas as pd  # noqa: F401 — used inside load_table
            import torch
        except ImportError as e:
            return ScorerResult(
                metrics={},
                meta={"dependency_missing": str(e)},
                skipped=True,
                skip_reason=f"missing dependency: {e}",
            )

        # Resolve the dataset reference from the eval spec. Caller (the
        # local adapter) has already confirmed the registry index lists
        # it, so failure here is genuinely unexpected.
        try:
            df = loader.load_table(eval_spec.dataset.source or "", split=eval_spec.dataset.split)
        except Exception as e:
            return ScorerResult(
                metrics={},
                meta={"load_error": str(e)},
                skipped=True,
                skip_reason=f"could not load dataset: {e}",
            )

        # M4 format: first column is series id, subsequent columns are
        # the series values, NaN-padded after the series ends. We drop
        # the id column and treat each row as a series.
        if df.shape[0] == 0:
            return ScorerResult(
                metrics={},
                meta={"reason": "empty M4 frame"},
                skipped=True,
                skip_reason="M4-monthly CSV had zero rows",
            )

        values_cols = [c for c in df.columns if c != df.columns[0]]
        # Pick series with enough non-NaN observations.
        eligible: list[np.ndarray] = []
        for _, row in df.iterrows():
            series = row[values_cols].to_numpy(dtype=np.float64)
            series = series[~np.isnan(series)]
            if len(series) >= MIN_HISTORY + FORECAST_HORIZON:
                eligible.append(series.astype(np.float32))
            if len(eligible) >= N_SERIES:
                break

        if len(eligible) == 0:
            return ScorerResult(
                metrics={},
                meta={"series_eligible": 0, "min_history_required": MIN_HISTORY + FORECAST_HORIZON},
                skipped=True,
                skip_reason="no M4 series met the minimum-length requirement",
            )

        # Per-series: hold out last FORECAST_HORIZON points, predict, score.
        model_squared_errs: list[float] = []
        naive_squared_errs: list[float] = []
        naive_abs_errs: list[float] = []
        model_abs_errs: list[float] = []

        device = next(
            (
                p.device
                for _, mod in getattr(model, "modules", [])
                if hasattr(mod, "parameters")
                for p in mod.parameters()
            ),
            None,
        )

        for series in eligible:
            # Z-normalise per series — matches the training preprocessing
            # and stabilises the model's feature distribution on real data.
            mean = float(series.mean())
            std = float(series.std()) or 1.0
            normed = (series - mean) / std

            t_history = len(normed) - FORECAST_HORIZON
            target = normed[t_history:]

            # Build the feature row used during training:
            #   [y_{t-1}, ..., y_{t-NUM_LAGS}, sin(2π·t/season), cos(2π·t/season), t/T]
            # Predict at each held-out timestep i = t_history + h for h in 0..FORECAST_HORIZON-1.
            # For each step, lags come from the *actual* preceding values
            # (teacher-forced; one-step-ahead). This is the same shape
            # the prior emits, so the model sees in-distribution inputs.
            T_total = len(normed)
            feature_rows: list[np.ndarray] = []
            for h in range(FORECAST_HORIZON):
                i = t_history + h
                lags = np.zeros(NUM_LAGS, dtype=np.float32)
                for k in range(1, NUM_LAGS + 1):
                    if i - k >= 0:
                        lags[k - 1] = normed[i - k]
                sin_s = math.sin(2.0 * math.pi * i / SEASONALITY)
                cos_s = math.cos(2.0 * math.pi * i / SEASONALITY)
                t_norm = i / float(T_total)
                feature_rows.append(
                    np.concatenate([lags, [sin_s, cos_s, t_norm]]).astype(np.float32)
                )

            X = np.stack(feature_rows)  # (FORECAST_HORIZON, NUM_LAGS + 3)
            x_t = torch.from_numpy(X).float().unsqueeze(0)  # (1, H, F)
            if device is not None:
                x_t = x_t.to(device)

            with torch.no_grad():
                out = x_t
                for _, mod in getattr(model, "modules", []):
                    out = mod(out)
                preds_normed = out.squeeze(0).squeeze(-1).cpu().numpy()
                if preds_normed.ndim == 0:
                    preds_normed = preds_normed.reshape(1)

            # Inverse-normalise back to original scale before scoring,
            # so the MSE is interpretable.
            preds = preds_normed.astype(np.float32) * std + mean
            target_orig = target.astype(np.float32) * std + mean

            # Seasonal-naive baseline: predict y[t] = y[t-SEASONALITY].
            naive_preds = np.zeros_like(target_orig)
            for h in range(FORECAST_HORIZON):
                i = t_history + h
                if i - SEASONALITY >= 0:
                    # Use the un-normalised history value at the same season offset.
                    naive_preds[h] = normed[i - SEASONALITY] * std + mean
                else:
                    naive_preds[h] = mean

            err_model = preds - target_orig
            err_naive = naive_preds - target_orig
            model_squared_errs.extend((err_model**2).tolist())
            naive_squared_errs.extend((err_naive**2).tolist())
            model_abs_errs.extend(np.abs(err_model).tolist())
            naive_abs_errs.extend(np.abs(err_naive).tolist())

        if not model_squared_errs:
            return ScorerResult(
                metrics={},
                meta={"reason": "no predictions produced"},
                skipped=True,
                skip_reason="scorer produced zero predictions",
            )

        mse = float(sum(model_squared_errs) / len(model_squared_errs))
        naive_mse = float(sum(naive_squared_errs) / len(naive_squared_errs))
        mae = float(sum(model_abs_errs) / len(model_abs_errs))
        naive_mae = float(sum(naive_abs_errs) / len(naive_abs_errs))

        # MASE: model MAE / seasonal-naive MAE. < 1 means we beat the naive baseline.
        mase = float(mae / naive_mae) if naive_mae > 0 else float("nan")

        return ScorerResult(
            metrics={"mse": mse, "mase": mase},
            meta={
                "series_scored": len(eligible),
                "forecast_horizon": FORECAST_HORIZON,
                "seasonal_naive_mse": naive_mse,
                "seasonal_naive_mae": naive_mae,
                "model_mae": mae,
            },
        )
