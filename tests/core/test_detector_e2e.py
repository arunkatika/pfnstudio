"""End-to-end test for the auto-detector.

A trained AxisDetector reading context data from bayesian_linear with
a positive monotonicity tag should propose "positive" with high
confidence — and the same for "negative". This is the proof that the
detector pre-fill UX has real signal behind it: when the user uploads
their data, the chip will start in a sensible position instead of
forcing them to fill an empty form.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pfnstudio_core import AxisDetector, detect, get_axis, train_detector
from pfnstudio_core.prior import Prior


class _PromptableBayesianLinear(Prior):
    """Same shape as the catalog bayesian_linear, with monotonicity
    axis support. Mirrors templates.ts; trimmed of fields irrelevant
    to detector training (no n_ctx packing, just raw (X, y) pairs)."""

    spec = None
    axes = ["monotonicity"]

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 32,
        weight_std: float = 1.0,
        noise_scale: float = 0.1,
        x_range: float = 2.0,
        tag: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal(0.0, weight_std))
        b = float(rng.normal(0.0, weight_std))
        from pfnstudio_core import is_unknown

        mono = (tag or {}).get("monotonicity") if tag else None
        if mono == "positive":
            a = abs(a)
        elif mono == "negative":
            a = -abs(a)
        elif mono is None or is_unknown(mono) or mono == "mixed":
            pass
        else:
            raise ValueError(f"unsupported monotonicity: {mono!r}")

        x = rng.uniform(-x_range, x_range, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, noise_scale, size=num_points)).astype(np.float32)
        # Packed format: (x, y, 1.0) — detector reads observed pairs.
        # Always 1.0 for is_context_flag since detector sees full context.
        X = np.stack([x, y, np.ones_like(x)], axis=1).astype(np.float32)
        return {"X": X, "a_true": a, "y": y}


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="needs torch",
)
def test_detector_recovers_monotonicity_from_data():
    """Train detector for 800 steps; check that on positive-only data
    it proposes 'positive' with > random-baseline confidence, same for
    negative. Threshold is loose on purpose: this test asserts the
    detector *can* recover the sign, not that it's calibrated to a
    particular confidence — calibration is its own session's work."""
    import torch

    # Pin torch's global RNG so transformer init is reproducible —
    # the detector's confidence near the random-baseline regime is
    # sensitive enough that an unlucky init can flip the test.
    torch.manual_seed(0)

    monotonicity = get_axis("monotonicity")
    prior = _PromptableBayesianLinear()

    detector = AxisDetector(axes=[monotonicity], d_model=32, n_layers=2, n_heads=4)

    train_result = train_detector(
        detector=detector, prior=prior, steps=800, batch_size=16, lr=1e-3, seed=42
    )
    assert train_result["status"] == "ok"
    # Final-batch accuracy is a weak guarantee but catches obvious
    # broken-training cases (random init would sit near 1/3 = 0.33
    # for 3-class classification).
    assert train_result["last_batch_accuracy"]["monotonicity"] > 0.6, (
        f"detector training accuracy was {train_result['last_batch_accuracy']} "
        "— training may be broken"
    )

    # Now the real test: feed it positive-only data, expect "positive"
    # proposal with high confidence.
    pos_batch = prior.sample_batch(batch_size=8, seed=1000, tag={"monotonicity": "positive"})
    import torch

    pos_X = torch.stack([torch.from_numpy(b["X"]).float() for b in pos_batch])
    proposal_pos = detect(detector=detector, context=pos_X)

    assert "monotonicity" in proposal_pos
    pos_pred = proposal_pos["monotonicity"]
    assert pos_pred["value"] == "positive", (
        f"detector proposed {pos_pred['value']!r} on positive-only data "
        f"(probs: {pos_pred['probs']})"
    )
    # Beat the random-uniform 3-class baseline (0.333) by a real margin.
    # A *broken* detector would hover around 0.33 ± noise; we want to
    # see real signal but not pin a specific calibrated number.
    assert pos_pred["confidence"] > 0.45, (
        f"detector confidence on positive data was {pos_pred['confidence']:.3f}; "
        "should be > 0.45 (well above the 0.333 random baseline)"
    )

    # And the symmetric check on negative data.
    neg_batch = prior.sample_batch(batch_size=8, seed=2000, tag={"monotonicity": "negative"})
    neg_X = torch.stack([torch.from_numpy(b["X"]).float() for b in neg_batch])
    proposal_neg = detect(detector=detector, context=neg_X)
    neg_pred = proposal_neg["monotonicity"]
    assert neg_pred["value"] == "negative", (
        f"detector proposed {neg_pred['value']!r} on negative-only data "
        f"(probs: {neg_pred['probs']})"
    )
    assert neg_pred["confidence"] > 0.45


def test_detector_with_no_scored_axes_is_a_noop():
    """A detector built with only non-categorical axes (or no axes
    at all) should silently no-op rather than crash. This is what
    keeps the API safe to use even when no axes exist yet."""
    detector = AxisDetector(axes=[], d_model=8, n_layers=1, n_heads=1)
    assert detector.scored_axes == []
    train_result = train_detector(
        detector=detector,
        prior=_PromptableBayesianLinear(),
        steps=10,
    )
    assert train_result["status"] == "skipped"


def test_detect_returns_full_probability_dict():
    """The proposal payload includes ``probs`` so the UI can render
    uncertainty (e.g. '60% positive · 35% mixed · 5% negative')."""
    import torch

    monotonicity = get_axis("monotonicity")
    detector = AxisDetector(axes=[monotonicity], d_model=8, n_layers=1, n_heads=2)
    # Untrained — we don't care about correctness, just shape.
    x = torch.randn(2, 16, 3)
    proposal = detect(detector=detector, context=x)
    assert set(proposal["monotonicity"]["probs"]) == {"positive", "negative", "mixed"}
    total = sum(proposal["monotonicity"]["probs"].values())
    assert abs(total - 1.0) < 1e-5
