"""Phase 2 acceptance — pfns-reference matches the closed-form Bayesian
posterior after training.

This is the canary the regression-ICL refactor is built to pass. If a
PFN trained on bayesian_linear with packed-token ICL can match the
analytic Bayesian posterior mean within tolerance, that's the canonical
Müller 2022 verdict — and proves the new architecture works end-to-end.

What we do:
  1. Construct the catalog's bayesian_linear@0.2.0 prior directly from
     a thin wrapper around the packed-token implementation. The Python
     code in templates.ts is replicated here verbatim so the test
     covers the *catalog's* prior, not a separate inline copy.
  2. Train the catalog's pfn_transformer@0.2.0 model spec on it for
     1500 steps (small enough to run in CI on CPU, large enough that
     the network actually learns ICL).
  3. Run the ClosedFormBaseline scorer end-to-end.
  4. Assert ``rmse_vs_posterior`` < tolerance.

A passing canary means:
  - The trainer's regression branch handles packed-token batches.
  - The predict path produces correct shapes.
  - The scorer registry routes the eval correctly.
  - The PFN actually learned in-context Bayesian inference.

Tolerance: 0.15 RMSE against the analytic posterior on a small CPU
train. Larger paper-pinned runs (16k steps, the catalog default)
should drive this much lower.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pfnstudio_core.model import BlockConfig, Model, ModelSpec, OutputHead
from pfnstudio_core.prior import Prior
from pfnstudio_core.run import EvalRef, ModelRef, PriorRef, RunSpec
from pfnstudio_core.scorers import BUILTIN_SCORERS
from pfnstudio_core.training.loop import train_pfn

# pfns-reference catalog hyperparams: weight_std=1.0, noise_scale=0.1,
# packed-token format with fixed 75% n_ctx. Matches templates.ts
# PFNS_BAYESIAN_LINEAR_PY at version 0.2.0.
_CONTEXT_FRACTION = 0.75
_NUM_FEATURES = 1


def _apply_monotonicity(a: float, tag: dict[str, Any] | None) -> float:
    """Mirrors templates.ts ``_apply_monotonicity``. See docstring there."""
    from pfnstudio_core import is_unknown

    mono = (tag or {}).get("monotonicity") if tag else None
    if mono is None or is_unknown(mono) or mono == "mixed":
        return a
    if mono == "positive":
        return abs(a)
    if mono == "negative":
        return -abs(a)
    raise ValueError(f"bayesian_linear: unsupported monotonicity value {mono!r}")


class _CatalogBayesianLinearPrior(Prior):
    """Local copy of the catalog's bayesian_linear @ 0.2.0 prior code.

    Kept verbatim with templates.ts so this test covers the actual
    catalog prior shape rather than a parallel implementation. If the
    catalog prior changes, mirror the change here.
    """

    spec = None
    axes = ["monotonicity"]

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 100,
        weight_std: float = 1.0,
        noise_scale: float = 0.1,
        x_range: float = 2.0,
        tag: dict[str, Any] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal(0.0, weight_std))
        b = float(rng.normal(0.0, weight_std))
        a = _apply_monotonicity(a, tag)
        x = rng.uniform(-x_range, x_range, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, noise_scale, size=num_points)).astype(np.float32)

        perm = rng.permutation(num_points)
        x_p, y_p = x[perm], y[perm]

        n_ctx = int(num_points * _CONTEXT_FRACTION)
        n_qry = num_points - n_ctx

        ctx_tok = np.stack([x_p[:n_ctx], y_p[:n_ctx], np.ones(n_ctx, dtype=np.float32)], axis=1)
        q_tok = np.stack(
            [
                x_p[n_ctx:],
                np.zeros(n_qry, dtype=np.float32),
                np.zeros(n_qry, dtype=np.float32),
            ],
            axis=1,
        )
        seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)

        return {
            "X": seq,
            "y": y_p[n_ctx:],
            "n_ctx": n_ctx,
            "a_true": a,
            "b_true": b,
        }


@pytest.fixture(autouse=True)
def _ensure_blocks_registered():
    """Make sure built-in blocks are in the registry before this test runs.

    Other tests call ``_clear_for_tests()``, which wipes the registry;
    decorator-based registration only fires on first import. Reloading
    re-runs the decorators — but if the registry is *already* populated
    the redecorate raises 'already registered'. Clear-then-reload makes
    this idempotent regardless of suite order.
    """
    import pfnstudio_core.blocks as _blocks_pkg
    from pfnstudio_core.blocks import heads, tabular, transformer
    from pfnstudio_core.registry import _BLOCKS  # type: ignore[attr-defined]

    _BLOCKS.clear()
    importlib.reload(heads)
    importlib.reload(tabular)
    importlib.reload(transformer)
    importlib.reload(_blocks_pkg)
    yield


def _build_catalog_model() -> Model:
    """Catalog's pfn_transformer @ 0.2.0 — 4-layer encoder, d_model=128.

    Matches templates.ts PFNS_TEMPLATE seedModels[0] exactly.
    """
    return Model(
        ModelSpec(
            id="pfn_transformer",
            name="PFN Transformer (4-layer)",
            version="0.2.0",
            blocks=[
                BlockConfig(type="tabular_embedder", config={"d_model": 128}),
                BlockConfig(
                    type="transformer_encoder",
                    config={"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.0},
                ),
                BlockConfig(type="scalar_head", config={"d_model": 128, "d_out": 1}),
            ],
            output_heads=[OutputHead(name="pred_y", task="forecast")],
        )
    )


def _register_prior_globally(prior_cls: type) -> None:
    """Register the prior in the global registry so the scorer can resolve it.

    The scorer calls ``get_prior(run_spec.prior.id)``, which goes through
    the registry. For the test we want the catalog-shaped prior class
    bound to the slug ``bayesian_linear``.
    """
    from pfnstudio_core.registry import _PRIORS  # type: ignore[attr-defined]

    _PRIORS["bayesian_linear"] = prior_cls


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="needs torch",
)
def test_pfns_reference_matches_closed_form_posterior():
    """Train pfns-reference @ 0.2.0; scorer's rmse_vs_posterior must be small."""
    _register_prior_globally(_CatalogBayesianLinearPrior)

    prior = _CatalogBayesianLinearPrior()
    model = _build_catalog_model()
    run = RunSpec(
        id="closed_form_canary",
        prior=PriorRef(
            id="bayesian_linear",
            version="0.2.0",
            overrides={"num_points": 100, "weight_std": 1.0, "noise_scale": 0.1},
        ),
        model=ModelRef(id="pfn_transformer", version="0.2.0"),
        evals=[EvalRef(id="closed_form_baseline", version="0.1.0")],
        hyperparams={"steps": 1500, "batch_size": 16, "lr": 5e-4, "seed": 42},
    )

    result = train_pfn(model, prior, run)
    assert result.get("status") in (
        "ok",
        None,
    ), f"train_pfn failed: {result}"

    # Run the closed-form scorer end-to-end. EvalSpec / loader are
    # not used by ClosedFormBaseline, but we pass plausible stand-ins.
    scorer = BUILTIN_SCORERS["closed_form_baseline"]
    eval_spec = SimpleNamespace(
        id="closed_form_baseline",
        dataset=SimpleNamespace(source=None, split=None),
    )
    score = scorer.score(model=model, eval_spec=eval_spec, loader=None, run_spec=run)

    assert not score.skipped, f"scorer skipped: {score.skip_reason}"

    rmse = score.metrics["rmse"]
    rmse_vs_posterior = score.metrics["rmse_vs_posterior"]

    # The verdict: a correctly trained PFN should match the analytic
    # posterior. 0.15 is a generous tolerance for a 1500-step CPU run;
    # paper-pinned runs (16k+ steps) should drive this to ~0.02.
    assert rmse_vs_posterior < 0.15, (
        f"rmse_vs_posterior = {rmse_vs_posterior:.4f} > 0.15 "
        f"— PFN didn't match the analytic posterior. "
        f"(rmse vs latent y_q = {rmse:.4f})"
    )


# ---------------------------------------------------------------------------
# Promptable-axis back-compat for the pfns-reference prior. These don't
# require a training run — they verify the prior's sampler honors the
# monotonicity tag and stays byte-identical when tagged UNKNOWN.
# ---------------------------------------------------------------------------


def test_pfns_reference_unknown_tag_byte_identical():
    """tag=UNKNOWN (or missing/empty/None) must be bit-identical to no
    tag. This is what guarantees adding the axis can't regress the
    closed-form benchmark above."""
    from pfnstudio_core import UNKNOWN

    p = _CatalogBayesianLinearPrior()
    base = p.sample(seed=1234, num_points=64)
    for label, t in (
        ("UNKNOWN", {"monotonicity": UNKNOWN}),
        ("empty {}", {}),
        ("None", None),
    ):
        out = p.sample(seed=1234, num_points=64, tag=t)
        assert np.array_equal(base["X"], out["X"]), f"X drift with {label}"
        assert np.array_equal(base["y"], out["y"]), f"y drift with {label}"
        assert base["a_true"] == out["a_true"], f"a_true drift with {label}"


def test_pfns_reference_monotonicity_clamps_slope_sign():
    """Tag values that fix monotonicity must clamp the slope sign
    accordingly, while preserving the magnitude distribution."""
    p = _CatalogBayesianLinearPrior()
    for seed in range(20):
        pos = p.sample(seed=seed, num_points=8, tag={"monotonicity": "positive"})
        neg = p.sample(seed=seed, num_points=8, tag={"monotonicity": "negative"})
        unk = p.sample(seed=seed, num_points=8)
        assert pos["a_true"] >= 0, f"positive tag let a={pos['a_true']} through"
        assert neg["a_true"] <= 0, f"negative tag let a={neg['a_true']} through"
        # Magnitude preserved — sign clamp only flips sign, never re-samples.
        assert abs(pos["a_true"]) == pytest.approx(abs(unk["a_true"]))
        assert abs(neg["a_true"]) == pytest.approx(abs(unk["a_true"]))
