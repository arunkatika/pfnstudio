"""End-to-end smoke test for promptable training.

This is the *proof* that promptable priors work: a tag-aware brain
trained on the bayesian_linear prior with the monotonicity axis must
produce a non-trivial honoring score — meaning when we flip the
monotonicity tag at inference, the model's prediction visibly
changes.

The training run is intentionally small (300 steps) — we're not
proving best-in-class predictions, we're proving the *machinery*
end-to-end: sampler honors tag → trainer encodes tag → model uses
tag in attention → end-of-training honoring metric registers > 0.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from pfnstudio_core.model import BlockConfig, Model, ModelSpec, OutputHead
from pfnstudio_core.prior import Prior
from pfnstudio_core.run import EvalRef, ModelRef, PriorRef, RunSpec


class _PromptableBayesianLinear(Prior):
    """Mirror of the pfns-reference catalog prior with monotonicity
    axis support. Kept here so this test is self-contained and the
    catalog drift problem (test goes stale because templates.ts
    changed) is contained to a single file the test owner inspects."""

    spec = None
    axes = ["monotonicity"]

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 64,
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

        perm = rng.permutation(num_points)
        x_p, y_p = x[perm], y[perm]
        n_ctx = int(num_points * 0.75)
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
        return {"X": seq, "y": y_p[n_ctx:], "n_ctx": n_ctx, "a_true": a, "b_true": b}


@pytest.fixture(autouse=True)
def _ensure_blocks_registered():
    """Same registry-reload dance the closed-form test uses — other
    tests in the suite clear the block registry, so we have to repopulate."""
    import pfnstudio_core.blocks as _blocks_pkg
    from pfnstudio_core.blocks import heads, tabular, transformer
    from pfnstudio_core.registry import _BLOCKS

    _BLOCKS.clear()
    importlib.reload(heads)
    importlib.reload(tabular)
    importlib.reload(transformer)
    importlib.reload(_blocks_pkg)
    yield


def _build_tag_aware_model() -> Model:
    """Tiny tag-aware model: tabular embedder → tag-aware encoder →
    scalar head. ``tag_dim=4`` matches the monotonicity axis encoding
    (3 categorical values + 1 UNKNOWN slot)."""
    return Model(
        ModelSpec(
            id="promptable_pfn",
            name="Promptable PFN (test)",
            version="0.1.0",
            blocks=[
                BlockConfig(type="tabular_embedder", config={"d_model": 32}),
                BlockConfig(
                    type="tag_aware_transformer_encoder",
                    config={
                        "d_model": 32,
                        "n_heads": 4,
                        "n_layers": 2,
                        "dropout": 0.0,
                        "tag_dim": 4,
                    },
                ),
                BlockConfig(type="scalar_head", config={"d_model": 32, "d_out": 1}),
            ],
            output_heads=[OutputHead(name="pred_y", task="forecast")],
        )
    )


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="needs torch",
)
def test_promptable_training_produces_nontrivial_honoring(tmp_path, monkeypatch):
    """Train a tag-aware brain for a short run; verify the end-of-training
    honoring score is non-trivially > 0 (i.e. the model actually uses
    the tag rather than ignoring it)."""
    monkeypatch.chdir(tmp_path)  # checkpoint writes happen in cwd
    from pfnstudio_core.registry import _PRIORS
    from pfnstudio_core.training.loop import train_pfn

    _PRIORS["bayesian_linear"] = _PromptableBayesianLinear

    prior = _PromptableBayesianLinear()
    model = _build_tag_aware_model()
    run = RunSpec(
        id="promptable_e2e",
        prior=PriorRef(id="bayesian_linear", version="0.1.0"),
        model=ModelRef(id="promptable_pfn", version="0.1.0"),
        evals=[EvalRef(id="noop", version="0.1.0")],
        hyperparams={
            "steps": 300,
            "batch_size": 8,
            "lr": 1e-3,
            "seed": 11,
            "promptable_training": True,
        },
    )

    result = train_pfn(model, prior, run)
    assert result["status"] == "ok"
    assert "axis_honoring" in result, (
        "trainer didn't compute honoring even though promptable_training=True"
    )

    honoring = result["axis_honoring"]
    assert "monotonicity" in honoring
    score = honoring["monotonicity"]
    # Sanity: meaningful divergence between positive vs negative tag.
    # 0.05 is a low-but-real bar that 300 steps on a tiny model can
    # comfortably clear. A *broken* tag pathway (silently ignored)
    # would score < 1e-4.
    assert score["divergence_mean"] > 0.05, (
        f"monotonicity honoring = {score['divergence_mean']:.6f} — tag may be silently dropped"
    )
    assert score["values_compared"] == ("positive", "negative")
