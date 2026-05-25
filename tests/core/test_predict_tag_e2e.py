"""Inference-side proof of the chip-toggle UX.

Train a tag-aware brain, run the predict path twice with the *same*
context but different tag values — assert the predictions differ.
This is the train-end honoring metric's mirror at inference time:
the metric guarantees the trained brain *can* honor the tag; this
test guarantees the predict path actually *forwards* the tag to the
model rather than silently dropping it on the floor.

If the chip UI eventually breaks, the failure surface is here.
"""

from __future__ import annotations

import importlib
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _ensure_blocks_registered():
    """Same registry reload dance as the other e2e tests — pytest
    runs after other suites that clear the block registry."""
    import pfnstudio_core.blocks as _blocks_pkg
    from pfnstudio_core.blocks import heads, tabular, transformer
    from pfnstudio_core.registry import _BLOCKS

    _BLOCKS.clear()
    importlib.reload(heads)
    importlib.reload(tabular)
    importlib.reload(transformer)
    importlib.reload(_blocks_pkg)
    yield


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="needs torch",
)
def test_predict_with_tag_moves_predictions():
    """Train a tiny tag-aware brain, then forward identical context
    twice with different monotonicity tags. Predictions must differ."""
    import torch
    from pfnstudio_core import encode_tag, get_axis
    from pfnstudio_core.model import BlockConfig, Model, ModelSpec, OutputHead
    from pfnstudio_core.prior import Prior
    from pfnstudio_core.registry import _PRIORS
    from pfnstudio_core.run import EvalRef, ModelRef, PriorRef, RunSpec
    from pfnstudio_core.training.loop import train_pfn
    from pfnstudio_core.training.predict import _forward_full_sequence

    # Pin torch's global RNG so the transformer's init + training
    # are reproducible across test orderings. The train_pfn loop sets
    # its own seed from RunSpec hyperparams, but block constructors
    # fire before that, and a stale RNG state from a prior test can
    # change init enough to push the divergence below threshold.
    torch.manual_seed(0)

    # Inline prior — same shape as the e2e brain test in session 5.
    class _PromptablePrior(Prior):
        spec = None
        axes = ["monotonicity"]

        def sample(self, *, seed, num_points=48, tag=None, **_):
            rng = np.random.default_rng(seed)
            a = float(rng.normal(0.0, 1.0))
            b = float(rng.normal(0.0, 1.0))
            from pfnstudio_core import is_unknown

            mono = (tag or {}).get("monotonicity") if tag else None
            if mono == "positive":
                a = abs(a)
            elif mono == "negative":
                a = -abs(a)
            elif mono is None or is_unknown(mono) or mono == "mixed":
                pass
            else:
                raise ValueError(f"unsupported: {mono!r}")

            x = rng.uniform(-2.0, 2.0, size=num_points).astype(np.float32)
            y = (a * x + b + rng.normal(0.0, 0.1, size=num_points)).astype(np.float32)
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
            return {"X": seq, "y": y_p[n_ctx:], "n_ctx": n_ctx}

    _PRIORS["promptable_inline"] = _PromptablePrior

    model = Model(
        ModelSpec(
            id="promptable_inline_model",
            name="Promptable inline",
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

    run = RunSpec(
        id="predict_e2e",
        prior=PriorRef(id="promptable_inline", version="0.1.0"),
        model=ModelRef(id="promptable_inline_model", version="0.1.0"),
        evals=[EvalRef(id="noop", version="0.1.0")],
        hyperparams={
            "steps": 1000,
            "batch_size": 8,
            "lr": 1e-3,
            "seed": 13,
            "promptable_training": True,
        },
    )

    # Train inside a tmp cwd so checkpoint writes don't pollute the repo.
    with tempfile.TemporaryDirectory() as td:
        cwd_before = os.getcwd()
        os.chdir(td)
        try:
            result = train_pfn(model, _PromptablePrior(), run)
            assert result["status"] == "ok"
        finally:
            os.chdir(cwd_before)

    # Build a synthetic context + query sequence. Same data both times.
    rng = np.random.default_rng(99)
    n_ctx, n_qry = 24, 8
    x_ctx = rng.uniform(-2.0, 2.0, size=n_ctx).astype(np.float32)
    y_ctx = (0.5 * x_ctx + rng.normal(0.0, 0.1, size=n_ctx)).astype(np.float32)
    x_qry = rng.uniform(-2.0, 2.0, size=n_qry).astype(np.float32)
    ctx_tok = np.stack([x_ctx, y_ctx, np.ones(n_ctx, dtype=np.float32)], axis=1)
    q_tok = np.stack(
        [x_qry, np.zeros(n_qry, dtype=np.float32), np.zeros(n_qry, dtype=np.float32)],
        axis=1,
    )
    seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)
    inp = torch.from_numpy(seq).float().unsqueeze(0)  # (1, N, 3)

    axes = [get_axis("monotonicity")]
    tag_pos = torch.from_numpy(encode_tag({"monotonicity": "positive"}, axes)).float().unsqueeze(0)
    tag_neg = torch.from_numpy(encode_tag({"monotonicity": "negative"}, axes)).float().unsqueeze(0)

    with torch.no_grad():
        out_pos = _forward_full_sequence(model, inp, tag_tensor=tag_pos)
        out_neg = _forward_full_sequence(model, inp, tag_tensor=tag_neg)

    preds_pos = out_pos[0, n_ctx:, 0].cpu().numpy()
    preds_neg = out_neg[0, n_ctx:, 0].cpu().numpy()
    divergence = float(np.linalg.norm(preds_pos - preds_neg))

    # The question this test answers is binary: did the tag reach
    # the model at all? A *broken* predict-tag pathway (tag dropped
    # on the floor before forward) would score literally 0 because
    # _forward_full_sequence would be deterministic given the same
    # input. Anything meaningfully > 0 proves the dispatch routed
    # the tag to the encoder. 1e-3 sits well above floating-point
    # noise but well below the threshold the model's small size
    # makes brittle.
    assert divergence > 1e-3, (
        f"Predict path returned identical predictions for opposite "
        f"tags (L2={divergence:.5f}). Tag tensor isn't reaching the model."
    )
