"""Phase 1 canary — verifies the existing trainer can handle packed-token
regression priors WITHOUT any loop.py changes.

The hypothesis we're testing: the trainer's regression branch at
loop.py:191-200 already supports `n_ctx` slicing and packed-token
input (the infrastructure was waiting for priors to use it). If that's
true, the foundational refactor is just prior code + predict.py + a
version bump — no training-loop surgery needed.

What we do:
  1. Define a packed-token Bayesian linear-regression prior inline.
     Each task: sample (a, b), generate (x, y), split into context +
     query, pack tokens as (x, y_or_zero, is_context_flag).
  2. Build a minimal transformer model spec (embedder → 2-layer
     encoder → scalar head) directly via the block registry.
  3. Run `train_pfn` for 400 steps with the existing trainer.
  4. Verify:
     a. Loss decreases (final < 0.5 × initial).
     b. The trained model, given a fresh task's context, predicts the
        query y's better than the context mean.
     c. Predictions track the true (a*x + b) line — i.e. the model
        is using the context, not just outputting an average.

If all three assertions pass, the trainer infrastructure is verified
and Phase 2 (refactor bayesian_linear in templates.ts + fix predict.py)
becomes a mechanical port — no architecture changes to the trainer.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from pfnstudio_core.model import BlockConfig, Model, ModelSpec, OutputHead
from pfnstudio_core.prior import Prior
from pfnstudio_core.run import ModelRef, PriorRef, RunSpec
from pfnstudio_core.training.loop import train_pfn


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


class PackedBayesianLinearPrior(Prior):
    """Bayesian linear regression in packed-token PFN format.

    Each task:
      - Sample (a, b) ~ N(0, 1)
      - Sample N points x ~ U(-1, 1), y = a*x + b + N(0, noise)
      - Split into n_ctx context + (N - n_ctx) query
      - Pack: (x, y_or_zero, is_context_flag) per token
      - Target: y at query positions only
    """

    spec = None  # tests don't need a PriorSpec; trainer doesn't look at it

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 32,
        noise: float = 0.05,
        **_: object,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal())
        b = float(rng.normal())
        x = rng.uniform(-1.0, 1.0, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, noise, size=num_points)).astype(np.float32)

        # Fixed 75% context split, matching pfns_classification. Varying
        # n_ctx across tasks would generalize better (Müller 2022 §3) but
        # the current trainer can't stack variable-length query tensors
        # within a batch — see Phase 1 finding in docs/refactor-regression-icl.md.
        # Phase 2 may revisit this with padding + masked loss.
        n_ctx = int(num_points * 0.75)
        n_qry = num_points - n_ctx

        # Shuffle so context isn't trivially the first half.
        perm = rng.permutation(num_points)
        x_p, y_p = x[perm], y[perm]

        # Pack tokens: (x, y_or_zero, is_context_flag). Token dim = 3.
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
            "X": seq,  # (num_points, 3) packed tokens
            "y": y_p[n_ctx:],  # (n_qry,) — query targets only
            "n_ctx": n_ctx,
            "a_true": a,
            "b_true": b,
        }


def _build_model() -> Model:
    """Tiny model: tabular_embedder → 2-layer encoder → scalar_head.

    TabularEmbedder uses LazyLinear, so it auto-infers the input dim
    (3 packed-token columns) on first forward pass.
    """
    return Model(
        ModelSpec(
            id="packed_canary",
            name="Packed-Token Canary",
            version="0.1.0",
            blocks=[
                BlockConfig(type="tabular_embedder", config={"d_model": 64}),
                BlockConfig(
                    type="transformer_encoder",
                    config={"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.0},
                ),
                BlockConfig(type="scalar_head", config={"d_model": 64, "d_out": 1}),
            ],
            output_heads=[OutputHead(name="pred_y", task="forecast")],
        )
    )


def _run_spec(steps: int) -> RunSpec:
    return RunSpec(
        id="canary",
        prior=PriorRef(id="packed_bayesian_linear", version="0.1.0"),
        model=ModelRef(id="packed_canary", version="0.1.0"),
        evals=[],
        hyperparams={"steps": steps, "batch_size": 16, "lr": 5e-4, "seed": 42},
    )


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="canary needs torch",
)
def test_packed_token_regression_trains_and_uses_context():
    """Phase 1 acceptance: trainer handles packed tokens without changes."""
    import torch

    prior = PackedBayesianLinearPrior()
    model = _build_model()
    run = _run_spec(steps=400)

    losses: list[float] = []

    def on_step(_step: int, loss: float) -> None:
        losses.append(loss)

    result = train_pfn(model, prior, run, on_step=on_step)

    # ── Assertion 1: training ran to completion (no silent skip). ────
    assert result.get("status") in (
        "ok",
        None,
    ), f"train_pfn skipped or failed: {result}"
    assert len(losses) == 400, f"expected 400 step losses, got {len(losses)}"

    # ── Assertion 2: loss decreased substantially. ────────────────────
    initial = float(np.mean(losses[:20]))
    final = float(np.mean(losses[-20:]))
    assert final < 0.5 * initial, (
        f"loss didn't drop enough — initial={initial:.4f}, final={final:.4f}"
    )

    # ── Assertion 3: the trained model uses context. ──────────────────
    # Generate a fresh task. Feed the packed sequence; check that
    # query-position predictions track (a_true * x + b_true) more
    # tightly than the context-mean baseline.
    fresh = prior.sample(seed=9999, num_points=32, noise=0.0)
    seq = torch.from_numpy(fresh["X"]).float().unsqueeze(0)  # (1, N, 3)

    encoder_blocks, head_blocks = [], []
    for _, mod in model.modules:
        cls_name = type(mod).__name__.lower()
        (head_blocks if cls_name.endswith("head") else encoder_blocks).append(mod)

    with torch.no_grad():
        x = seq
        for mod in encoder_blocks:
            x = mod(x)
        # Single scalar_head in this model spec.
        out = head_blocks[0](x)  # (1, N, 1)

    n_ctx = fresh["n_ctx"]
    query_preds = out[0, n_ctx:, 0].cpu().numpy()  # (n_qry,)
    y_query_true = fresh["y"]  # (n_qry,) — targets
    x_query = fresh["X"][n_ctx:, 0]  # x at query positions
    a_true = fresh["a_true"]
    b_true = fresh["b_true"]

    # Ground-truth predictions (zero-noise eval).
    y_query_gt = a_true * x_query + b_true

    # Context-mean baseline (what the model would predict if it
    # ignored context features). Since context y values were packed
    # into column 1 of the input, the baseline is the mean of those.
    ctx_y = fresh["X"][:n_ctx, 1]
    mean_pred = np.full_like(y_query_true, float(ctx_y.mean()))

    rmse_model = float(np.sqrt(np.mean((query_preds - y_query_gt) ** 2)))
    rmse_mean = float(np.sqrt(np.mean((mean_pred - y_query_gt) ** 2)))

    # The model should beat the mean baseline by a real margin. If the
    # model were ignoring context (the bug we're testing for), its
    # predictions would be roughly the prior's mean over y — comparable
    # to the context-mean baseline.
    assert rmse_model < rmse_mean * 0.7, (
        f"model RMSE {rmse_model:.4f} did not beat context-mean baseline "
        f"{rmse_mean:.4f} by 30% — model may not be using context"
    )
