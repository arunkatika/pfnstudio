"""02 — Train a small classification PFN on the two-moons prior.

Companion to ``01_linear_regression.py``. Same training shape, different
target: instead of regressing a real-valued y, the model emits a binary
logit per point and we train against 0/1 labels.

Each task samples a fresh "two interlocking half-moons" geometry —
random rotation, random noise — and the PFN learns to classify any new
2-D point as belonging to moon A or moon B. After training, we compare
against two baselines:

  - **Majority class**  always predict whichever class has more support
                        in the context. A useless model bottoms out
                        here.
  - **Logistic regression** closed-form linear classifier fitted to the
                        context. A correct PFN should beat this when
                        the geometry is non-linear (which two-moons is).

Expected runtime: 3–6 minutes on a modern laptop CPU.

Requires: ``pip install pfnstudio-core[torch]``
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Model, ModelSpec, RunSpec
from pfnstudio_core.model import BlockConfig, OutputHead
from pfnstudio_core.prior import Prior, PriorSpec
from pfnstudio_core.registry import register_prior
from pfnstudio_core.run import ModelRef, PriorRef
from pfnstudio_core.training import train_pfn

# ── 1. The prior ──────────────────────────────────────────────────────────


@register_prior("two_moons_demo")
class TwoMoonsPrior(Prior):
    """Each task is a fresh random two-moons geometry with Gaussian noise."""

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 96,
        noise_scale: float = 0.15,
        radius: float = 1.0,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        n0 = num_points // 2
        n1 = num_points - n0

        # The two moons before any per-task transformation.
        t0 = rng.uniform(0, np.pi, size=n0)
        t1 = rng.uniform(0, np.pi, size=n1)
        moon0 = np.stack([radius * np.cos(t0), radius * np.sin(t0)], axis=-1)
        moon1 = np.stack(
            [radius - radius * np.cos(t1), -radius * np.sin(t1) + 0.5 * radius], axis=-1
        )

        # Random rotation per task — forces the PFN to discover the
        # geometry from context rather than memorising a fixed orientation.
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        R = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
            dtype=np.float32,
        )

        X = np.concatenate([moon0, moon1], axis=0).astype(np.float32) @ R.T
        X = X + rng.normal(0.0, noise_scale, size=X.shape).astype(np.float32)
        labels = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)

        # Shuffle so context vs query splits don't see class blocks in order.
        perm = rng.permutation(num_points)
        return {"X": X[perm], "labels": labels[perm]}


# ── 2. The model ──────────────────────────────────────────────────────────


def build_model() -> Model:
    """Two-layer tabular PFN with a single-logit head for binary classification.

    Output shape: (B, N, 1) — one logit per in-context point. The training
    loop's default step auto-detects ``labels`` in the batch and uses
    binary-cross-entropy-with-logits.
    """
    spec = ModelSpec(
        id="two_moons_demo_model",
        name="Two-moons demo PFN",
        version="0.1.0",
        description="Two-layer tabular PFN classifier for binary 2-D inputs.",
        blocks=[
            BlockConfig(type="tabular_embedder", config={"d_model": 32}),
            BlockConfig(
                type="transformer_encoder",
                config={
                    "d_model": 32,
                    "n_heads": 2,
                    "n_layers": 2,
                    "dropout": 0.0,
                },
            ),
            BlockConfig(type="scalar_head", config={"d_model": 32, "d_out": 1}),
        ],
        output_heads=[OutputHead(name="pred_logit", task="classification")],
    )
    return Model(spec)


# ── 3. Training ───────────────────────────────────────────────────────────


def train(steps: int = 500) -> tuple[Model, dict[str, Any]]:
    prior_spec = PriorSpec(
        id="two_moons_demo",
        name="Two moons (demo)",
        version="0.1.0",
        kind="tabular",
        parameters={
            "num_points": {"type": "int", "range": [60, 120]},
            "noise_scale": {"type": "float", "range": [0.05, 0.25]},
        },
        outputs={
            "variables": [
                {"name": "X", "type": "tensor", "shape": "(N, 2)"},
                {"name": "labels", "type": "tensor", "shape": "(N,)"},
            ]
        },
    )
    prior = TwoMoonsPrior()
    prior.spec = prior_spec

    run = RunSpec(
        id="two_moons_demo_run",
        prior=PriorRef(id="two_moons_demo", version="0.1.0"),
        model=ModelRef(id="two_moons_demo_model", version="0.1.0"),
        evals=[],
        hyperparams={
            "optimizer": "adamw",
            "lr": 5e-4,
            "batch_size": 16,
            "steps": steps,
            "seed": 42,
        },
    )

    model = build_model()
    return model, train_pfn(model=model, prior=prior, run=run)


# ── 4. Verification ───────────────────────────────────────────────────────


def evaluate(model: Model, num_tasks: int = 50, num_points: int = 96) -> dict[str, float]:
    """Held-out accuracy on fresh tasks. PFN vs majority + logistic baselines."""
    import torch

    prior = TwoMoonsPrior()

    pfn_correct = 0
    maj_correct = 0
    logreg_correct = 0
    total = 0

    for k in range(num_tasks):
        task = prior.sample(seed=10_000 + k, num_points=num_points, noise_scale=0.15)
        X = task["X"]
        y = task["labels"]

        # Split: last 16 points are the query set.
        n_query = 16
        X_query = X[-n_query:]
        y_query = y[-n_query:]
        X_ctx = X[:-n_query]
        y_ctx = y[:-n_query]

        # PFN — feed the query x through the model, threshold at 0.5.
        with torch.no_grad():
            inp = torch.from_numpy(X_query).float().unsqueeze(0)  # (1, n_query, 2)
            out = inp
            for _, mod in model.modules:
                out = mod(out)
            logits = out.squeeze(0).squeeze(-1).cpu().numpy()
            pfn_pred = (logits >= 0.0).astype(np.float32)  # logit ≥ 0 → P ≥ 0.5

        # Majority baseline.
        maj_class = float(round(y_ctx.mean()))
        maj_pred = np.full_like(y_query, maj_class)

        # Logistic-regression baseline (closed-form via numpy lstsq).
        logreg_pred = _logreg(X_ctx, y_ctx, X_query)

        pfn_correct += int(np.sum(pfn_pred == y_query))
        maj_correct += int(np.sum(maj_pred == y_query))
        logreg_correct += int(np.sum(logreg_pred == y_query))
        total += n_query

    return {
        "pfn_acc": pfn_correct / total,
        "maj_acc": maj_correct / total,
        "logreg_acc": logreg_correct / total,
        "tasks": num_tasks,
    }


def _logreg(X_ctx: np.ndarray, y_ctx: np.ndarray, X_q: np.ndarray) -> np.ndarray:
    """Lightweight one-shot logistic regression: a few IRLS-ish gradient steps."""
    # Add bias column.
    Xc = np.concatenate([X_ctx, np.ones((X_ctx.shape[0], 1))], axis=1).astype(np.float64)
    Xq = np.concatenate([X_q, np.ones((X_q.shape[0], 1))], axis=1).astype(np.float64)
    w = np.zeros(Xc.shape[1])
    for _ in range(50):
        z = Xc @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xc.T @ (p - y_ctx) / len(y_ctx)
        w -= 0.5 * grad
    logits = Xq @ w
    return (logits >= 0.0).astype(np.float32)


# ── 5. Glue ───────────────────────────────────────────────────────────────


def main() -> None:
    print("Training a 2-layer PFN classifier on the two-moons prior (500 steps)…")
    print()
    model, result = train(steps=500)
    print()
    print(
        f"Training: status={result.get('status')!r}  "
        f"final_loss={result.get('final_loss'):.4f}  "
        f"wall_time={result.get('wall_time_s'):.1f}s"
    )
    print()

    if result.get("status") != "ok":
        print(f"Training did not complete cleanly: {result.get('reason')}")
        return

    print("Evaluating on 50 fresh held-out tasks…")
    metrics = evaluate(model, num_tasks=50, num_points=96)
    print()
    print(f"  PFN accuracy            : {metrics['pfn_acc']:.3f}")
    print(
        f"  Majority baseline       : {metrics['maj_acc']:.3f}    "
        "(always predict the majority class — what an uninformed model does)"
    )
    print(
        f"  Logistic-regression     : {metrics['logreg_acc']:.3f}    "
        "(closed-form linear classifier — limited because two-moons is non-linear)"
    )
    print()

    if metrics["pfn_acc"] > metrics["maj_acc"]:
        print(
            f"✓ PFN beats the majority baseline by {metrics['pfn_acc'] - metrics['maj_acc']:.3f} "
            "absolute — it has learned to use the (x₁, x₂) coordinates."
        )
    else:
        print(
            "⚠ PFN does not beat the majority baseline — likely needs more training "
            "steps or a deeper model."
        )

    if metrics["pfn_acc"] > metrics["logreg_acc"]:
        print(
            f"✓ PFN beats logistic regression by {metrics['pfn_acc'] - metrics['logreg_acc']:.3f} "
            "absolute — encouraging signal that it's modelling the non-linear "
            "moon geometry rather than just a linear decision boundary."
        )
    else:
        print(
            f"~ PFN matches/ slightly under logistic regression "
            f"({metrics['pfn_acc']:.3f} vs {metrics['logreg_acc']:.3f}). At this scale "
            "and step count, picking up the non-linearity often takes a deeper model."
        )


if __name__ == "__main__":
    main()
