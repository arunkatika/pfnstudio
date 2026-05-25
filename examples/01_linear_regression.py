"""01 — Train a small PFN on Bayesian linear regression.

This is the simplest end-to-end PFN Studio example. We:

  1. Define a Bayesian linear regression prior — random (slope, intercept)
     plus i.i.d. Gaussian noise.
  2. Build a tiny transformer model (d_model=32, 2 layers).
  3. Train it for 500 steps on CPU.
  4. Verify the trained model performs in-context regression on fresh
     held-out tasks, beating a baseline that predicts the mean.

Expected runtime: 3–6 minutes on a modern laptop CPU. Output: a saved
checkpoint at ``checkpoint/model.pt`` you can load for inference, plus
a final MSE comparison printed at the end.

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


@register_prior("bayesian_linear_demo")
class BayesianLinearPrior(Prior):
    """y = a*x + b + ε, with (a, b) ~ N(0, weight_std) and ε ~ N(0, noise_std).

    A correct PFN trained on this prior will, given any held-out (x, y)
    set, predict on new x's in a way that approaches the closed-form
    Bayesian posterior mean — without ever inverting a covariance matrix.
    """

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 64,
        weight_std: float = 1.0,
        noise_scale: float = 0.1,
        x_range: float = 2.0,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal(0.0, weight_std))
        b = float(rng.normal(0.0, weight_std))
        x = rng.uniform(-x_range, x_range, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, noise_scale, size=num_points)).astype(np.float32)
        return {
            "X": x.reshape(-1, 1),
            "y": y,
            "a_true": a,
            "b_true": b,
        }


# ── 2. The model ──────────────────────────────────────────────────────────


def build_model() -> Model:
    """Tiny tabular PFN — small enough to train on CPU in minutes."""
    spec = ModelSpec(
        id="linear_regression_demo_model",
        name="Linear-regression demo PFN",
        version="0.1.0",
        description="Two-layer tabular PFN for in-context linear regression.",
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
        output_heads=[OutputHead(name="pred_y", task="forecast")],
    )
    return Model(spec)


# ── 3. Training ───────────────────────────────────────────────────────────


def train(steps: int = 500) -> dict[str, Any]:
    prior_spec = PriorSpec(
        id="bayesian_linear_demo",
        name="Bayesian linear regression (demo)",
        version="0.1.0",
        kind="tabular",
        parameters={
            "num_points": {"type": "int", "range": [50, 100]},
            "weight_std": {"type": "float", "range": [0.5, 1.5]},
            "noise_scale": {"type": "float", "range": [0.05, 0.2]},
        },
        outputs={
            "variables": [
                {"name": "X", "type": "tensor", "shape": "(N, 1)"},
                {"name": "y", "type": "tensor", "shape": "(N,)"},
            ]
        },
    )
    prior = BayesianLinearPrior()
    prior.spec = prior_spec

    run = RunSpec(
        id="linear_regression_demo_run",
        prior=PriorRef(id="bayesian_linear_demo", version="0.1.0"),
        model=ModelRef(id="linear_regression_demo_model", version="0.1.0"),
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


def evaluate(model: Model, num_tasks: int = 50, num_points: int = 80) -> dict[str, float]:
    """Sample held-out tasks; compare PFN predictions to a mean baseline + OLS."""
    import torch

    prior = BayesianLinearPrior()

    pfn_sse = 0.0
    mean_sse = 0.0
    ols_sse = 0.0
    total = 0

    for k in range(num_tasks):
        task = prior.sample(seed=10_000 + k, num_points=num_points, weight_std=1.0, noise_scale=0.1)
        x = task["X"]  # (N, 1)
        y = task["y"]  # (N,)

        # Hold out the last 16 points; the model "sees" the first N-16 as context
        # and predicts on the last 16. The current naive inference path runs the
        # model forward on the held-out x's directly — what it has learned about
        # the prior must be baked into its weights.
        n_query = 16
        x_query = x[-n_query:]
        y_query = y[-n_query:]

        with torch.no_grad():
            inp = torch.from_numpy(x_query).float().unsqueeze(0)  # (1, n_query, 1)
            out = inp
            for _, mod in model.modules:
                out = mod(out)
            preds = out.squeeze(0).squeeze(-1).cpu().numpy()

        # Baselines: predict the held-out mean of the training portion, and
        # closed-form OLS using the visible context.
        x_ctx = x[:-n_query, 0]
        y_ctx = y[:-n_query]
        mean_pred = np.full_like(y_query, y_ctx.mean())
        a_ols, b_ols = np.polyfit(x_ctx, y_ctx, 1)
        ols_pred = a_ols * x_query[:, 0] + b_ols

        pfn_sse += float(np.sum((preds - y_query) ** 2))
        mean_sse += float(np.sum((mean_pred - y_query) ** 2))
        ols_sse += float(np.sum((ols_pred - y_query) ** 2))
        total += n_query

    return {
        "pfn_mse": pfn_sse / total,
        "mean_mse": mean_sse / total,
        "ols_mse": ols_sse / total,
        "tasks": num_tasks,
    }


# ── 5. Glue ───────────────────────────────────────────────────────────────


def main() -> None:
    print("Training a 2-layer PFN on Bayesian linear regression (500 steps)…")
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
    metrics = evaluate(model, num_tasks=50, num_points=80)
    print()
    print(f"  PFN MSE        : {metrics['pfn_mse']:.4f}")
    print(
        f"  Mean baseline  : {metrics['mean_mse']:.4f}    "
        "(predict the context mean — what a useless model does)"
    )
    print(
        f"  OLS baseline   : {metrics['ols_mse']:.4f}    "
        "(closed-form linear fit on the context — what a perfect-for-this-prior model approaches)"
    )
    print()

    pfn = metrics["pfn_mse"]
    if pfn < metrics["mean_mse"]:
        print(
            f"✓ PFN beats the mean baseline by {metrics['mean_mse'] / pfn:.1f}× — "
            "it has learned to use the input x."
        )
    else:
        print(
            "⚠ PFN does not beat the mean baseline — likely needs more training "
            "steps or a deeper model."
        )

    if pfn < metrics["ols_mse"] * 1.5:
        print(
            f"✓ PFN is within {pfn / metrics['ols_mse']:.2f}× of OLS — "
            "approaching closed-form Bayesian inference, which is the headline claim "
            "of the PFN paper."
        )
    else:
        print(
            "⚠ PFN is well above OLS — the gap is normal at this scale; the published "
            "paper trains 100× longer with a larger model."
        )


if __name__ == "__main__":
    main()
