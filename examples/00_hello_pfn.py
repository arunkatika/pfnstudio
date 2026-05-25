"""00 — Hello PFN. Sample one task from a prior.

The smallest possible PFN Studio program. Defines a prior, samples one
task from it, prints the result. Runs in under a second. No PyTorch
needed (no training happens), so this is the right thing to run after
``pip install pfnstudio-core`` to confirm the install works before
committing to the longer 01/02 training demos.

What a PFN trains on is exactly this: many such tasks, drawn fresh from
the prior every step.

Requires: ``pip install pfnstudio-core``
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pfnstudio_core import Prior
from pfnstudio_core.registry import register_prior


@register_prior("hello_linear")
class HelloPrior(Prior):
    """The tiniest sensible PFN prior — random y = a·x + b + ε."""

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 8,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal(0.0, 1.0))
        b = float(rng.normal(0.0, 1.0))
        x = rng.uniform(-2.0, 2.0, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, 0.1, size=num_points)).astype(np.float32)
        return {
            "X": x.reshape(-1, 1),
            "y": y,
            "a_true": a,
            "b_true": b,
        }


def main() -> None:
    task = HelloPrior().sample(seed=42, num_points=8)

    print("Sampled one task from the hello-linear prior:")
    print()
    print(f"  Latent line:  y = {task['a_true']:+.3f} · x + {task['b_true']:+.3f}")
    print("  Noise model:  ε ~ N(0, 0.1)")
    print()
    print("  x:", np.round(task["X"].ravel(), 3).tolist())
    print("  y:", np.round(task["y"], 3).tolist())
    print()
    print("Every training step of a PFN samples a fresh task like this one.")
    print("Next stop: examples/01_linear_regression.py — train a small PFN")
    print("that learns to do in-context regression over this family of tasks.")


if __name__ == "__main__":
    main()
