"""Verifies the trainer's opt-in promptable-training pathway.

Default behavior: trainer passes ``tag=None`` to ``prior.sample_batch``
(verified by the closed-form benchmark staying green). This file checks
the *positive* path — when ``hp["promptable_training"] = True`` and the
prior declares axes, the trainer samples real tags and the prior
receives them.

Doesn't run the full PyTorch training loop. Substitutes a spy prior
that records every tag it's given and a step_fn that returns a no-op
loss, then asserts on the recorded tags.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pfnstudio_core import UNKNOWN, Prior, register_prior
from pfnstudio_core.registry import _clear_for_tests
from pfnstudio_core.run import EvalRef, ModelRef, PriorRef, RunSpec


class _SpyPrior(Prior):
    """Records every tag it receives. Returns numpy data so the
    trainer's default step_fn can build a tensor (and return None,
    causing a clean 'skipped' exit before any real training)."""

    spec = None
    axes = ["monotonicity"]
    seen_tags: list[Any] = []

    def sample(self, *, seed, num_points=8, tag=None, **_) -> dict[str, Any]:
        self.__class__.seen_tags.append(tag)
        rng = np.random.default_rng(seed)
        # Emit a shape the default step_fn will reject (no A/y/labels),
        # so the trainer returns 'skipped' cleanly after one batch.
        return {"some_unknown_target": rng.normal(size=num_points).astype(np.float32)}


def _tiny_trainable_model() -> Any:
    """Minimum model the trainer will accept: must expose ``modules``
    with at least one nn.Module attached so ``params`` is non-empty
    (otherwise the trainer exits with status='skipped' before sampling
    any batch). step_fn still returns None on the spy prior's batch
    shape, so the trainer exits cleanly after sampling — exactly the
    state these tests want to inspect."""
    import torch.nn as nn

    class _M:
        modules = [("noop", nn.Linear(1, 1))]

    return _M()


def _make_run(promptable: bool, steps: int = 4) -> RunSpec:
    hp: dict[str, Any] = {"steps": steps, "batch_size": 2, "lr": 1e-4, "seed": 7}
    if promptable:
        hp["promptable_training"] = True
    return RunSpec(
        id="optin_test",
        prior=PriorRef(id="_spy_prior", version="0.1.0"),
        model=ModelRef(id="noop", version="0.1.0"),
        evals=[EvalRef(id="noop_eval", version="0.1.0")],
        hyperparams=hp,
    )


@pytest.fixture(autouse=True)
def _reset_registry_and_spy():
    _clear_for_tests()
    _SpyPrior.seen_tags = []
    # Re-register the built-in blocks the trainer needs to import.
    import importlib

    import pfnstudio_core.blocks as _blocks_pkg
    from pfnstudio_core.blocks import heads, tabular, transformer

    importlib.reload(heads)
    importlib.reload(tabular)
    importlib.reload(transformer)
    importlib.reload(_blocks_pkg)
    yield


def test_promptable_disabled_passes_none_tags():
    """Default (no promptable hyperparam) → trainer passes tag=None
    every batch even though the prior declares axes."""
    torch = pytest.importorskip("torch")  # noqa: F841 — trainer needs torch
    from pfnstudio_core.training.loop import train_pfn

    register_prior("_spy_prior")(_SpyPrior)
    prior = _SpyPrior()

    model = _tiny_trainable_model()

    result = train_pfn(model, prior, _make_run(promptable=False))
    assert result["status"] == "skipped"
    assert _SpyPrior.seen_tags, "prior was never sampled — trainer exited too early"
    # Every recorded tag must be None.
    assert all(t is None for t in _SpyPrior.seen_tags), (
        f"trainer passed real tags despite promptable_training=False: {_SpyPrior.seen_tags}"
    )


def test_promptable_enabled_passes_real_tags():
    """Opt-in via hyperparam → trainer samples real tags and passes
    them to prior.sample_batch."""
    torch = pytest.importorskip("torch")  # noqa: F841
    from pfnstudio_core.training.loop import train_pfn

    register_prior("_spy_prior")(_SpyPrior)
    prior = _SpyPrior()

    model = _tiny_trainable_model()

    result = train_pfn(model, prior, _make_run(promptable=True))
    assert result["status"] == "skipped"
    assert _SpyPrior.seen_tags, "prior was never sampled"

    # Each recorded tag must be a dict with "monotonicity" set.
    assert all(isinstance(t, dict) for t in _SpyPrior.seen_tags), (
        f"trainer didn't sample real tags: {_SpyPrior.seen_tags}"
    )
    values = [t["monotonicity"] for t in _SpyPrior.seen_tags]
    valid = {"positive", "negative", "mixed", UNKNOWN}
    assert all(v in valid for v in values), f"unexpected tag value: {values}"


def test_promptable_enabled_no_axes_passes_none_tags():
    """Opt-in with a prior that has no axes → trainer still passes None."""
    torch = pytest.importorskip("torch")  # noqa: F841
    from pfnstudio_core.training.loop import train_pfn

    class _AxisLessSpyPrior(_SpyPrior):
        axes = []

    register_prior("_spy_prior")(_AxisLessSpyPrior)
    prior = _AxisLessSpyPrior()

    model = _tiny_trainable_model()

    _AxisLessSpyPrior.seen_tags = []
    result = train_pfn(model, prior, _make_run(promptable=True))
    assert result["status"] == "skipped"
    assert _AxisLessSpyPrior.seen_tags
    assert all(t is None for t in _AxisLessSpyPrior.seen_tags)
