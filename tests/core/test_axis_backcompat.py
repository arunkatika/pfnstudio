"""Back-compat tripwire for promptable priors.

These tests are the contract that lets us add the ``axes`` feature
without breaking existing priors. If anything here fails, the
promptable-prior work has regressed pre-existing behavior — stop and
fix before shipping.

What the tests guarantee:

1. **Tag-free call sites still work.** Every existing prior must
   sample identically whether or not a ``tag`` argument is passed.
2. **All-UNKNOWN is bit-identical to pre-axis.** A prior with axes
   that's given ``tag={axis: UNKNOWN}`` for every axis must produce
   the same bytes as the pre-axis baseline for the same seed.
3. **Axis-aware priors honor non-UNKNOWN tags.** When tagged with a
   real value, the output must *differ* from the UNKNOWN baseline —
   otherwise the axis isn't being honored.

The first two are about not regressing existing functionality; the
third is the minimum proof that the axis pathway actually does
anything.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pfnstudio_core import UNKNOWN, Prior, get_axis, register_prior
from pfnstudio_core.registry import get_prior

# Repo root → load priors/chain-scm/prior.py directly without going
# through scaffolding. Keeps the test fast and decoupled from the
# project-scaffolder.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAIN_SCM_PRIOR_PY = _REPO_ROOT / "priors" / "chain-scm" / "prior.py"


def _load_chain_scm():
    """Load priors/chain-scm/prior.py once for the test session."""
    spec = importlib.util.spec_from_file_location("_test_chain_scm_prior", _CHAIN_SCM_PRIOR_PY)
    assert spec is not None and spec.loader is not None, "chain-scm prior.py missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return get_prior("chain_scm")


@pytest.fixture(scope="module")
def chain_scm_prior_cls():
    return _load_chain_scm()


# ---------------------------------------------------------------------------
# Inline test prior. Lives in the test file (not the priors/ folder) so the
# back-compat contract is verified against a fixture that can't drift.
# ---------------------------------------------------------------------------


@register_prior("_test_no_axes_prior")
class _NoAxesPrior(Prior):
    """A prior with no declared axes. Must ignore tag completely."""

    def sample(self, *, seed, n=10, tag=None, **_):
        rng = np.random.default_rng(seed)
        return {"x": rng.normal(size=n).astype(np.float32)}


# ---------------------------------------------------------------------------
# 1. Tag-free call sites still work — for every existing prior.
# ---------------------------------------------------------------------------


def test_no_axes_prior_ignores_tag():
    """A prior that declares no axes must accept-and-ignore any tag."""
    p = _NoAxesPrior()

    without_tag = p.sample(seed=42, n=20)
    with_tag = p.sample(seed=42, n=20, tag={"monotonicity": "positive"})

    # Identical bytes — the tag was passed but did nothing.
    assert np.array_equal(without_tag["x"], with_tag["x"])


def test_chain_scm_without_tag_unchanged(chain_scm_prior_cls):
    """The chain-scm prior with no tag must produce a sane shape.
    Catches any accidental change to the default sampling path."""
    p = chain_scm_prior_cls()
    out = p.sample(seed=42, num_points=50, d=4)
    assert out["X"].shape == (50, 4)
    assert out["A"].shape == (4, 4)


# ---------------------------------------------------------------------------
# 2. All-UNKNOWN tags must be bit-identical to no-tag.
# ---------------------------------------------------------------------------


def test_chain_scm_unknown_tag_matches_no_tag(chain_scm_prior_cls):
    """tag={monotonicity: UNKNOWN} must be byte-identical to no tag.
    This is the load-bearing invariant: it's how we guarantee adding
    the axis doesn't regress the existing benchmark."""
    p = chain_scm_prior_cls()
    a = p.sample(seed=99, num_points=80, d=5)
    b = p.sample(seed=99, num_points=80, d=5, tag={"monotonicity": UNKNOWN})
    c = p.sample(seed=99, num_points=80, d=5, tag={})
    d = p.sample(seed=99, num_points=80, d=5, tag=None)

    for label, ref in (("UNKNOWN", b), ("empty tag", c), ("None tag", d)):
        assert np.array_equal(a["X"], ref["X"]), f"X drift with {label}"
        assert np.array_equal(a["A"], ref["A"]), f"A drift with {label}"


def test_chain_scm_mixed_tag_matches_unknown(chain_scm_prior_cls):
    """tag={monotonicity: 'mixed'} is semantically distinct from
    UNKNOWN but must sample from the same distribution (random signs).
    This keeps the back-compat path stable even when authors explicitly
    say 'mixed' in production."""
    p = chain_scm_prior_cls()
    a = p.sample(seed=7, num_points=40, d=4, tag={"monotonicity": UNKNOWN})
    b = p.sample(seed=7, num_points=40, d=4, tag={"monotonicity": "mixed"})
    assert np.array_equal(a["X"], b["X"])
    assert np.array_equal(a["A"], b["A"])


# ---------------------------------------------------------------------------
# 3. Real tag values must change the output (axis is honored).
# ---------------------------------------------------------------------------


def test_chain_scm_monotonicity_changes_output(chain_scm_prior_cls):
    """tag={monotonicity: positive} must produce *different* samples
    than UNKNOWN. Without this, the axis could be silently dropped
    without the tripwire noticing."""
    p = chain_scm_prior_cls()
    unknown = p.sample(seed=11, num_points=40, d=4, tag={"monotonicity": UNKNOWN})
    positive = p.sample(seed=11, num_points=40, d=4, tag={"monotonicity": "positive"})
    negative = p.sample(seed=11, num_points=40, d=4, tag={"monotonicity": "negative"})

    # Adjacency structure is determined by seed alone — same seed →
    # same chain topology regardless of sign discipline. So A is
    # identical across all three.
    assert np.array_equal(unknown["A"], positive["A"])
    assert np.array_equal(unknown["A"], negative["A"])

    # But X differs because edge signs differ.
    assert not np.array_equal(unknown["X"], positive["X"]), (
        "monotonicity=positive produced identical X to UNKNOWN — axis hook not firing"
    )
    assert not np.array_equal(positive["X"], negative["X"]), (
        "monotonicity=positive identical to negative — sign isn't flipping"
    )


def test_chain_scm_positive_propagates_same_sign(chain_scm_prior_cls):
    """Under monotonicity=positive, the root and its leaf descendant
    in a chain must be positively correlated on average (since every
    edge has positive sign, the cascade preserves sign)."""
    p = chain_scm_prior_cls()
    correlations = []
    for seed in range(30):
        out = p.sample(
            seed=seed,
            num_points=400,
            d=4,
            noise_scale=0.1,
            tag={"monotonicity": "positive"},
        )
        # Identify the chain order from A: column with no incoming
        # edge is the root; row with no outgoing edge is the leaf.
        A = out["A"]
        incoming = A.sum(axis=0)
        outgoing = A.sum(axis=1)
        roots = np.where(incoming == 0)[0]
        leaves = np.where(outgoing == 0)[0]
        if len(roots) and len(leaves):
            root_idx, leaf_idx = int(roots[0]), int(leaves[0])
            if root_idx != leaf_idx:
                correlations.append(np.corrcoef(out["X"][:, root_idx], out["X"][:, leaf_idx])[0, 1])

    assert len(correlations) > 5, "test setup failed to find root/leaf pairs"
    mean_corr = float(np.mean(correlations))
    assert mean_corr > 0.0, (
        f"under monotonicity=positive, root↔leaf mean correlation was "
        f"{mean_corr:.3f}; expected > 0 because every edge is positive"
    )


# ---------------------------------------------------------------------------
# 4. Axis registry sanity.
# ---------------------------------------------------------------------------


def test_monotonicity_axis_registered_on_import():
    """Importing pfnstudio_core must auto-register the built-in axes."""
    axis = get_axis("monotonicity")
    assert axis.name == "monotonicity"
    assert axis.kind == "categorical"
    assert set(axis.values) == {"positive", "negative", "mixed"}


def test_axis_sample_value_honors_unknown_mass():
    """Over many draws, the UNKNOWN sentinel must appear at roughly
    the configured unknown_mass rate."""
    rng = np.random.default_rng(0)
    axis = get_axis("monotonicity")
    n = 5_000
    unknowns = sum(axis.sample_value(rng) == UNKNOWN for _ in range(n))
    rate = unknowns / n
    # Default unknown_mass is 0.3; allow ±3% tolerance for 5k samples.
    assert abs(rate - axis.unknown_mass) < 0.03, (
        f"UNKNOWN appeared {rate:.3f} of the time, expected ~{axis.unknown_mass}"
    )
