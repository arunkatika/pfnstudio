"""Tests for the tag_aware_transformer_encoder block.

Three guarantees this file enforces:

1. **Tag-aware block with tag_dim=0 has the same param set** as a
   regular transformer_encoder — checkpoints saved by one load into
   the other (key shape compatibility). This is what lets us roll the
   block out without invalidating any existing trained brain.

2. **Tag-aware block ignores tag when None** — passing tag=None
   short-circuits to the regular encoder forward, so models that
   were trained tag-aware can still run inference with no constraints.

3. **Tag-aware block with a real tag produces a different output**
   than the same context with no tag — proves the tag is actually
   being used, not silently discarded.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pfnstudio_core import get_block


def _make_tag_aware(tag_dim: int):
    cls = get_block("tag_aware_transformer_encoder")
    return cls(
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        ff_mult=2,
        tag_dim=tag_dim,
    )


def _make_regular():
    cls = get_block("transformer_encoder")
    return cls(d_model=16, n_heads=2, n_layers=1, dropout=0.0, ff_mult=2)


def _param_keys(block) -> list[str]:
    """Sorted list of state-dict keys for the block's encoder module.
    Used to verify checkpoint-shape compatibility."""
    return sorted(block.module.state_dict().keys())


# ---------------------------------------------------------------------------
# 1. tag_dim=0 → state-dict-compatible with regular transformer_encoder
# ---------------------------------------------------------------------------


def test_tag_dim_zero_no_tag_embedder_param():
    """When tag_dim=0, the block must not allocate any tag_embedder
    weights. Otherwise loading a checkpoint into the *other* block
    type would fail with extra/missing key errors."""
    ta = _make_tag_aware(tag_dim=0)
    assert ta.tag_embedder is None


def test_tag_dim_zero_state_dict_matches_regular_encoder():
    """The encoder module's state-dict keys must be identical to the
    regular transformer_encoder block. Cross-loading must round-trip."""
    ta = _make_tag_aware(tag_dim=0)
    reg = _make_regular()
    assert _param_keys(ta) == _param_keys(reg)


# ---------------------------------------------------------------------------
# 2. tag=None short-circuits to the regular encoder forward
# ---------------------------------------------------------------------------


def test_tag_none_matches_regular_forward():
    """For the same weights and the same input, passing tag=None to a
    tag-aware block must produce the same output as the regular encoder
    on the same input."""
    torch.manual_seed(42)
    ta = _make_tag_aware(tag_dim=4)

    torch.manual_seed(42)
    reg = _make_regular()

    # Copy the regular encoder's weights into the tag-aware block so
    # they're identical apart from the tag_embedder layer (unused here).
    ta.module.load_state_dict(reg.module.state_dict())

    x = torch.randn(2, 5, 16)
    out_ta = ta(x, tag=None)
    out_reg = reg(x)
    assert torch.allclose(out_ta, out_reg, atol=1e-6)


def test_tag_aware_output_shape_preserved():
    """Output shape must match input shape (the tag token is stripped
    after the encoder runs)."""
    ta = _make_tag_aware(tag_dim=4)
    x = torch.randn(3, 7, 16)
    tag = torch.randn(3, 4)
    out = ta(x, tag=tag)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# 3. With a real tag, output differs from no-tag — proves the tag is used
# ---------------------------------------------------------------------------


def test_real_tag_changes_output():
    torch.manual_seed(7)
    ta = _make_tag_aware(tag_dim=4)

    x = torch.randn(2, 5, 16)
    out_no_tag = ta(x, tag=None)
    out_with_tag = ta(x, tag=torch.randn(2, 4))

    # Same x, same weights — tag is the only thing different.
    assert not torch.allclose(out_no_tag, out_with_tag, atol=1e-4)


def test_different_tags_produce_different_outputs():
    """Two different tag vectors → two different outputs (sanity check
    that the tag actually flows through the attention)."""
    torch.manual_seed(11)
    ta = _make_tag_aware(tag_dim=4)
    x = torch.randn(2, 5, 16)

    tag_a = torch.zeros(2, 4)
    tag_a[:, 0] = 1.0
    tag_b = torch.zeros(2, 4)
    tag_b[:, 1] = 1.0

    out_a = ta(x, tag=tag_a)
    out_b = ta(x, tag=tag_b)
    assert not torch.allclose(out_a, out_b, atol=1e-4)
