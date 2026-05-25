"""Tests for the fixed-length tag encoding utilities.

These don't touch torch or training — pure numpy + axis-registry tests
that verify the encoder produces stable, well-shaped vectors and
honors the back-compat contract for empty / unknown tags.
"""

from __future__ import annotations

import numpy as np
import pytest
from pfnstudio_core import UNKNOWN, Axis, encode_tag, get_axis, register_axis, sample_tag, tag_dim


@pytest.fixture
def monotonicity():
    return get_axis("monotonicity")


@pytest.fixture
def range_axis():
    """A range axis registered just for these tests."""
    axis = Axis(
        name="_test_range",
        kind="range",
        values=(0.0, 24.0),
        unknown_mass=0.3,
    )
    # register_axis is idempotent for identical definitions, so this is
    # safe to call across test runs.
    return register_axis(axis)


@pytest.fixture
def boolean_axis():
    axis = Axis(
        name="_test_bool",
        kind="boolean",
        unknown_mass=0.3,
    )
    return register_axis(axis)


# ---------------------------------------------------------------------------
# tag_dim
# ---------------------------------------------------------------------------


def test_tag_dim_categorical(monotonicity):
    # 3 values + 1 UNKNOWN slot
    assert tag_dim([monotonicity]) == 4


def test_tag_dim_range(range_axis):
    assert tag_dim([range_axis]) == 2


def test_tag_dim_boolean(boolean_axis):
    assert tag_dim([boolean_axis]) == 3


def test_tag_dim_combined(monotonicity, range_axis, boolean_axis):
    assert tag_dim([monotonicity, range_axis, boolean_axis]) == 4 + 2 + 3


def test_tag_dim_empty():
    assert tag_dim([]) == 0


# ---------------------------------------------------------------------------
# encode_tag — back-compat (empty / None / all-UNKNOWN equivalence)
# ---------------------------------------------------------------------------


def test_encode_empty_tag_matches_none(monotonicity, range_axis):
    """Empty dict, None, and explicit UNKNOWN for every axis must all
    produce the same encoded vector."""
    axes = [monotonicity, range_axis]
    a = encode_tag(None, axes)
    b = encode_tag({}, axes)
    c = encode_tag({"monotonicity": UNKNOWN, "_test_range": UNKNOWN}, axes)

    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_encode_no_axes_returns_empty():
    assert encode_tag(None, []).shape == (0,)
    assert encode_tag({"monotonicity": "positive"}, []).shape == (0,)


# ---------------------------------------------------------------------------
# encode_tag — categorical
# ---------------------------------------------------------------------------


def test_encode_categorical_positive(monotonicity):
    vec = encode_tag({"monotonicity": "positive"}, [monotonicity])
    # values=("positive", "negative", "mixed"), then UNKNOWN slot.
    assert vec.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_encode_categorical_unknown_slot(monotonicity):
    vec = encode_tag({"monotonicity": UNKNOWN}, [monotonicity])
    assert vec.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_encode_categorical_rejects_unknown_value(monotonicity):
    with pytest.raises(ValueError, match="not in"):
        encode_tag({"monotonicity": "bogus"}, [monotonicity])


# ---------------------------------------------------------------------------
# encode_tag — range
# ---------------------------------------------------------------------------


def test_encode_range_real_value(range_axis):
    vec = encode_tag({"_test_range": 12.0}, [range_axis])
    # is_unknown_flag = 0, normalised = 12/24 = 0.5
    assert vec.tolist() == [0.0, 0.5]


def test_encode_range_unknown(range_axis):
    vec = encode_tag({"_test_range": UNKNOWN}, [range_axis])
    # is_unknown_flag = 1, normalised stays 0
    assert vec.tolist() == [1.0, 0.0]


def test_encode_range_clips_out_of_bounds(range_axis):
    vec_low = encode_tag({"_test_range": -5.0}, [range_axis])
    vec_high = encode_tag({"_test_range": 100.0}, [range_axis])
    assert vec_low.tolist() == [0.0, 0.0]
    assert vec_high.tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# encode_tag — boolean
# ---------------------------------------------------------------------------


def test_encode_boolean_true(boolean_axis):
    vec = encode_tag({"_test_bool": True}, [boolean_axis])
    assert vec.tolist() == [0.0, 1.0, 0.0]


def test_encode_boolean_false(boolean_axis):
    vec = encode_tag({"_test_bool": False}, [boolean_axis])
    assert vec.tolist() == [1.0, 0.0, 0.0]


def test_encode_boolean_unknown(boolean_axis):
    vec = encode_tag({"_test_bool": UNKNOWN}, [boolean_axis])
    assert vec.tolist() == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# sample_tag — rate honoring
# ---------------------------------------------------------------------------


def test_sample_tag_returns_one_value_per_axis(monotonicity, range_axis):
    rng = np.random.default_rng(0)
    tag = sample_tag([monotonicity, range_axis], rng)
    assert set(tag.keys()) == {"monotonicity", "_test_range"}


def test_sample_tag_no_axes_returns_empty_dict():
    rng = np.random.default_rng(0)
    assert sample_tag([], rng) == {}
