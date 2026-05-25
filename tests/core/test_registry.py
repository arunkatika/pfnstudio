import pytest
from pfnstudio_core.registry import (
    _clear_for_tests,
    get_block,
    get_prior,
    list_blocks,
    list_priors,
    register_block,
    register_prior,
)


def setup_function():
    _clear_for_tests()


def test_register_and_get_prior():
    @register_prior("foo")
    class FooPrior:
        pass

    assert get_prior("foo") is FooPrior
    assert "foo" in list_priors()


def test_register_and_get_block():
    @register_block("widget")
    class Widget:
        def __init__(self, n: int = 1):
            self.n = n

    cls = get_block("widget")
    assert cls(n=4).n == 4
    assert "widget" in list_blocks()


def test_duplicate_registration_raises():
    @register_prior("dup")
    class A:
        pass

    with pytest.raises(ValueError):

        @register_prior("dup")
        class B:
            pass


def test_unknown_lookup_raises():
    with pytest.raises(KeyError):
        get_prior("nope")
