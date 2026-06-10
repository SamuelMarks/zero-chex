"""Tests for classes."""

import pytest

from zero_chex import ChexVariantType, Dimensions, TestCase


def test_chex_variant_type() -> None:
    """Test ChexVariantType."""
    assert str(ChexVariantType.WITH_JIT) == "_with_jit"
    assert str(ChexVariantType.WITHOUT_JIT) == "_without_jit"


def test_dimensions() -> None:
    """Test Dimensions."""
    dims = Dimensions(B=3, T=5, N=7)
    assert dims["NBT"] == (7, 3, 5)
    assert dims.size("BT") == 15
    assert dims["(BT)N"] == (15, 7)

    dims.W = None

    assert dims["BT123"] == (3, 5, 1, 2, 3)

    dims["__M"] = (10, 20, 30)
    assert dims["M"] == (30,)

    with pytest.raises(ValueError, match="cannot take product"):
        dims.size("BTW")

    with pytest.raises(ValueError, match="cannot take product"):
        dims.size("BT0")

    with pytest.raises(ValueError, match="nested parentheses"):
        dims["((B))"]

    with pytest.raises(ValueError, match="unmatched parentheses"):
        dims[")B("]

    with pytest.raises(ValueError, match="empty parentheses"):
        dims["()B"]

    with pytest.raises(ValueError, match="unmatched parentheses"):
        dims["(B"]

    dims["XY"] = (2, 4)
    assert repr(dims) == "Dimensions(B=3, M=30, N=7, T=5, W=None, X=2, Y=4)"

    dims["_"] = (5,)

    del dims["X"]
    with pytest.raises(KeyError):
        dims["X"]

    with pytest.raises(ValueError, match="different lengths"):
        dims["AB"] = (1, 2, 3)

    with pytest.raises(TypeError, match="key must be a string"):
        dims[1] = (1,)  # type: ignore

    with pytest.raises(TypeError, match="key must be a string"):
        dims[1]  # type: ignore

    with pytest.raises(TypeError, match="key must be a string"):
        del dims[1]  # type: ignore

    with pytest.raises(TypeError, match="value must be sized"):
        dims["A"] = 1  # type: ignore

    with pytest.raises(TypeError, match="dimension name must be a string"):
        dims._validate_dim(1)

    with pytest.raises(KeyError, match="contain letters"):
        dims._validate_dim("1")

    del dims["_"]  # should return None and not raise

    dims._deldim("_")

    with pytest.raises(KeyError):
        dims._deldim("NOT_EXIST")

    with pytest.raises(KeyError):
        del dims["NOT_EXIST"]

    assert dims._getdim("*") is None
    assert dims._getdim("5") == 5
    with pytest.raises(KeyError):
        dims._getdim("NOT_EXIST")

    # hit line 30
    dims2 = Dimensions(W=None)
    assert dims2["W"] == (None,)


def test_test_case() -> None:
    """Test TestCase."""
    tc = TestCase()
    with pytest.raises(RuntimeError, match="self.variant is not defined"):
        tc.variant()
