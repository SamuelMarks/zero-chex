"""Tests for scalar assertions."""

from __future__ import annotations

import pytest
import zero_jax.numpy as jnp
from zero_flax.nnx.variables import Param

from zero_chex import (
    assert_scalar,
    assert_scalar_in,
    assert_scalar_negative,
    assert_scalar_non_negative,
    assert_scalar_positive,
)
from zero_chex.types import ZeroArray, is_scalar


class _DummyZeroDimShape:
    """Helper class exposing only a 0-D shape tuple with no scalar value."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return empty shape tuple."""
        return ()


class _DummyEmptyZeroDimArray:
    """Helper class implementing ArrayProtocol with shape () and empty to_list()."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return empty shape tuple."""
        return ()

    @property
    def dtype(self) -> str:
        """Return float32 dtype string."""
        return "float32"

    @property
    def size(self) -> int:
        """Return size 0."""
        return 0

    @property
    def ndim(self) -> int:
        """Return rank 0."""
        return 0

    @property
    def sharding(self) -> None:
        """Return None sharding."""
        return None

    def devices(self) -> tuple[()]:
        """Return empty devices tuple."""
        return ()

    def to_list(self) -> list[float]:
        """Return empty list."""
        return []


def test_assert_scalar() -> None:
    """Test assert_scalar on Python scalars, 0-D arrays, and invalid non-scalars."""
    assert_scalar(1)
    assert_scalar(1.5)
    assert_scalar(True)
    assert_scalar(1 + 2j)
    assert_scalar(ZeroArray(3.0))
    assert_scalar(ZeroArray(1 + 2j))
    assert_scalar(Param(jnp.array(3.0)))
    assert is_scalar(Param(jnp.array(3.0)))
    assert not is_scalar(Param(jnp.array([1.0, 2.0])))

    with pytest.raises(AssertionError, match="empty 0-D array"):
        assert_scalar(_DummyEmptyZeroDimArray())
    with pytest.raises(AssertionError, match="must be a scalar"):
        assert_scalar(_DummyZeroDimShape())
    with pytest.raises(AssertionError, match="must be a scalar"):
        assert_scalar([1])
    with pytest.raises(AssertionError, match="must be a scalar"):
        assert_scalar("not a scalar")
    with pytest.raises(AssertionError, match="must be a scalar"):
        assert_scalar(ZeroArray([1.0, 2.0], shape=(2,)))


def test_assert_scalar_in() -> None:
    """Test assert_scalar_in with closed, open, NaN, Inf, and complex inputs."""
    assert_scalar_in(5, 0, 10)
    assert_scalar_in(0, 0, 10, included=True)
    assert_scalar_in(10, 0, 10, included=True)
    assert_scalar_in(ZeroArray(5.0), 0, 10)

    with pytest.raises(AssertionError, match="must be a scalar"):
        assert_scalar_in(_DummyZeroDimShape(), -1, 1)
    with pytest.raises(AssertionError, match="must be a real scalar"):
        assert_scalar_in(1 + 2j, 0, 10)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(-1, 0, 10)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(11, 0, 10)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(float("nan"), 0, 10)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(float("inf"), 0, 10)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(float("-inf"), 0, 10)

    assert_scalar_in(5, 0, 10, included=False)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(0, 0, 10, included=False)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(10, 0, 10, included=False)
    with pytest.raises(AssertionError, match="must be in"):
        assert_scalar_in(float("nan"), 0, 10, included=False)


def test_assert_scalar_negative() -> None:
    """Test assert_scalar_negative for negative, zero, positive, NaN, and complex values."""
    assert_scalar_negative(-1)
    assert_scalar_negative(-0.5)
    with pytest.raises(AssertionError, match="must be negative"):
        assert_scalar_negative(0)
    with pytest.raises(AssertionError, match="must be negative"):
        assert_scalar_negative(1)
    with pytest.raises(AssertionError, match="must be negative"):
        assert_scalar_negative(float("nan"))
    with pytest.raises(AssertionError, match="must be a real scalar"):
        assert_scalar_negative(-1 + 2j)


def test_assert_scalar_non_negative() -> None:
    """Test assert_scalar_non_negative for positive, zero, negative, NaN, and complex values."""
    assert_scalar_non_negative(0)
    assert_scalar_non_negative(1)
    assert_scalar_non_negative(0.5)
    with pytest.raises(AssertionError, match="must be non-negative"):
        assert_scalar_non_negative(-1)
    with pytest.raises(AssertionError, match="must be non-negative"):
        assert_scalar_non_negative(-0.1)
    with pytest.raises(AssertionError, match="must be non-negative"):
        assert_scalar_non_negative(float("nan"))
    with pytest.raises(AssertionError, match="must be a real scalar"):
        assert_scalar_non_negative(1 + 2j)


def test_assert_scalar_positive() -> None:
    """Test assert_scalar_positive for positive, zero, negative, NaN, and complex values."""
    assert_scalar_positive(1)
    assert_scalar_positive(0.5)
    with pytest.raises(AssertionError, match="must be positive"):
        assert_scalar_positive(0)
    with pytest.raises(AssertionError, match="must be positive"):
        assert_scalar_positive(-1)
    with pytest.raises(AssertionError, match="must be positive"):
        assert_scalar_positive(float("nan"))
    with pytest.raises(AssertionError, match="must be a real scalar"):
        assert_scalar_positive(1 + 2j)
