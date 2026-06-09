"""Tests for scalar assertions."""

import pytest

from zero_chex import (
    assert_scalar,
    assert_scalar_in,
    assert_scalar_negative,
    assert_scalar_non_negative,
    assert_scalar_positive,
)


def test_assert_scalar() -> None:
    """Test assert_scalar."""
    assert_scalar(1)
    assert_scalar(1.0)
    with pytest.raises(AssertionError, match="must be a scalar"):
        assert_scalar("1")  # type: ignore


def test_assert_scalar_in() -> None:
    """Test assert_scalar_in."""
    assert_scalar_in(1, 0, 2)
    assert_scalar_in(1, 1, 2)
    assert_scalar_in(2, 1, 2)

    with pytest.raises(AssertionError, match=r"must be in \[0, 2\]"):
        assert_scalar_in(3, 0, 2)

    assert_scalar_in(1.5, 1, 2, included=False)
    with pytest.raises(AssertionError, match=r"must be in \(1, 2\)"):
        assert_scalar_in(1, 1, 2, included=False)
    with pytest.raises(AssertionError, match=r"must be in \(1, 2\)"):
        assert_scalar_in(2, 1, 2, included=False)


def test_assert_scalar_negative() -> None:
    """Test assert_scalar_negative."""
    assert_scalar_negative(-1)
    assert_scalar_negative(-0.5)
    with pytest.raises(AssertionError, match="must be negative"):
        assert_scalar_negative(0)
    with pytest.raises(AssertionError, match="must be negative"):
        assert_scalar_negative(1)


def test_assert_scalar_non_negative() -> None:
    """Test assert_scalar_non_negative."""
    assert_scalar_non_negative(0)
    assert_scalar_non_negative(1)
    with pytest.raises(AssertionError, match="must be non-negative"):
        assert_scalar_non_negative(-1)


def test_assert_scalar_positive() -> None:
    """Test assert_scalar_positive."""
    assert_scalar_positive(1)
    assert_scalar_positive(0.5)
    with pytest.raises(AssertionError, match="must be positive"):
        assert_scalar_positive(0)
    with pytest.raises(AssertionError, match="must be positive"):
        assert_scalar_positive(-1)
