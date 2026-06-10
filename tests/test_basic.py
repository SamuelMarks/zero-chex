"""Tests for basic assertions."""

from typing import Any
from unittest.mock import patch
import pytest

from zero_chex._src.asserts.basic import (
    _num_devices_available,
    assert_devices_available,
    assert_equal,
    assert_exactly_one_is_none,
    assert_gpu_available,
    assert_is_broadcastable,
    assert_is_divisible,
    assert_not_both_none,
    assert_tpu_available,
)


def test_assert_equal() -> None:
    """Test assert_equal."""
    assert_equal(1, 1)
    with pytest.raises(AssertionError):
        assert_equal(1, 2)


def test_assert_exactly_one_is_none() -> None:
    """Test assert_exactly_one_is_none."""
    assert_exactly_one_is_none(None, 1)
    assert_exactly_one_is_none(1, None)
    with pytest.raises(AssertionError, match="One and exactly one"):
        assert_exactly_one_is_none(None, None)
    with pytest.raises(AssertionError, match="One and exactly one"):
        assert_exactly_one_is_none(1, 2)


def test_assert_not_both_none() -> None:
    """Test assert_not_both_none."""
    assert_not_both_none(1, None)
    assert_not_both_none(None, 1)
    assert_not_both_none(1, 2)
    with pytest.raises(AssertionError, match="At least one"):
        assert_not_both_none(None, None)


def test_num_devices_available() -> None:
    """Test _num_devices_available."""
    assert _num_devices_available("cpu") >= 1
    with pytest.raises(ValueError, match="Unknown device type"):
        _num_devices_available("unknown")


def test_assert_devices_available() -> None:
    """Test assert_devices_available."""
    # We should have at least 1 CPU
    assert_devices_available(1, "cpu", not_less_than=True)

    with pytest.raises(AssertionError, match="Only"):
        assert_devices_available(999, "cpu", not_less_than=True)

    with pytest.raises(AssertionError):
        assert_devices_available(999, "cpu", not_less_than=False)


@patch("zero_chex._src.asserts.basic._num_devices_available")
def test_assert_gpu_available(mock_num: Any) -> None:
    """Test assert_gpu_available."""
    mock_num.return_value = 1
    assert_gpu_available()

    mock_num.return_value = 0
    with pytest.raises(AssertionError, match="No GPU devices available"):
        assert_gpu_available()


@patch("zero_chex._src.asserts.basic._num_devices_available")
def test_assert_tpu_available(mock_num: Any) -> None:
    """Test assert_tpu_available."""
    mock_num.return_value = 1
    assert_tpu_available()

    mock_num.return_value = 0
    with pytest.raises(AssertionError, match="No TPU devices available"):
        assert_tpu_available()


def test_assert_is_broadcastable() -> None:
    """Test assert_is_broadcastable."""
    assert_is_broadcastable((3,), (5, 3))
    assert_is_broadcastable((1, 3), (5, 3))

    with pytest.raises(AssertionError, match="is not broadcastable"):
        assert_is_broadcastable((3,), (5, 4))

    assert_is_broadcastable((5, 3), (3,))


def test_assert_is_divisible() -> None:
    """Test assert_is_divisible."""
    assert_is_divisible(10, 2)
    assert_is_divisible(10, 5)
    with pytest.raises(AssertionError, match="not divisible"):
        assert_is_divisible(10, 3)
