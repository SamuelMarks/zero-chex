"""Tests for basic value, broadcastability, divisibility, and device availability assertions."""

from __future__ import annotations

import pytest

from zero_chex import (
    assert_devices_available,
    assert_equal,
    assert_exactly_one_is_none,
    assert_gpu_available,
    assert_is_broadcastable,
    assert_is_divisible,
    assert_not_both_none,
    assert_tpu_available,
    restrict_backends,
    set_n_cpu_devices,
)
from zero_chex.devices import _num_devices_available
from zero_chex.mocks import _STATE


def test_num_devices_available() -> None:
    """Test _num_devices_available across cpu, gpu, tpu, and invalid platforms."""
    assert _num_devices_available("cpu") == 1
    assert _num_devices_available("gpu") == 1
    assert _num_devices_available("tpu") == 1
    set_n_cpu_devices(4)
    assert _num_devices_available("cpu") == 4
    with pytest.raises(ValueError, match="Unknown device type"):
        _num_devices_available("unknown")


def test_assert_devices_available() -> None:
    """Test assert_devices_available with exact, not_less_than, and backend restriction modes."""
    assert_devices_available(1, "cpu")
    assert_devices_available(1, "gpu", backend="gpu")
    assert_devices_available(1, "tpu")
    assert_devices_available(1, "cpu", not_less_than=True)

    with pytest.raises(ValueError, match="Unknown backend"):
        assert_devices_available(1, "cpu", backend="quantum")

    with restrict_backends(forbidden=["gpu"]):
        with pytest.raises(AssertionError, match="forbidden by restrict_backends"):
            assert_devices_available(1, "gpu")

    with restrict_backends(allowed=["cpu"]):
        with pytest.raises(AssertionError, match="not in allowed backends"):
            assert_devices_available(1, "tpu")

    with pytest.raises(AssertionError, match="Only 1 < 2 CPUs available"):
        assert_devices_available(2, "cpu", not_less_than=True)

    with pytest.raises(AssertionError, match="No 2 CPUs available"):
        assert_devices_available(2, "cpu", not_less_than=False)


def test_assert_gpu_and_tpu_available() -> None:
    """Test assert_gpu_available and assert_tpu_available pass, failure, and backend restriction paths."""
    assert_gpu_available()
    assert_tpu_available()

    with restrict_backends(forbidden=["gpu"]):
        with pytest.raises(AssertionError, match="forbidden by restrict_backends"):
            assert_gpu_available()

    with restrict_backends(forbidden=["tpu"]):
        with pytest.raises(AssertionError, match="forbidden by restrict_backends"):
            assert_tpu_available()

    _STATE.n_gpu_devices = 0
    try:
        with pytest.raises(AssertionError, match="No GPU devices available"):
            assert_gpu_available()
    finally:
        _STATE.n_gpu_devices = 1

    _STATE.n_tpu_devices = 0
    try:
        with pytest.raises(AssertionError, match="No TPU devices available"):
            assert_tpu_available()
    finally:
        _STATE.n_tpu_devices = 1


def test_assert_equal() -> None:
    """Test assert_equal for equal and unequal values."""
    assert_equal(1, 1)
    assert_equal([1, 2], [1, 2])
    with pytest.raises(AssertionError, match="Assertion failed"):
        assert_equal(1, 2)


def test_assert_exactly_one_is_none() -> None:
    """Test assert_exactly_one_is_none for valid and invalid pairs."""
    assert_exactly_one_is_none(None, 1)
    assert_exactly_one_is_none(1, None)
    with pytest.raises(AssertionError, match="One and exactly one"):
        assert_exactly_one_is_none(None, None)
    with pytest.raises(AssertionError, match="One and exactly one"):
        assert_exactly_one_is_none(1, 2)


def test_assert_not_both_none() -> None:
    """Test assert_not_both_none for valid and invalid pairs."""
    assert_not_both_none(None, 1)
    assert_not_both_none(1, None)
    assert_not_both_none(1, 2)
    with pytest.raises(AssertionError, match="At least one of the arguments"):
        assert_not_both_none(None, None)


def test_assert_is_broadcastable() -> None:
    """Test assert_is_broadcastable with compatible and incompatible shapes."""
    assert_is_broadcastable((3, 1), (1, 4))
    assert_is_broadcastable((2, 3, 4), (3, 1))
    with pytest.raises(AssertionError, match="not broadcastable"):
        assert_is_broadcastable((3, 2), (3, 4))


def test_assert_is_divisible() -> None:
    """Test assert_is_divisible with divisible, non-divisible, and zero divisor inputs."""
    assert_is_divisible(10, 2)
    assert_is_divisible(10, 5)
    with pytest.raises(AssertionError, match="not divisible"):
        assert_is_divisible(10, 3)
    with pytest.raises(AssertionError, match="not divisible"):
        assert_is_divisible(10, 0)
