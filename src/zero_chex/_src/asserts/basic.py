"""Internal basic assertions module re-exporting from zero_chex.assertions and zero_chex.devices."""

from __future__ import annotations

from zero_chex.assertions import (
    assert_equal,
    assert_exactly_one_is_none,
    assert_is_broadcastable,
    assert_is_divisible,
    assert_not_both_none,
)
from zero_chex.devices import (
    _num_devices_available,
    assert_devices_available,
    assert_gpu_available,
    assert_tpu_available,
)

__all__ = [
    "_num_devices_available",
    "assert_devices_available",
    "assert_equal",
    "assert_exactly_one_is_none",
    "assert_gpu_available",
    "assert_is_broadcastable",
    "assert_is_divisible",
    "assert_not_both_none",
    "assert_tpu_available",
]
