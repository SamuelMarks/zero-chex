"""Pytest configuration for zero_chex test suite."""

from __future__ import annotations

import pytest

from zero_chex.mocks import clear_trace_counter, enable_asserts, set_n_cpu_devices


@pytest.fixture(autouse=True)
def reset_chex_state() -> None:
    """Reset global assertion state, trace counters, and device counts before each test."""
    enable_asserts()
    clear_trace_counter()
    set_n_cpu_devices(1)
