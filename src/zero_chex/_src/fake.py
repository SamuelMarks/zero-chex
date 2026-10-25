"""Internal fake execution contexts module re-exporting from zero_chex.mocks."""

from __future__ import annotations

from zero_chex.mocks import (
    FakeContext,
    FakeExecutionContext,
    fake_jit,
    fake_pmap,
    fake_pmap_and_jit,
    set_n_cpu_devices,
)

__all__ = [
    "FakeContext",
    "FakeExecutionContext",
    "fake_jit",
    "fake_pmap",
    "fake_pmap_and_jit",
    "set_n_cpu_devices",
]
