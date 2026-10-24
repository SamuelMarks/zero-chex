"""Internal scalar assertions module re-exporting zero_chex.scalars."""

from __future__ import annotations

from zero_chex.scalars import (
    assert_scalar,
    assert_scalar_in,
    assert_scalar_negative,
    assert_scalar_non_negative,
    assert_scalar_positive,
)

__all__ = [
    "assert_scalar",
    "assert_scalar_in",
    "assert_scalar_negative",
    "assert_scalar_non_negative",
    "assert_scalar_positive",
]
