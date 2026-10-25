"""Scalar value assertions for zero_chex."""

from __future__ import annotations

import math

from zero_chex.mocks import are_asserts_enabled
from zero_chex.types import (
    ArrayProtocol,
    Numeric,
    PyTree,
    Scalar,
    VariableLeafProtocol,
    _extract_flat_scalars,
)


def _extract_scalar_value(x: PyTree) -> Scalar:
    """Validate that x is a numeric scalar or 0-D array and return its raw scalar value.

    Args:
        x: Value to validate and extract.

    Returns:
        Python scalar (int, float, complex, or bool).

    Raises:
        AssertionError: If x is not a scalar or is a multi-dimensional array.
    """
    if isinstance(x, VariableLeafProtocol):
        return _extract_scalar_value(x.value)
    if isinstance(x, (int, float, complex, bool)):
        return x
    if isinstance(x, ArrayProtocol) and x.shape == ():
        items = _extract_flat_scalars(x)
        if items:
            return items[0]
        raise AssertionError(f"The argument {x!r} is an empty 0-D array with no scalar value.")
    raise AssertionError(f"The argument {x!r} must be a scalar, got {type(x).__name__}.")


def _extract_real_scalar_float(x: PyTree) -> float:
    """Validate that x is an ordered real numeric scalar and return its float value.

    Args:
        x: Value to validate and convert.

    Returns:
        Floating-point representation of x.

    Raises:
        AssertionError: If x is not a scalar, is complex, or is a multi-dimensional array.
    """
    raw = _extract_scalar_value(x)
    if isinstance(raw, complex):
        raise AssertionError(f"The argument {x!r} must be a real scalar for ordering comparisons, got complex {raw!r}.")
    return float(raw)


def assert_scalar(x: Numeric | PyTree) -> None:
    """Assert that x is a Python scalar (int, float, complex, bool) or 0-D array.

    Args:
        x: The value to check.

    Raises:
        AssertionError: If assertions are enabled and x is not a scalar.
    """
    if not are_asserts_enabled():
        return
    _extract_scalar_value(x)


def assert_scalar_in(
    x: Numeric | PyTree,
    min_: float | int,
    max_: float | int,
    included: bool = True,
) -> None:
    """Assert that scalar x lies within the interval bounded by min_ and max_.

    Args:
        x: The scalar value to check.
        min_: Lower bound of the interval.
        max_: Upper bound of the interval.
        included: If True, checks closed interval [min_, max_]; otherwise open (min_, max_).

    Raises:
        AssertionError: If x is not a real scalar, is NaN, or falls outside the specified interval.
    """
    if not are_asserts_enabled():
        return
    val = _extract_real_scalar_float(x)
    if included:
        if math.isnan(val) or not (min_ <= val <= max_):
            raise AssertionError(f"The argument x must be in [{min_}, {max_}], was {val}.")
    else:
        if math.isnan(val) or not (min_ < val < max_):
            raise AssertionError(f"The argument x must be in ({min_}, {max_}), was {val}.")


def assert_scalar_negative(x: Numeric | PyTree) -> None:
    """Assert that scalar x is strictly negative (x < 0).

    Args:
        x: The scalar value to check.

    Raises:
        AssertionError: If x is not a real scalar, is NaN, or x >= 0.
    """
    if not are_asserts_enabled():
        return
    val = _extract_real_scalar_float(x)
    if math.isnan(val) or not (val < 0):
        raise AssertionError(f"The argument x must be negative, was {val}.")


def assert_scalar_non_negative(x: Numeric | PyTree) -> None:
    """Assert that scalar x is non-negative (x >= 0).

    Args:
        x: The scalar value to check.

    Raises:
        AssertionError: If x is not a real scalar, is NaN, or x < 0.
    """
    if not are_asserts_enabled():
        return
    val = _extract_real_scalar_float(x)
    if math.isnan(val) or not (val >= 0):
        raise AssertionError(f"The argument x must be non-negative, was {val}.")


def assert_scalar_positive(x: Numeric | PyTree) -> None:
    """Assert that scalar x is strictly positive (x > 0).

    Args:
        x: The scalar value to check.

    Raises:
        AssertionError: If x is not a real scalar, is NaN, or x <= 0.
    """
    if not are_asserts_enabled():
        return
    val = _extract_real_scalar_float(x)
    if math.isnan(val) or not (val > 0):
        raise AssertionError(f"The argument x must be positive, was {val}.")


__all__ = [
    "assert_scalar",
    "assert_scalar_in",
    "assert_scalar_negative",
    "assert_scalar_non_negative",
    "assert_scalar_positive",
]
