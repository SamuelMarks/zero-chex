# ruff: noqa: F821, F403, F405, E402, E731, F841, E721

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        user = None


"""Scalar assertions."""

from typing import Any


def assert_scalar(x: "float | int") -> "NoneType":
    """Checks that ``x`` is a scalar, as defined in `pytypes.py` (int or float).

    Args:
      x: An object to check.

    Raises:
      AssertionError: If ``x`` is not a scalar as per definition in pytypes.py.
    """

    if not isinstance(x, (int, float)) and not (
        hasattr(x, "shape") and getattr(x, "shape") == ()
    ):
        raise AssertionError(f"The argument {x} must be a scalar, got {type(x)}.")


def assert_scalar_in(
    x: "Any", min_: "float | int", max_: "float | int", included: "bool" = True
) -> "NoneType":
    """Checks that argument is a scalar within segment (by default).

    Args:
      x: An object to check.
      min_: A left border of the segment.
      max_: A right border of the segment.
      included: Whether to include the borders of the segment in the set of
        allowed values.

    Raises:
      AssertionError: If ``x`` is not a scalar; if ``x`` falls out of the segment.
    """
    assert_scalar(x)
    if included:
        if not min_ <= x <= max_:
            raise AssertionError(f"The argument must be in [{min_}, {max_}], got {x}.")
    else:
        if not min_ < x < max_:
            raise AssertionError(f"The argument must be in ({min_}, {max_}), got {x}.")


def assert_scalar_negative(x: "float | int") -> "NoneType":
    """Checks that a scalar is negative.

    Args:
      x: An object to check.

    Raises:
      AssertionError: If ``x`` is not a scalar; if ``x`` is non-negative.
    """
    assert_scalar(x)
    if not x < 0:
        raise AssertionError(f"The argument must be negative, got {x}.")


def assert_scalar_non_negative(x: "float | int") -> "NoneType":
    """Checks that a scalar is non-negative.

    Args:
      x: An object to check.

    Raises:
      AssertionError: If ``x`` is not a scalar; if ``x`` is negative.
    """
    assert_scalar(x)
    if not x >= 0:
        raise AssertionError(f"The argument must be non-negative, got {x}.")


def assert_scalar_positive(x: "float | int") -> "NoneType":
    """Checks that a scalar is positive.

    Args:
      x: An object to check.

    Raises:
      AssertionError: If ``x`` is not a scalar; if ``x`` is non-positive.
    """
    assert_scalar(x)
    if not x > 0:
        raise AssertionError(f"The argument must be positive, got {x}.")
