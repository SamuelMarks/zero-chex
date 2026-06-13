"""Scalar assertions."""


def assert_scalar(x):
    """Asserts that x is a scalar value.

    Args:
        x: The value to check.

    Returns:
        None

    Raises:
        AssertionError: If x is not a scalar (int, float, complex).
    """
    if not isinstance(x, (int, float, complex)):
        raise AssertionError("x must be a scalar")


def assert_scalar_in(x, min_, max_, included=True):
    """Asserts that x is within the specified range.

    Args:
        x: The scalar value to check.
        min_val: The lower bound of the range.
        max_val: The upper bound of the range.
        included: If True, the range is inclusive [min_val, max_val]. If False, the range is exclusive (min_val, max_val).

    Returns:
        None

    Raises:
        AssertionError: If x is outside the specified range.
    """
    if included:
        if x < min_ or x > max_:
            raise AssertionError("x must be in [%s, %s]" % (min_, max_))
    else:
        if x <= min_ or x >= max_:
            raise AssertionError("x must be in (%s, %s)" % (min_, max_))


def assert_scalar_negative(x):
    """Asserts that x is negative.

    Args:
        x: The scalar value to check.

    Returns:
        None

    Raises:
        AssertionError: If x is not strictly less than 0.
    """
    if x >= 0:
        raise AssertionError("x must be negative")


def assert_scalar_non_negative(x):
    """Asserts that x is non-negative.

    Args:
        x: The scalar value to check.

    Returns:
        None

    Raises:
        AssertionError: If x is strictly less than 0.
    """
    if x < 0:
        raise AssertionError("x must be non-negative")


def assert_scalar_positive(x):
    """Asserts that x is positive.

    Args:
        x: The scalar value to check.

    Returns:
        None

    Raises:
        AssertionError: If x is not strictly greater than 0.
    """
    if x <= 0:
        raise AssertionError("x must be positive")
