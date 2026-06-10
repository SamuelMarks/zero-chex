"""Module docstring."""


def assert_scalar(x):
    """Docstring."""
    if not isinstance(x, (int, float, complex)):
        raise AssertionError("x must be a scalar")


def assert_scalar_in(x, min_val, max_val, included=True):
    """Docstring."""
    if included:
        if x < min_val or x > max_val:
            raise AssertionError("x must be in [%s, %s]" % (min_val, max_val))
    else:
        if x <= min_val or x >= max_val:
            raise AssertionError("x must be in (%s, %s)" % (min_val, max_val))


def assert_scalar_negative(x):
    """Docstring."""
    if x >= 0:
        raise AssertionError("x must be negative")


def assert_scalar_non_negative(x):
    """Docstring."""
    if x < 0:
        raise AssertionError("x must be non-negative")


def assert_scalar_positive(x):
    """Docstring."""
    if x <= 0:
        raise AssertionError("x must be positive")
