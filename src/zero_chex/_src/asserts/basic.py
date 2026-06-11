"""Basic assertions and utilities."""

import ml_switcheroo.shape


def assert_equal(a, b):
    """Asserts that two values are equal.

    Args:
        a: The first value to compare.
        b: The second value to compare.

    Returns:
        None

    Raises:
        AssertionError: If a and b are not equal.
    """
    if a != b:
        raise AssertionError("Values are not equal")


def assert_exactly_one_is_none(*args):
    """Asserts that exactly one of the provided arguments is None.

    Args:
        *args: Variable number of arguments to check.

    Returns:
        None

    Raises:
        AssertionError: If the number of None arguments is not exactly one.
    """
    num_nones = sum(1 for a in args if a is None)
    if num_nones != 1:
        raise AssertionError("One and exactly one argument must be None")


def assert_not_both_none(a, b):
    """Asserts that at least one of the two arguments is not None.

    Args:
        a: The first argument to check.
        b: The second argument to check.

    Returns:
        None

    Raises:
        AssertionError: If both a and b are None.
    """
    if a is None and b is None:
        raise AssertionError("At least one argument must be non-None")


def _num_devices_available(platform):
    """Helper function to get the number of available devices for a given platform.

    Args:
        platform: The platform type as a string (e.g., 'cpu', 'gpu', 'tpu').

    Returns:
        The number of available devices (always 1 for known platforms in this mock).

    Raises:
        ValueError: If the platform is unknown.
    """
    if platform not in ["cpu", "gpu", "tpu"]:
        raise ValueError("Unknown device type")
    return 1


def assert_devices_available(n, platform="cpu", not_less_than=False):
    """Asserts that a specific number of devices are available for a given platform.

    Args:
        n: The expected number of devices.
        platform: The platform type as a string (default: "cpu").
        not_less_than: If True, asserts that at least `n` devices are available.
                       If False, asserts exactly `n` devices are available.

    Returns:
        None

    Raises:
        AssertionError: If the availability condition is not met.
    """
    av = _num_devices_available(platform)
    if not_less_than:
        if av < n:
            raise AssertionError("Only %d %s devices available" % (av, platform))
    elif av != n:
        raise AssertionError(
            "Exactly %d %s devices available, not %d" % (av, platform, n)
        )


def assert_gpu_available():
    """Asserts that at least one GPU device is available.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If no GPU devices are available.
    """
    if _num_devices_available("gpu") == 0:
        raise AssertionError("No GPU devices available")


def assert_tpu_available():
    """Asserts that at least one TPU device is available.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If no TPU devices are available.
    """
    if _num_devices_available("tpu") == 0:
        raise AssertionError("No TPU devices available")


def assert_is_broadcastable(shape1, shape2):
    """Asserts that two shapes are broadcastable with each other.

    Args:
        shape1: The first shape to check.
        shape2: The second shape to check.

    Returns:
        None

    Raises:
        AssertionError: If the shapes cannot be broadcasted together.
    """
    try:
        ml_switcheroo.shape.broadcast_shapes(shape1, shape2)
    except ValueError:
        raise AssertionError(
            "Shape %s is not broadcastable with shape %s" % (shape1, shape2)
        )


def assert_is_divisible(a, b):
    """Asserts that the first argument is divisible by the second argument.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        None

    Raises:
        AssertionError: If `a` modulo `b` is not 0.
    """
    if a % b != 0:
        raise AssertionError("%d is not divisible by %d" % (a, b))
