import numpy as np


def assert_equal(a, b):
    if a != b:
        raise AssertionError("Values are not equal")


def assert_exactly_one_is_none(*args):
    num_nones = sum(1 for a in args if a is None)
    if num_nones != 1:
        raise AssertionError("One and exactly one argument must be None")


def assert_not_both_none(a, b):
    if a is None and b is None:
        raise AssertionError("At least one argument must be non-None")


def _num_devices_available(platform):
    if platform not in ["cpu", "gpu", "tpu"]:
        raise ValueError("Unknown device type")
    return 1


def assert_devices_available(n, platform="cpu", not_less_than=False):
    av = _num_devices_available(platform)
    if not_less_than:
        if av < n:
            raise AssertionError("Only %d %s devices available" % (av, platform))
    elif av != n:
        raise AssertionError(
            "Exactly %d %s devices available, not %d" % (av, platform, n)
        )


def assert_gpu_available():
    if _num_devices_available("gpu") == 0:
        raise AssertionError("No GPU devices available")


def assert_tpu_available():
    if _num_devices_available("tpu") == 0:
        raise AssertionError("No TPU devices available")


def assert_is_broadcastable(shape1, shape2):
    try:
        np.broadcast_shapes(shape1, shape2)
    except ValueError:
        raise AssertionError(
            "Shape %s is not broadcastable with shape %s" % (shape1, shape2)
        )


def assert_is_divisible(a, b):
    if a % b != 0:
        raise AssertionError("%d is not divisible by %d" % (a, b))
