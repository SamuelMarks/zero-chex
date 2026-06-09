# ruff: noqa: F821, F403, F405, E402, E731, F841, E721

"""Module docstring."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        """Docstring."""

        user = None


"""Basic logic and device assertions."""

from typing import Any, Optional, Sequence
import unittest
import zero_jax as jax


def assert_equal(first: "Any", second: "Any") -> "NoneType":
    """Checks that the two objects are equal as determined by the `==` operator.

    Arrays with more than one element cannot be compared.
    Use ``assert_trees_all_close`` to compare arrays.

    Args:
      first: A first object.
      second: A second object.

    Raises:
      AssertionError: If not ``(first == second)``.
    """
    unittest.TestCase().assertEqual(first, second)


def assert_exactly_one_is_none(first: "Any", second: "Any") -> "NoneType":
    """Checks that one and only one of the arguments is `None`.

    Args:
      first: A first object.
      second: A second object.

    Raises:
      AssertionError: If ``(first is None) xor (second is None)`` is `False`.
    """
    if (first is None) == (second is None):
        raise AssertionError(
            f"One and exactly one of inputs should be `None`, got {first} and {second}."
        )


def assert_not_both_none(first: "Any", second: "Any") -> "NoneType":
    """Checks that at least one of the arguments is not `None`.

    Args:
      first: A first object.
      second: A second object.

    Raises:
      AssertionError: If ``(first is None) and (second is None)``.
    """
    if first is None and second is None:
        raise AssertionError(
            "At least one of the arguments must be different from `None`."
        )


def _num_devices_available(devtype: str, backend: Optional[str] = None) -> int:
    """Returns the number of available device of the given type."""
    devtype = devtype.lower()
    supported_types = ("cpu", "gpu", "tpu")
    if devtype not in supported_types:
        raise ValueError(
            f"Unknown device type '{devtype}' (expected one of {supported_types})."
        )
    return 1 if devtype == "cpu" else 0


def assert_devices_available(
    n: "int",
    devtype: "str",
    backend: "str | None" = None,
    not_less_than: "bool" = False,
) -> "NoneType":
    """Checks that `n` devices of a given type are available.

    Args:
      n: A required number of devices of the given type.
      devtype: A type of devices, one of ``{'cpu', 'gpu', 'tpu'}``.
      backend: A type of backend to use (uses Jax default if not provided).
      not_less_than: Whether to check if the number of devices is not less than
        `n`, instead of precise comparison.

    Raises:
      AssertionError: If number of available device of a given type is not equal
                      or less than `n`.
    """
    n_available = _num_devices_available(devtype, backend=backend)
    devs = jax.devices(backend)
    if not_less_than and n_available < n:
        raise AssertionError(
            f"Only {n_available} < {n} {devtype.upper()}s available in {devs}."
        )
    elif not not_less_than and n_available != n:
        raise AssertionError(f"No {n} {devtype.upper()}s available in {devs}.")


def assert_gpu_available(backend: "str | None" = None) -> "NoneType":
    """Checks that at least one GPU device is available.

    Args:
      backend: A type of backend to use (uses JAX default if not provided).

    Raises:
      AssertionError: If no GPU device available.
    """
    if not _num_devices_available("gpu", backend=backend):
        raise AssertionError(f"No GPU devices available in {[]}.")


def assert_tpu_available(backend: "str | None" = None) -> "NoneType":
    """Checks that at least one TPU device is available.

    Args:
      backend: A type of backend to use (uses JAX default if not provided).

    Raises:
      AssertionError: If no TPU device available.
    """
    if not _num_devices_available("tpu", backend=backend):
        raise AssertionError(f"No TPU devices available in {[]}.")


from zero_jax.numpy import broadcast_shapes


def assert_is_broadcastable(
    shape_a: "Sequence[int]", shape_b: "Sequence[int]"
) -> "NoneType":
    """Checks that an array of ``shape_a`` is broadcastable to one of ``shape_b``.

    Args:
      shape_a: A shape of the array to check.
      shape_b: A target shape after broadcasting.

    Raises:
      AssertionError: If ``shape_a`` is not broadcastable to ``shape_b``.
    """
    try:
        broadcast_shapes(shape_a, shape_b)
    except ValueError:
        raise AssertionError(
            f"Shape {shape_a} is not broadcastable to shape {shape_b}."
        )

    ndim_a = len(shape_a)
    ndim_b = len(shape_b)
    if ndim_a > ndim_b:
        raise AssertionError(
            f"Shape {shape_a} is not broadcastable to shape {shape_b}."
        )


def assert_is_divisible(numerator: "int", denominator: "int") -> "NoneType":
    """Checks that ``numerator`` is divisible by ``denominator``.

    Args:
      numerator: A numerator.
      denominator: A denominator.

    Raises:
      AssertionError: If ``numerator`` is not divisible by ``denominator``.
    """
    if numerator % denominator != 0:
        raise AssertionError(f"{numerator} is not divisible by {denominator}.")
