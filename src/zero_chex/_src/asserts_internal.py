"""Internal assertion helpers, error formatting, and path utilities for zero_chex."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from typing_extensions import ParamSpec

from zero_chex.assertions import _raise_chex_error
from zero_chex.devices import _num_devices_available
from zero_chex.mocks import _STATE, are_asserts_enabled
from zero_chex.trees import _format_path
from zero_chex.types import DictKey, EllipsisType, TDimMatcher, TShapeMatcher
from zero_chex.utils import get_err_regex

P = ParamSpec("P")
R = TypeVar("R")

ERR_PREFIX: str = "[Chex] Assertion"
DISABLE_ASSERTIONS: bool = False
TRACE_COUNTER: dict[str, int] = _STATE.trace_counts


def chex_assertion(
    assert_fn: Callable[P, R],
    jittable_assert_fn: Callable[P, R] | None = None,
) -> Callable[P, R]:
    """Wrap a host assertion function with optional jittable assertion fallback.

    Args:
        assert_fn: Primary host assertion callable.
        jittable_assert_fn: Optional jittable assertion callable for traced execution.

    Returns:
        Wrapped assertion callable returning the result of assert_fn.
    """
    _ = jittable_assert_fn
    return assert_fn


def format_shape_matcher(shape: TShapeMatcher) -> str:
    """Format a shape matcher sequence into a human-readable string representation.

    Args:
        shape: Sequence of dimension matchers (ints, sets, strings, ellipses, or None).

    Returns:
        Formatted string representation of the shape matcher.
    """
    parts: list[str] = []
    for d in shape:
        if isinstance(d, EllipsisType):
            parts.append("...")
        elif d is None:
            parts.append("*")
        else:
            parts.append(str(d))
    return f"({', '.join(parts)})"


def format_tree_path(path: tuple[DictKey, ...]) -> str:
    """Format a PyTree key path tuple into a slash-separated string representation.

    Args:
        path: Tuple of PyTree keys from root to leaf.

    Returns:
        Formatted path string.
    """
    return "/".join(str(p) for p in path) if path else _format_path(path)


def get_chexify_err_message(name: str, msg: str) -> str:
    """Construct a formatted chexify error message with assertion name and detail string.

    Args:
        name: Name of the chexified assertion function.
        msg: Error detail message.

    Returns:
        Formatted chexify error string.
    """
    return f"{ERR_PREFIX} {name} failed: {msg}"


def num_devices_available(devtype: str, backend: str | None = None) -> int:
    """Return the number of available devices matching devtype and optional backend.

    Args:
        devtype: Device platform name (e.g. 'cpu', 'gpu', 'tpu').
        backend: Optional backend filter name.

    Returns:
        Count of available matching devices.
    """
    return _num_devices_available(devtype, backend=backend)


__all__ = [
    "DISABLE_ASSERTIONS",
    "ERR_PREFIX",
    "TDimMatcher",
    "TRACE_COUNTER",
    "TShapeMatcher",
    "_raise_chex_error",
    "are_asserts_enabled",
    "chex_assertion",
    "format_shape_matcher",
    "format_tree_path",
    "get_chexify_err_message",
    "get_err_regex",
    "num_devices_available",
]
