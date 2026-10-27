"""Internal assertion helpers, runtime state, error formatting, and path utilities for zero_chex."""

from __future__ import annotations

import functools
import math
import re
import threading
from collections.abc import Callable, Sequence
from typing import Generic, NoReturn, TypeVar

import zero_jax
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_ir import DType, TensorShape, TensorSpec
from typing_extensions import ParamSpec

from zero_chex.types import (
    ArrayDType,
    ArrayProtocol,
    CompilerTensor,
    DictKey,
    DTypeAttrProtocol,
    DTypeLikeProtocol,
    EllipsisType,
    JaxNdarray,
    MaskedNode,
    NamedTuplePyTreeProtocol,
    NestedScalarSequence,
    PyTree,
    Scalar,
    SerializedArray,
    ShapeOnlyProtocol,
    SharedMemoryArray,
    TDimMatcher,
    ToListProtocol,
    TShapeMatcher,
    VariableLeafProtocol,
)

P = ParamSpec("P")
R = TypeVar("R")

ERR_PREFIX: str = "[Chex] Assertion"
DISABLE_ASSERTIONS: bool = False

VALID_DTYPE_NAMES: frozenset[str] = frozenset(
    {
        "bool",
        "bool_",
        "int",
        "int4",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint4",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex",
        "complex64",
        "complex128",
        "str",
    }
)


class AssertionRuntimeState:
    """Thread-safe container for global assertion state, trace counters, and device configs."""

    def __init__(self) -> None:
        """Initialize the default assertion runtime state."""
        self.enabled: bool = True
        self.in_chexify_context: int = 0
        self.lock: threading.Lock = threading.Lock()
        self.trace_counts: dict[str, int] = {}
        self.trace_seen_signatures: dict[str, set[str]] = {}
        self._next_fn_token: int = 0
        self.n_cpu_devices: int = 1
        self.n_gpu_devices: int = 1
        self.n_tpu_devices: int = 1
        self.allowed_backends: frozenset[str] | None = None
        self.forbidden_backends: frozenset[str] | None = None
        self.jit_disabled: bool = False
        self.pmap_disabled: bool = False
        self.fake_pmap_jit_result: bool = False
        self.fake_pmap_ignore_groups: bool = False
        self.fake_pmap_parallel_axis: bool = False
        self.pending_chexify_errors: list[AssertionError] = []
        self.pending_chexify_queues: list[list[AssertionError]] = []

    def allocate_fn_key(self, fn_name: str) -> str:
        """Allocate a collision-free trace key independent of garbage-collected object ids.

        Args:
            fn_name: Function name for diagnostics.

        Returns:
            Unique string key for tracking trace counts.
        """
        with self.lock:
            self._next_fn_token += 1
            return f"{fn_name}#{self._next_fn_token}"


RUNTIME_STATE = AssertionRuntimeState()
TRACE_COUNTER: dict[str, int] = RUNTIME_STATE.trace_counts


def are_asserts_enabled() -> bool:
    """Return whether Chex assertions are currently enabled globally.

    Returns:
        True if assertions are enabled, False if disabled via disable_asserts() or DISABLE_ASSERTIONS.
    """
    return RUNTIME_STATE.enabled and not DISABLE_ASSERTIONS


def disable_asserts() -> None:
    """Disable all Chex assertions globally and synchronize DISABLE_ASSERTIONS."""
    global DISABLE_ASSERTIONS
    DISABLE_ASSERTIONS = True
    RUNTIME_STATE.enabled = False


def enable_asserts() -> None:
    """Enable all Chex assertions globally and synchronize DISABLE_ASSERTIONS."""
    global DISABLE_ASSERTIONS
    DISABLE_ASSERTIONS = False
    RUNTIME_STATE.enabled = True


def clear_trace_counter() -> None:
    """Reset all recorded function trace counters, JAX caches, and pending chexify errors."""
    zero_jax.clear_caches()
    with RUNTIME_STATE.lock:
        RUNTIME_STATE.trace_counts.clear()
        RUNTIME_STATE.trace_seen_signatures.clear()
        for q in RUNTIME_STATE.pending_chexify_queues:
            q.clear()
        RUNTIME_STATE.pending_chexify_queues.clear()
        RUNTIME_STATE.pending_chexify_errors.clear()
        RUNTIME_STATE.jit_disabled = False
        RUNTIME_STATE.pmap_disabled = False
        RUNTIME_STATE.fake_pmap_jit_result = False
        RUNTIME_STATE.fake_pmap_ignore_groups = False
        RUNTIME_STATE.fake_pmap_parallel_axis = False


def set_n_cpu_devices(n: int | None = None) -> None:
    """Configure the number of simulated host CPU devices and synchronize with zero_jax.

    Args:
        n: Number of CPU devices to simulate (defaults to 1 if None).

    Raises:
        ValueError: If n is provided and is less than 1.
    """
    if n is not None and n < 1:
        raise ValueError(f"Number of CPU devices must be >= 1, got {n}.")
    resolved = n if n is not None else 1
    RUNTIME_STATE.n_cpu_devices = resolved
    zero_jax.set_host_device_count(resolved)


def get_num_devices_available(platform: str) -> int:
    """Return the simulated number of available devices for a given platform.

    Args:
        platform: Hardware platform identifier ('cpu', 'gpu', or 'tpu').

    Returns:
        Number of simulated devices available on that platform.

    Raises:
        ValueError: If platform is not one of 'cpu', 'gpu', or 'tpu'.
    """
    if platform == "cpu":
        return RUNTIME_STATE.n_cpu_devices
    if platform == "gpu":
        return RUNTIME_STATE.n_gpu_devices
    if platform == "tpu":
        return RUNTIME_STATE.n_tpu_devices
    raise ValueError(f"Unknown device type: {platform}")


def check_backend_permitted(backend: str | None, default_platform: str) -> None:
    """Validate that a target backend/platform is permitted under active restrict_backends.

    Args:
        backend: Explicit backend name or None.
        default_platform: Fallback platform name if backend is None.

    Raises:
        ValueError: If backend is not a recognized platform ('cpu', 'gpu', 'tpu').
        AssertionError: If the target backend is forbidden or excluded by restrict_backends.
    """
    target = backend if backend is not None else default_platform
    if target not in ("cpu", "gpu", "tpu"):
        raise ValueError(f"Unknown backend: {target}")
    if RUNTIME_STATE.forbidden_backends is not None and target in RUNTIME_STATE.forbidden_backends:
        raise AssertionError(f"Backend {target!r} is forbidden by restrict_backends.")
    if RUNTIME_STATE.allowed_backends is not None and target not in RUNTIME_STATE.allowed_backends:
        raise AssertionError(f"Backend {target!r} is not in allowed backends {RUNTIME_STATE.allowed_backends!r}.")


def num_devices_available(devtype: str, backend: str | None = None) -> int:
    """Return the number of available devices matching devtype and optional backend.

    Args:
        devtype: Device platform name (e.g. 'cpu', 'gpu', 'tpu').
        backend: Optional backend filter name.

    Returns:
        Count of available matching devices.
    """
    if backend is not None:
        check_backend_permitted(backend, devtype)
    return get_num_devices_available(devtype)


def raise_chex_error(
    default_msg: str,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> NoReturn:
    """Format and raise a Chex assertion error with optional custom message formatting.

    Args:
        default_msg: The standard diagnostic message generated by the assertion.
        custom_message: Optional user-supplied message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include default_msg alongside custom_message.
        exception_type: The Exception class to instantiate and raise.

    Raises:
        Exception: An instance of exception_type with the formatted message.
    """
    if custom_message is not None:
        formatted_custom = (
            custom_message.format(*custom_message_format_vars) if custom_message_format_vars else custom_message
        )
        final_msg = (
            f"{default_msg} [{formatted_custom}]"
            if include_default_message
            else f"[Chex] Assertion failed: {formatted_custom}"
        )
    else:
        final_msg = default_msg
    raise exception_type(final_msg)


class ChexAssertionWrapper(Generic[P, R]):
    """Typed callable wrapper for a Chex assertion function with optional jittable fallback."""

    def __init__(
        self,
        assert_fn: Callable[P, R],
        jittable_assert_fn: Callable[P, R] | None = None,
    ) -> None:
        """Initialize a ChexAssertionWrapper.

        Args:
            assert_fn: Primary host assertion callable.
            jittable_assert_fn: Optional jittable assertion callable for traced execution.
        """
        self._assert_fn = assert_fn
        self.jittable_assert_fn: Callable[P, R] | None = jittable_assert_fn
        self.__wrapped__ = assert_fn
        functools.update_wrapper(self, assert_fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        """Execute the wrapped assertion if assertions are enabled.

        Args:
            *args: Positional arguments forwarded to the assertion function.
            **kwargs: Keyword arguments forwarded to the assertion function.

        Returns:
            The result of the assertion function if enabled, or None if disabled.
        """
        if not are_asserts_enabled():
            return None
        if RUNTIME_STATE.in_chexify_context > 0 and self.jittable_assert_fn is not None:
            return self.jittable_assert_fn(*args, **kwargs)
        return self._assert_fn(*args, **kwargs)


def chex_assertion(
    assert_fn: Callable[P, R],
    jittable_assert_fn: Callable[P, R] | None = None,
) -> ChexAssertionWrapper[P, R]:
    """Wrap a host assertion function with optional jittable assertion fallback.

    Args:
        assert_fn: Primary host assertion callable.
        jittable_assert_fn: Optional jittable assertion callable for traced execution.

    Returns:
        Wrapped assertion callable respecting are_asserts_enabled() and chexify context.
    """
    return ChexAssertionWrapper(assert_fn, jittable_assert_fn=jittable_assert_fn)


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
    return "/".join(str(p) for p in path) if path else "root"


def get_chexify_err_message(name: str, msg: str) -> str:
    """Construct a formatted chexify error message with assertion name and detail string.

    Args:
        name: Name of the chexified assertion function.
        msg: Error detail message.

    Returns:
        Formatted chexify error string.
    """
    return f"{ERR_PREFIX} {name} failed: {msg}"


def get_err_regex(message: str) -> str:
    """Return a regular expression pattern matching the provided error message.

    Args:
        message: The error message string.

    Returns:
        The escaped error message regex string.
    """
    return re.escape(message)


def _flatten_nested_scalars(data: NestedScalarSequence | Scalar) -> list[Scalar]:
    """Recursively flatten a scalar or nested sequence of scalars into a flat list.

    Args:
        data: Scalar or nested sequence of scalars.

    Returns:
        Flat list of scalars.
    """
    if isinstance(data, (int, float, complex, bool)):
        return [data]
    flat_items: list[Scalar] = []
    for elem in data:
        flat_items.extend(_flatten_nested_scalars(elem))
    return flat_items


def extract_flat_scalars(leaf: PyTree) -> list[Scalar]:
    """Extract a flat list of scalar values from a leaf array, Variable, or scalar.

    Args:
        leaf: Array, Variable, or scalar leaf value.

    Returns:
        Flat list of numeric scalars.
    """
    if isinstance(leaf, VariableLeafProtocol):
        return extract_flat_scalars(leaf.value)
    if isinstance(leaf, CompilerTensor):
        if leaf.ndim == 0:
            return [leaf.item()]
        out: list[Scalar] = []
        for sub_tensor in leaf:
            out.extend(extract_flat_scalars(sub_tensor))
        return out
    if isinstance(leaf, (JaxNdarray, SerializedArray)):
        raw_list = leaf.tolist()
        if isinstance(raw_list, (int, float, complex, bool)):
            return [raw_list]
        assert isinstance(raw_list, list)
        return _flatten_nested_scalars(raw_list)
    if isinstance(leaf, (int, float, complex, bool)):
        return [leaf]
    if isinstance(leaf, ToListProtocol):
        return list(leaf.to_list())
    return []


def has_valid_shape_tuple(val: ShapeOnlyProtocol) -> bool:
    """Check whether a ShapeOnlyProtocol object has a valid shape tuple or TensorShape.

    Args:
        val: Candidate object exposing `.shape`.

    Returns:
        True if `val.shape` is a TensorShape or a tuple of non-negative integers.
    """
    raw_shape = val.shape
    if isinstance(raw_shape, TensorShape):
        return True
    return all(not isinstance(d, bool) and d >= 0 for d in raw_shape)


def get_leaf_shape(x: PyTree) -> tuple[int, ...]:
    """Extract the shape tuple of a scalar, array, TensorSpec, Variable, or nested sequence.

    Args:
        x: The value whose shape should be determined.

    Returns:
        A tuple of integer dimensions.

    Raises:
        AssertionError: If a nested sequence has ragged sub-shapes.
    """
    if isinstance(x, VariableLeafProtocol):
        return get_leaf_shape(x.value)
    if isinstance(x, SharedMemoryArray):
        return tuple(x.metadata.shape)
    if isinstance(x, TensorSpec):
        return x.static_shape
    if isinstance(x, (ArrayProtocol, SerializedArray)):
        return tuple(x.shape)
    if isinstance(x, ShapeOnlyProtocol) and has_valid_shape_tuple(x):
        raw_s = x.shape
        return raw_s.static_shape if isinstance(raw_s, TensorShape) else tuple(int(d) for d in raw_s)
    if isinstance(x, (list, tuple)) and not isinstance(x, (NamedTuplePyTreeProtocol, MaskedNode)):
        if not x:
            return (0,)
        first_sub = get_leaf_shape(x[0])
        for elem in x[1:]:
            if get_leaf_shape(elem) != first_sub:
                raise AssertionError(f"Ragged sequence has inconsistent sub-shapes: {[get_leaf_shape(e) for e in x]}")
        return (len(x),) + first_sub
    return ()


def get_leaf_size(x: PyTree) -> int:
    """Extract the total element count of a scalar, array, Variable, or nested sequence.

    Args:
        x: The value whose size should be determined.

    Returns:
        Total number of elements.
    """
    return math.prod(get_leaf_shape(x))


def normalize_dtype_str(dt: ArrayDType) -> str:
    """Normalize a dtype specification or Python type to a canonical lowercase string.

    Args:
        dt: A DType enum, CompilerDType, DTypeLikeProtocol, Python type, or string name.

    Returns:
        Canonical dtype name string (e.g., 'float32', 'int', 'bool').

    Raises:
        ValueError: If dt is an unrecognized DTypeLikeProtocol name.
    """
    if isinstance(dt, (DType, CompilerDType)):
        return str(dt.value).lower()
    if isinstance(dt, DTypeLikeProtocol):
        cand = dt.name.lower()
        if cand.startswith("dtype."):
            cand = cand.split(".", 1)[1]
        if cand not in VALID_DTYPE_NAMES:
            raise ValueError(f"Unrecognized dtype name: {dt.name!r}")
        return "bool" if cand == "bool_" else cand
    if isinstance(dt, type):
        if dt is bool:
            return "bool"
        if dt is int:
            return "int"
        if dt is float:
            return "float"
        if dt is complex:
            return "complex"
        return dt.__name__.lower()
    raw = str(dt).lower()
    if raw.startswith("dtype."):
        raw = raw.split(".", 1)[1]
    return "bool" if raw == "bool_" else raw


def get_leaf_type(x: PyTree) -> ArrayDType:
    """Extract the dtype or Python scalar type of an input value.

    Args:
        x: Scalar, array, Variable, or sequence to inspect.

    Returns:
        The extracted ArrayDType.
    """
    if isinstance(x, VariableLeafProtocol):
        return get_leaf_type(x.value)
    if isinstance(x, SharedMemoryArray):
        return str(x.metadata.dtype) if x.metadata.dtype is not None else "float32"
    if isinstance(x, TensorSpec):
        return x.dtype
    if isinstance(x, (ArrayProtocol, SerializedArray, DTypeAttrProtocol)):
        return x.dtype
    if isinstance(x, (int, float, complex, bool)):
        return type(x)
    return str(type(x).__name__)


_FLOAT_CATEGORY_DTYPES: frozenset[str] = frozenset({"float", "float16", "float32", "float64", "bfloat16"})
_INT_CATEGORY_DTYPES: frozenset[str] = frozenset({"int", "int4", "int8", "int16", "int32", "int64"})
_UINT_CATEGORY_DTYPES: frozenset[str] = frozenset({"uint", "uint4", "uint8", "uint16", "uint32", "uint64"})
_COMPLEX_CATEGORY_DTYPES: frozenset[str] = frozenset({"complex", "complex64", "complex128"})
_BOOL_CATEGORY_DTYPES: frozenset[str] = frozenset({"bool", "bool_"})


def dtypes_match(actual: ArrayDType, expected: ArrayDType) -> bool:
    """Compare an actual dtype and an expected dtype or Python scalar type category.

    Args:
        actual: Actual dtype extracted from an input value.
        expected: Expected dtype or Python type category.

    Returns:
        True if actual matches expected (including broad category matching like int32 -> int or bfloat16 -> float).
    """
    norm_a = normalize_dtype_str(actual)
    norm_e = normalize_dtype_str(expected)
    if norm_a == norm_e:
        return True
    for category, members in (
        ("float", _FLOAT_CATEGORY_DTYPES),
        ("int", _INT_CATEGORY_DTYPES),
        ("uint", _UINT_CATEGORY_DTYPES),
        ("complex", _COMPLEX_CATEGORY_DTYPES),
        ("bool", _BOOL_CATEGORY_DTYPES),
    ):
        if (norm_e == category and norm_a in members) or (norm_a == category and norm_e in members):
            return True
    return False


__all__ = [
    "AssertionRuntimeState",
    "ChexAssertionWrapper",
    "DISABLE_ASSERTIONS",
    "ERR_PREFIX",
    "RUNTIME_STATE",
    "TDimMatcher",
    "TRACE_COUNTER",
    "TShapeMatcher",
    "VALID_DTYPE_NAMES",
    "are_asserts_enabled",
    "check_backend_permitted",
    "chex_assertion",
    "clear_trace_counter",
    "disable_asserts",
    "dtypes_match",
    "enable_asserts",
    "extract_flat_scalars",
    "format_shape_matcher",
    "format_tree_path",
    "get_chexify_err_message",
    "get_err_regex",
    "get_leaf_shape",
    "get_leaf_size",
    "get_leaf_type",
    "get_num_devices_available",
    "has_valid_shape_tuple",
    "normalize_dtype_str",
    "num_devices_available",
    "raise_chex_error",
    "set_n_cpu_devices",
]
