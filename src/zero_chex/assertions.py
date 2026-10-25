"""Shape, rank, axis dimension, size, dtype, broadcast, and divisibility assertions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn, Union

from ml_switcheroo_ir import DType, ShapeMismatchError, TensorShape, TensorSpec, broadcast_shapes
from typing_extensions import TypeGuard

from zero_chex._src.tree_util import tree_leaves
from zero_chex.mocks import are_asserts_enabled
from zero_chex.types import (
    _VALID_DTYPE_NAMES,
    ArrayDType,
    ArrayProtocol,
    ArrayTree,
    CompilerDType,
    DTypeAttrProtocol,
    DTypeLikeProtocol,
    EllipsisType,
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    PyTree,
    Scalar,
    ShapeOnlyProtocol,
    SharedMemoryArray,
    StateMappingProtocol,
    TDimMatcher,
    TShapeMatcher,
    VariableLeafProtocol,
    _has_valid_shape_tuple,
)


def _raise_chex_error(
    default_msg: str,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> NoReturn:
    """Format and raise a Chex assertion error with optional custom message and exception type.

    Args:
        default_msg: The standard failure message produced by the assertion.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include default_msg alongside custom_message.
        exception_type: Exception class to instantiate and raise.

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


def _get_shape(x: PyTree) -> tuple[int, ...]:
    """Extract the shape tuple of an array-like object, sequence, or scalar.

    Args:
        x: The array, nested sequence, or scalar to inspect.

    Returns:
        A tuple of dimension integers representing the shape of x.

    Raises:
        AssertionError: If x is a ragged nested sequence with inconsistent sub-shapes.
    """
    if isinstance(x, VariableLeafProtocol):
        return _get_shape(x.value)
    if isinstance(x, SharedMemoryArray):
        return tuple(x.metadata.shape)
    if isinstance(x, TensorSpec):
        return x.static_shape
    if isinstance(x, ArrayProtocol):
        return tuple(x.shape)
    if isinstance(x, ShapeOnlyProtocol) and _has_valid_shape_tuple(x):
        raw_s = x.shape
        if isinstance(raw_s, TensorShape):
            return raw_s.static_shape
        return tuple(int(d) for d in raw_s)
    if isinstance(x, (list, tuple)) and not isinstance(x, NamedTuplePyTreeProtocol):
        if not x:
            return (0,)
        first_sub = _get_shape(x[0])
        for elem in x[1:]:
            if _get_shape(elem) != first_sub:
                raise AssertionError(f"Ragged sequence has inconsistent sub-shapes: {[_get_shape(e) for e in x]}")
        return (len(x),) + first_sub
    return ()


def _get_rank(x: PyTree) -> int:
    """Compute the rank (number of dimensions) of an array-like object.

    Args:
        x: The array, sequence, or scalar to inspect.

    Returns:
        Integer rank of x.
    """
    return len(_get_shape(x))


def _get_size(x: PyTree) -> int:
    """Compute the total number of scalar elements in an array-like object.

    Args:
        x: The array, sequence, or scalar to inspect.

    Returns:
        Product of all dimensions in the shape of x.
    """
    return math.prod(_get_shape(x))


def _normalize_dtype(dt: ArrayDType) -> str:
    """Normalize a DType enum, Python scalar type, or string into a canonical dtype label.

    Args:
        dt: Data type specification to normalize.

    Returns:
        Canonical lowercase dtype category or name string.

    Raises:
        ValueError: If dt is a DTypeLikeProtocol whose name is not a valid dtype name.
    """
    if isinstance(dt, (DType, CompilerDType)):
        return str(dt.value).lower()
    if isinstance(dt, DTypeLikeProtocol):
        cand = dt.name.lower()
        if cand.startswith("dtype."):
            cand = cand.split(".", 1)[1]
        if cand not in _VALID_DTYPE_NAMES:
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


def _get_type(x: PyTree) -> ArrayDType:
    """Retrieve the raw data type of an array or Python scalar.

    Args:
        x: The array or scalar to inspect.

    Returns:
        The dtype attribute of x if present, or its Python type.
    """
    if isinstance(x, VariableLeafProtocol):
        return _get_type(x.value)
    if isinstance(x, SharedMemoryArray):
        return str(x.metadata.dtype) if x.metadata.dtype is not None else "float32"
    if isinstance(x, TensorSpec):
        return x.dtype
    if isinstance(x, (ArrayProtocol, DTypeAttrProtocol)):
        return x.dtype
    if isinstance(x, (int, float, complex, bool)):
        return type(x)
    return str(type(x).__name__)


def _dtypes_match(actual: ArrayDType, expected: ArrayDType) -> bool:
    """Compare an actual dtype against an expected dtype specification.

    Args:
        actual: The actual dtype extracted from an input value.
        expected: The expected dtype specification.

    Returns:
        True if actual matches expected under Chex dtype promotion/category rules.
    """
    norm_a = _normalize_dtype(actual)
    norm_e = _normalize_dtype(expected)
    if norm_a == norm_e:
        return True
    if norm_e in ("int", "float", "complex", "bool") and norm_a.startswith(norm_e):
        return True
    if norm_a in ("int", "float", "complex", "bool") and norm_e.startswith(norm_a):
        return True
    return False


def _is_pytree_container(inputs: ArrayTree) -> bool:
    """Determine whether inputs is a PyTree container of multiple array/scalar leaves.

    Args:
        inputs: Input argument passed to assert_rank, assert_shape, assert_size, or assert_type.

    Returns:
        True if inputs is a Mapping, custom PyTree container protocol, or list/tuple of arrays.
    """
    if isinstance(inputs, (Mapping, MappableDataclassProtocol, NamedTuplePyTreeProtocol, StateMappingProtocol)):
        return True
    if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
        return all(
            isinstance(
                item,
                (
                    ArrayProtocol,
                    VariableLeafProtocol,
                    ShapeOnlyProtocol,
                    TensorSpec,
                    Mapping,
                    MappableDataclassProtocol,
                    NamedTuplePyTreeProtocol,
                    StateMappingProtocol,
                ),
            )
            for item in inputs
        )
    return False


def assert_equal(
    first: PyTree,
    second: PyTree,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that two objects or values are equal.

    Args:
        first: The first value to compare.
        second: The second value to compare.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If assertions are enabled and first != second.
    """
    if not are_asserts_enabled():
        return
    if first != second:
        _raise_chex_error(
            f"Assertion failed: {first!r} != {second!r}",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_exactly_one_is_none(
    first: PyTree,
    second: PyTree,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that one and only one of the two arguments is None.

    Args:
        first: The first argument to check.
        second: The second argument to check.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If both arguments are None or neither is None.
    """
    if not are_asserts_enabled():
        return
    if (first is None) == (second is None):
        _raise_chex_error(
            f"One and exactly one of ({first!r}, {second!r}) must be None.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_not_both_none(
    first: PyTree,
    second: PyTree,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that at least one of the two arguments is not None.

    Args:
        first: The first argument to check.
        second: The second argument to check.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If both first and second are None.
    """
    if not are_asserts_enabled():
        return
    if first is None and second is None:
        _raise_chex_error(
            "At least one of the arguments must be different from None.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_is_broadcastable(
    shape_a: Sequence[int],
    shape_b: Sequence[int],
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that two shape tuples are compatible under NumPy/IR broadcasting rules.

    Args:
        shape_a: First shape sequence of integers.
        shape_b: Second shape sequence of integers.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If shape_a and shape_b cannot be broadcast together.
    """
    if not are_asserts_enabled():
        return
    try:
        broadcast_shapes(tuple(shape_a), tuple(shape_b))
    except (ShapeMismatchError, ValueError):
        _raise_chex_error(
            f"Shapes {shape_a!r} and {shape_b!r} are not broadcastable.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_is_divisible(
    numerator: int,
    denominator: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that numerator is an exact integer multiple of denominator.

    Args:
        numerator: Dividend integer.
        denominator: Divisor integer.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If denominator is 0 or numerator % denominator != 0.
    """
    if not are_asserts_enabled():
        return
    if denominator == 0 or numerator % denominator != 0:
        _raise_chex_error(
            f"{numerator} is not divisible by {denominator}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_axis_dimension_comparator(
    tensor: ArrayTree,
    axis: int,
    pass_fn: Callable[[int], bool],
    error_string: str = "",
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that a specific axis of tensor satisfies a predicate pass_fn.

    Args:
        tensor: The array-like object to inspect.
        axis: The axis index (supports negative indexing).
        pass_fn: Predicate callable taking the integer dimension size and returning bool.
        error_string: Optional description included in the error message.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If axis is out of bounds or pass_fn(shape[axis]) returns False.
    """
    if not are_asserts_enabled():
        return
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        _raise_chex_error(
            f"Axis {axis} out of bounds for tensor of rank {len(shape)}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )
    dim = shape[axis]
    if not pass_fn(dim):
        detail = f" ({error_string})" if error_string else ""
        _raise_chex_error(
            f"Expected tensor to have dimension{detail} on axis {axis} but got {dim}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_axis_dimension(
    tensor: ArrayTree,
    axis: int,
    expected: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that a specific axis of tensor has dimension equal to expected.

    Args:
        tensor: The array-like object to inspect.
        axis: The axis index to check.
        expected: The expected integer size of the axis.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If axis is out of bounds or shape[axis] != expected.
    """
    assert_axis_dimension_comparator(
        tensor,
        axis,
        lambda dim: dim == expected,
        error_string=f"equal to {expected}",
        custom_message=custom_message,
        custom_message_format_vars=custom_message_format_vars,
        include_default_message=include_default_message,
        exception_type=exception_type,
    )


def assert_axis_dimension_gt(
    tensor: ArrayTree,
    axis: int,
    val: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that a specific axis of tensor has dimension strictly greater than val.

    Args:
        tensor: The array-like object to inspect.
        axis: The axis index to check.
        val: Lower bound (exclusive) for the axis dimension.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If axis is out of bounds or shape[axis] <= val.
    """
    assert_axis_dimension_comparator(
        tensor,
        axis,
        lambda dim: dim > val,
        error_string=f"greater than {val}",
        custom_message=custom_message,
        custom_message_format_vars=custom_message_format_vars,
        include_default_message=include_default_message,
        exception_type=exception_type,
    )


def assert_axis_dimension_gteq(
    tensor: ArrayTree,
    axis: int,
    val: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that a specific axis of tensor has dimension greater than or equal to val.

    Args:
        tensor: The array-like object to inspect.
        axis: The axis index to check.
        val: Lower bound (inclusive) for the axis dimension.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If axis is out of bounds or shape[axis] < val.
    """
    assert_axis_dimension_comparator(
        tensor,
        axis,
        lambda dim: dim >= val,
        error_string=f"greater than or equal to {val}",
        custom_message=custom_message,
        custom_message_format_vars=custom_message_format_vars,
        include_default_message=include_default_message,
        exception_type=exception_type,
    )


def assert_axis_dimension_lt(
    tensor: ArrayTree,
    axis: int,
    val: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that a specific axis of tensor has dimension strictly less than val.

    Args:
        tensor: The array-like object to inspect.
        axis: The axis index to check.
        val: Upper bound (exclusive) for the axis dimension.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If axis is out of bounds or shape[axis] >= val.
    """
    assert_axis_dimension_comparator(
        tensor,
        axis,
        lambda dim: dim < val,
        error_string=f"less than {val}",
        custom_message=custom_message,
        custom_message_format_vars=custom_message_format_vars,
        include_default_message=include_default_message,
        exception_type=exception_type,
    )


def assert_axis_dimension_lteq(
    tensor: ArrayTree,
    axis: int,
    val: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that a specific axis of tensor has dimension less than or equal to val.

    Args:
        tensor: The array-like object to inspect.
        axis: The axis index to check.
        val: Upper bound (inclusive) for the axis dimension.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If axis is out of bounds or shape[axis] > val.
    """
    assert_axis_dimension_comparator(
        tensor,
        axis,
        lambda dim: dim <= val,
        error_string=f"less than or equal to {val}",
        custom_message=custom_message,
        custom_message_format_vars=custom_message_format_vars,
        include_default_message=include_default_message,
        exception_type=exception_type,
    )


def assert_equal_rank(
    inputs: Sequence[ArrayTree] | ArrayTree,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that all arrays in inputs have the same rank.

    Args:
        inputs: Sequence of array-like objects to compare.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If inputs is not a list or tuple.
        Exception: If any array has a rank differing from the first array.
    """
    if not are_asserts_enabled():
        return
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of arrays.")
    if not inputs:
        return
    ref_rank = _get_rank(inputs[0])
    for item in inputs[1:]:
        if _get_rank(item) != ref_rank:
            _raise_chex_error(
                f"Arrays have different ranks: {[_get_rank(t) for t in inputs]}.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )


def assert_equal_shape(
    inputs: Sequence[ArrayTree] | ArrayTree,
    dims: int | Sequence[int] | None = None,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that all arrays in inputs have identical shape, or match along dims.

    Args:
        inputs: Sequence of array-like objects to compare.
        dims: Optional axis index or sequence of axis indices to compare.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If inputs is not a list/tuple or dims is out of bounds for inputs[0].
        Exception: If shapes (or specified dimensions) do not match across inputs.
    """
    if not are_asserts_enabled():
        return
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of arrays.")
    if not inputs:
        return
    if dims is None:
        ref_shape = _get_shape(inputs[0])
        for item in inputs[1:]:
            if _get_shape(item) != ref_shape:
                _raise_chex_error(
                    f"Arrays have different shapes: {[_get_shape(t) for t in inputs]}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
    else:
        dim_list = [dims] if isinstance(dims, int) else list(dims)
        shape0 = _get_shape(inputs[0])
        if any(d >= len(shape0) or d < -len(shape0) for d in dim_list):
            raise ValueError(f"dims {dim_list} out of bounds for shape {shape0}.")
        expected_dims = [shape0[d] for d in dim_list]
        for item in inputs[1:]:
            st = _get_shape(item)
            if any(d >= len(st) or d < -len(st) for d in dim_list):
                _raise_chex_error(
                    f"dims {dim_list} out of bounds for shape {st}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
            for d, expected_val in zip(dim_list, expected_dims):
                if st[d] != expected_val:
                    _raise_chex_error(
                        f"Dimension {d} mismatch: expected {expected_val}, got {st[d]}.",
                        custom_message=custom_message,
                        custom_message_format_vars=custom_message_format_vars,
                        include_default_message=include_default_message,
                        exception_type=exception_type,
                    )


def assert_equal_shape_prefix(
    inputs: Sequence[ArrayTree] | ArrayTree,
    prefix_len: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that the leading prefix_len dimensions of all inputs are equal.

    Args:
        inputs: Sequence of array-like objects to compare.
        prefix_len: Number of leading dimensions to compare.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If inputs is not a list or tuple.
        Exception: If any array has rank < prefix_len or its prefix differs.
    """
    if not are_asserts_enabled():
        return
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of arrays.")
    if not inputs:
        return
    shapes = [_get_shape(t) for t in inputs]
    if any(len(s) < prefix_len for s in shapes):
        _raise_chex_error(
            f"At least one shape has rank smaller than prefix_len={prefix_len}: {shapes}",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )
    prefix = shapes[0][:prefix_len]
    for s in shapes[1:]:
        if s[:prefix_len] != prefix:
            _raise_chex_error(
                f"Arrays have different shape prefixes of length {prefix_len}: {shapes}",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )


def assert_equal_shape_suffix(
    inputs: Sequence[ArrayTree] | ArrayTree,
    suffix_len: int,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that the trailing suffix_len dimensions of all inputs are equal.

    Args:
        inputs: Sequence of array-like objects to compare.
        suffix_len: Number of trailing dimensions to compare.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If inputs is not a list or tuple.
        Exception: If any array has rank < suffix_len or its suffix differs.
    """
    if not are_asserts_enabled():
        return
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of arrays.")
    if not inputs or suffix_len == 0:
        return
    shapes = [_get_shape(t) for t in inputs]
    if any(len(s) < suffix_len for s in shapes):
        _raise_chex_error(
            f"At least one shape has rank smaller than suffix_len={suffix_len}: {shapes}",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )
    suffix = shapes[0][-suffix_len:]
    for s in shapes[1:]:
        if s[-suffix_len:] != suffix:
            _raise_chex_error(
                f"Arrays have different shape suffixes of length {suffix_len}: {shapes}",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )


def assert_equal_size(
    inputs: Sequence[ArrayTree] | ArrayTree,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that all arrays in inputs have the same total element count.

    Args:
        inputs: Sequence of array-like objects to compare.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If inputs is not a list or tuple.
        Exception: If total element counts differ across inputs.
    """
    if not are_asserts_enabled():
        return
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("inputs must be a list or tuple of arrays.")
    if not inputs:
        return
    ref_size = _get_size(inputs[0])
    for item in inputs[1:]:
        if _get_size(item) != ref_size:
            _raise_chex_error(
                f"Arrays have different sizes: {[_get_size(t) for t in inputs]}.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )


def _match_single_rank(
    actual_rank: int,
    spec: int | set[int] | Sequence[int | set[int] | Sequence[int]],
) -> bool:
    """Check whether actual_rank matches a rank specification.

    Args:
        actual_rank: Integer rank of the array.
        spec: Expected integer rank or collection of allowed integer ranks.

    Returns:
        True if actual_rank satisfies spec.
    """
    if isinstance(spec, int):
        return actual_rank == spec
    return actual_rank in spec


def assert_rank(
    inputs: ArrayTree,
    expected_ranks: int | set[int] | Sequence[int | set[int] | Sequence[int]],
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that input array(s) or PyTree leaves have rank matching expected_ranks.

    Args:
        inputs: Single array-like object, sequence of arrays, or PyTree of arrays.
        expected_ranks: Integer rank, set of allowed ranks, or sequence of per-input rank specs.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If expected_ranks is invalid or length mismatches inputs.
        Exception: If any input's rank does not match expected_ranks.
    """
    if not are_asserts_enabled():
        return
    is_array_seq = _is_pytree_container(inputs)
    is_scalar_multi_rank = (
        isinstance(inputs, (list, tuple))
        and not isinstance(inputs, NamedTuplePyTreeProtocol)
        and len(inputs) > 0
        and isinstance(expected_ranks, (list, tuple))
        and all(
            isinstance(
                item,
                (
                    ArrayProtocol,
                    VariableLeafProtocol,
                    ShapeOnlyProtocol,
                    TensorSpec,
                    int,
                    float,
                    complex,
                    bool,
                ),
            )
            for item in inputs
        )
    )
    if is_array_seq or is_scalar_multi_rank:
        items_list = (
            list(inputs)
            if isinstance(inputs, (list, tuple)) and not isinstance(inputs, NamedTuplePyTreeProtocol)
            else tree_leaves(inputs)
        )
        if isinstance(expected_ranks, (list, tuple)):
            if len(items_list) != len(expected_ranks):
                raise ValueError(
                    f"Length of inputs ({len(items_list)}) != length of expected_ranks ({len(expected_ranks)})."
                )
            for item, spec in zip(items_list, expected_ranks):
                if not _match_single_rank(_get_rank(item), spec):
                    _raise_chex_error(
                        f"Rank {_get_rank(item)} does not match expected {spec}.",
                        custom_message=custom_message,
                        custom_message_format_vars=custom_message_format_vars,
                        include_default_message=include_default_message,
                        exception_type=exception_type,
                    )
            return
        for item in items_list:
            if not _match_single_rank(_get_rank(item), expected_ranks):
                _raise_chex_error(
                    f"Rank {_get_rank(item)} does not match expected {expected_ranks}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
        return

    r = _get_rank(inputs)
    if isinstance(expected_ranks, int):
        allowed: set[int] = {expected_ranks}
    elif isinstance(expected_ranks, set):
        allowed = expected_ranks
    elif all(isinstance(x, int) for x in expected_ranks):
        allowed = {int(x) for x in expected_ranks if isinstance(x, int)}
    else:
        raise ValueError(f"Invalid expected_ranks specification: {expected_ranks!r}")

    if r not in allowed:
        _raise_chex_error(
            f"Expected rank in {allowed}, got {r}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def _dim_matches(
    actual_dim: int,
    matcher: TDimMatcher,
    bindings: dict[str, int] | None = None,
) -> bool:
    """Check whether a single dimension size matches a dimension matcher token.

    Args:
        actual_dim: Actual integer dimension size.
        matcher: Expected int, set of ints, tuple of ints, symbolic string, or None wildcard.
        bindings: Optional dictionary tracking symbolic dimension bindings across a call.

    Returns:
        True if actual_dim satisfies matcher.
    """
    if matcher is None:
        return True
    if isinstance(matcher, int) and not isinstance(matcher, bool):
        return actual_dim == matcher
    if isinstance(matcher, (set, tuple)):
        return actual_dim in matcher
    if isinstance(matcher, str):
        if bindings is None:
            return True
        if matcher in bindings:
            return bindings[matcher] == actual_dim
        bindings[matcher] = actual_dim
        return True
    return False


def _match_single_shape(
    actual_shape: tuple[int, ...],
    pattern: TShapeMatcher,
    bindings: dict[str, int] | None = None,
) -> None:
    """Validate that actual_shape matches a single shape matcher pattern.

    Args:
        actual_shape: Tuple of integer dimensions from the array.
        pattern: Sequence of dimension matchers (supporting at most one Ellipsis).
        bindings: Optional dictionary tracking symbolic dimension bindings.

    Raises:
        ValueError: If pattern contains more than one Ellipsis.
        AssertionError: If actual_shape does not match pattern.
    """
    pat_tuple = tuple(pattern)
    if pat_tuple == ():
        if actual_shape != ():
            raise AssertionError(f"Expected scalar shape (), got {actual_shape}.")
        return

    ellipsis_indices = [i for i, tok in enumerate(pat_tuple) if tok is Ellipsis]
    if len(ellipsis_indices) > 1:
        raise ValueError(f"Shape pattern cannot contain multiple Ellipsis: {pat_tuple!r}")

    if len(ellipsis_indices) == 1:
        ell_idx = ellipsis_indices[0]
        prefix_pat = pat_tuple[:ell_idx]
        suffix_pat = pat_tuple[ell_idx + 1 :]
        if len(actual_shape) < len(prefix_pat) + len(suffix_pat):
            raise AssertionError(f"Shape {actual_shape} is shorter than pattern {pat_tuple!r}.")
        for a, m in zip(actual_shape[: len(prefix_pat)], prefix_pat):
            if not _dim_matches(a, m, bindings):
                raise AssertionError(f"Shape {actual_shape} prefix does not match pattern {pat_tuple!r}.")
        if suffix_pat:
            for a, m in zip(actual_shape[-len(suffix_pat) :], suffix_pat):
                if not _dim_matches(a, m, bindings):
                    raise AssertionError(f"Shape {actual_shape} suffix does not match pattern {pat_tuple!r}.")
        return

    if len(actual_shape) != len(pat_tuple):
        raise AssertionError(f"Shape {actual_shape} rank does not match expected pattern {pat_tuple!r}.")
    for a, m in zip(actual_shape, pat_tuple):
        if not _dim_matches(a, m, bindings):
            raise AssertionError(f"Shape {actual_shape} does not match expected pattern {pat_tuple!r}.")


def _is_multi_shape_sequence(
    shapes: TShapeMatcher | Sequence[TShapeMatcher],
) -> TypeGuard[Sequence[TShapeMatcher]]:
    """Determine whether expected_shapes is a sequence of per-input shape matcher patterns.

    Args:
        shapes: Shape matcher or sequence of shape matchers.

    Returns:
        True if every element in shapes is itself a list or tuple shape pattern.
    """
    return (
        isinstance(shapes, (list, tuple))
        and len(shapes) > 0
        and all(isinstance(s, (list, tuple)) and not isinstance(s, set) for s in shapes)
    )


def _to_dim_matcher_list(
    raw_shape: TShapeMatcher | Sequence[TShapeMatcher],
) -> list[TDimMatcher]:
    """Convert a shape matcher sequence into a typed list of TDimMatcher tokens.

    Args:
        raw_shape: Shape matcher sequence.

    Returns:
        List of individual TDimMatcher elements.

    Raises:
        ValueError: If any token in raw_shape is not a valid dimension matcher.
    """
    result: list[TDimMatcher] = []
    for item in raw_shape:
        if item is None or item is Ellipsis or isinstance(item, str):
            result.append(item)
        elif isinstance(item, int) and not isinstance(item, bool):
            result.append(item)
        elif isinstance(item, set):
            if any(isinstance(x, bool) for x in item):
                raise ValueError(f"Invalid boolean in shape dimension matcher set: {item!r}")
            result.append(item)
        elif isinstance(item, tuple):
            if any(not isinstance(x, int) or isinstance(x, bool) for x in item):
                raise ValueError(f"Invalid shape dimension matcher tuple: {item!r}")
            int_tuple: tuple[int, ...] = tuple(x for x in item if isinstance(x, int))
            result.append(int_tuple)
        else:
            raise ValueError(f"Invalid shape dimension matcher token: {item!r}")
    return result


def assert_shape(
    inputs: ArrayTree,
    expected_shapes: TShapeMatcher | Sequence[TShapeMatcher] | None = None,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that input array(s) or PyTree leaves match the expected shape pattern(s).

    Args:
        inputs: Single array-like object, sequence of arrays, or PyTree of arrays.
        expected_shapes: Shape matcher tuple or sequence of per-input shape matcher tuples.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If multiple Ellipsis appear in a pattern, tokens are invalid, or lengths mismatch.
        Exception: If any array's shape fails to match its expected pattern.
    """
    if not are_asserts_enabled():
        return
    resolved_shapes: TShapeMatcher | Sequence[TShapeMatcher] = () if expected_shapes is None else expected_shapes
    bindings: dict[str, int] = {}
    try:
        if _is_pytree_container(inputs):
            items_list = (
                list(inputs)
                if isinstance(inputs, (list, tuple)) and not isinstance(inputs, NamedTuplePyTreeProtocol)
                else tree_leaves(inputs)
            )
            if _is_multi_shape_sequence(resolved_shapes):
                if len(items_list) != len(resolved_shapes):
                    raise ValueError(
                        f"Length of inputs ({len(items_list)}) != length of expected_shapes ({len(resolved_shapes)})."
                    )
                for item, pat in zip(items_list, resolved_shapes):
                    _match_single_shape(_get_shape(item), _to_dim_matcher_list(pat), bindings)
                return
            single_pat = _to_dim_matcher_list(resolved_shapes)
            for item in items_list:
                _match_single_shape(_get_shape(item), single_pat, bindings)
            return

        single_pattern = _to_dim_matcher_list(resolved_shapes)
        _match_single_shape(_get_shape(inputs), single_pattern, bindings)
    except AssertionError as exc:
        _raise_chex_error(
            str(exc),
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


SingleSizeMatcher = Union[int, set[int], EllipsisType, Sequence[Union[int, set[int], EllipsisType]]]
SizeSpecElement = Union[SingleSizeMatcher, Sequence[SingleSizeMatcher]]


def _match_single_size(
    actual_size: int,
    spec: SizeSpecElement,
) -> bool:
    """Check whether actual_size matches an expected size specification.

    Args:
        actual_size: Total number of elements in the array.
        spec: Expected integer size, set of sizes, or sequence of candidate sizes.

    Returns:
        True if actual_size matches spec.
    """
    if spec is Ellipsis:
        return True
    if isinstance(spec, int):
        return actual_size == spec
    if isinstance(spec, set):
        return actual_size in spec
    if Ellipsis in spec:
        return True
    for candidate in spec:
        if isinstance(candidate, int) and actual_size == candidate:
            return True
        if isinstance(candidate, set) and actual_size in candidate:
            return True
    return False


def assert_size(
    inputs: ArrayTree,
    expected_sizes: SizeSpecElement,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that input array(s) or PyTree leaves have total element count matching expected_sizes.

    Args:
        inputs: Single array-like object, sequence of arrays, or PyTree of arrays.
        expected_sizes: Integer size, set of allowed sizes, Ellipsis, or sequence of sizes.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If any input's total element count does not match expected_sizes.
    """
    if not are_asserts_enabled():
        return
    if _is_pytree_container(inputs):
        items_list = (
            list(inputs)
            if isinstance(inputs, (list, tuple)) and not isinstance(inputs, NamedTuplePyTreeProtocol)
            else tree_leaves(inputs)
        )
        if isinstance(expected_sizes, list) or (
            isinstance(expected_sizes, tuple) and any(isinstance(s, (list, tuple, set)) for s in expected_sizes)
        ):
            if len(items_list) != len(expected_sizes):
                _raise_chex_error(
                    f"Length of inputs ({len(items_list)}) != length of expected_sizes ({len(expected_sizes)}).",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
            for item, spec in zip(items_list, expected_sizes):
                if not _match_single_size(_get_size(item), spec):
                    _raise_chex_error(
                        f"Size {_get_size(item)} does not match expected {spec}.",
                        custom_message=custom_message,
                        custom_message_format_vars=custom_message_format_vars,
                        include_default_message=include_default_message,
                        exception_type=exception_type,
                    )
            return
        for item in items_list:
            if not _match_single_size(_get_size(item), expected_sizes):
                _raise_chex_error(
                    f"Size {_get_size(item)} does not match expected {expected_sizes}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
        return

    s = _get_size(inputs)
    if not _match_single_size(s, expected_sizes):
        _raise_chex_error(
            f"Expected size {expected_sizes}, got {s}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def _match_single_type(
    item: PyTree,
    spec: ArrayDType | Sequence[ArrayDType | Sequence[ArrayDType]] | set[ArrayDType],
) -> bool:
    """Check whether item's dtype matches a single or collection of expected dtypes.

    Args:
        item: Array or scalar to check.
        spec: Expected dtype or collection of allowed dtypes.

    Returns:
        True if item's dtype matches spec.
    """
    actual_dt = _get_type(item)
    if isinstance(spec, (DType, CompilerDType, DTypeLikeProtocol, str, type)):
        return _dtypes_match(actual_dt, spec)
    return any(
        isinstance(candidate, (DType, CompilerDType, DTypeLikeProtocol, str, type))
        and _dtypes_match(actual_dt, candidate)
        for candidate in spec
    )


def assert_type(
    inputs: ArrayTree,
    expected_types: ArrayDType | Sequence[ArrayDType | Sequence[ArrayDType]] | set[ArrayDType],
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that input array(s), scalar(s), or PyTree leaves match expected_types.

    Args:
        inputs: Single array/scalar, sequence of arrays/scalars, or PyTree.
        expected_types: Expected dtype, collection of allowed dtypes, or per-input dtype specs.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If any input's dtype does not match expected_types.
    """
    if not are_asserts_enabled():
        return
    is_seq_of_scalars_or_arrays = (
        isinstance(inputs, (list, tuple))
        and not isinstance(inputs, NamedTuplePyTreeProtocol)
        and len(inputs) > 0
        and all(
            isinstance(
                item,
                (
                    ArrayProtocol,
                    VariableLeafProtocol,
                    ShapeOnlyProtocol,
                    TensorSpec,
                    int,
                    float,
                    complex,
                    bool,
                ),
            )
            for item in inputs
        )
    )
    if _is_pytree_container(inputs) or is_seq_of_scalars_or_arrays:
        items_list = (
            list(inputs)
            if isinstance(inputs, (list, tuple)) and not isinstance(inputs, NamedTuplePyTreeProtocol)
            else tree_leaves(inputs)
        )
        if isinstance(expected_types, list):
            if len(items_list) != len(expected_types):
                _raise_chex_error(
                    f"Length of inputs ({len(items_list)}) != length of expected_types ({len(expected_types)}).",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
            for item, spec in zip(items_list, expected_types):
                if not _match_single_type(item, spec):
                    _raise_chex_error(
                        f"Type {_get_type(item)!r} does not match expected {spec!r}.",
                        custom_message=custom_message,
                        custom_message_format_vars=custom_message_format_vars,
                        include_default_message=include_default_message,
                        exception_type=exception_type,
                    )
            return
        for item in items_list:
            if not _match_single_type(item, expected_types):
                _raise_chex_error(
                    f"Type {_get_type(item)!r} does not match expected {expected_types!r}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
        return

    if not _match_single_type(inputs, expected_types):
        _raise_chex_error(
            f"Expected type {expected_types!r}, got {_get_type(inputs)!r}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


__all__ = [
    "SizeSpecElement",
    "_raise_chex_error",
    "assert_axis_dimension",
    "assert_axis_dimension_comparator",
    "assert_axis_dimension_gt",
    "assert_axis_dimension_gteq",
    "assert_axis_dimension_lt",
    "assert_axis_dimension_lteq",
    "assert_equal",
    "assert_equal_rank",
    "assert_equal_shape",
    "assert_equal_shape_prefix",
    "assert_equal_shape_suffix",
    "assert_equal_size",
    "assert_exactly_one_is_none",
    "assert_is_broadcastable",
    "assert_is_divisible",
    "assert_not_both_none",
    "assert_rank",
    "assert_shape",
    "assert_size",
    "assert_type",
]
