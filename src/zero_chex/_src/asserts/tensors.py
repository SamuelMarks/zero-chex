# ruff: noqa: F821, F403, F405, E402, E731, F841, E721

"""Module docstring."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy
    import jax
    from typing import *

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        """Docstring."""

        user = None


"""Tensor and shape assertions."""

import collections.abc
from typing import Any, Callable, Optional, Sequence, Set, Union

import zero_jax as jax
import numpy as np


def _assert_collection_of_arrays(inputs: Any) -> None:
    """Docstring."""
    if not isinstance(inputs, collections.abc.Collection):
        raise ValueError(f"`inputs` is not a collection of arrays: {inputs}.")


def _unelided_shape_matches(
    actual_shape: Sequence[int],
    expected_shape: Sequence[Optional[Union[int, Set[int]]]],
) -> bool:
    """Docstring."""
    if len(actual_shape) != len(expected_shape):
        return False
    for actual, expected in zip(actual_shape, expected_shape):
        if expected is None:
            continue
        if isinstance(expected, set):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _shape_matches(actual_shape: Sequence[int], expected_shape: Any) -> bool:
    """Docstring."""
    expected_prefix = []
    expected_suffix = None
    for dim in expected_shape:
        if dim is Ellipsis:
            if expected_suffix is not None:
                raise ValueError(
                    "`expected_shape` may not contain more than one ellipsis"
                )
            expected_suffix = []
        elif expected_suffix is None:
            expected_prefix.append(dim)
        else:
            expected_suffix.append(dim)

    if expected_suffix is None:
        assert len(expected_prefix) == len(expected_shape)
        return _unelided_shape_matches(actual_shape, expected_prefix)

    if len(actual_shape) < len(expected_prefix) + len(expected_suffix):
        return False

    if expected_prefix:
        actual_prefix = actual_shape[: len(expected_prefix)]
        if not _unelided_shape_matches(actual_prefix, expected_prefix):
            return False

    if expected_suffix:
        actual_suffix = actual_shape[-len(expected_suffix) :]
        if not _unelided_shape_matches(actual_suffix, expected_suffix):
            return False

    return True


def _format_shape_matcher(shape: Any) -> str:
    """Docstring."""
    return f"({', '.join('...' if d is Ellipsis else str(d) for d in shape)})"


def assert_axis_dimension_comparator(
    tensor: "jax.Array | numpy.ndarray | numpy.bool | numpy.number",
    axis: "int",
    pass_fn: "Callable[[int], bool]",
    error_string: "str",
):
    """Asserts that `pass_fn(tensor.shape[axis])` passes."""
    if not isinstance(tensor, (jax.Array, np.ndarray)):
        tensor = np.asarray(tensor)
    if axis >= len(tensor.shape) or axis < -len(tensor.shape):
        raise AssertionError(
            f"Expected tensor to have dim {error_string} on axis "
            f"'{axis}' but axis '{axis}' not available: tensor rank is "
            f"'{len(tensor.shape)}'."
        )
    if not pass_fn(tensor.shape[axis]):
        raise AssertionError(
            f"Expected tensor to have dimension {error_string} on axis"
            f" '{axis}' but got '{tensor.shape[axis]}' instead."
        )


def assert_axis_dimension(
    tensor: "jax.Array | numpy.ndarray | numpy.bool | numpy.number",
    axis: "int",
    expected: "int",
) -> "NoneType":
    """Checks that ``tensor.shape[axis] == expected``."""
    assert_axis_dimension_comparator(
        tensor,
        axis,
        pass_fn=lambda tensor_dim: tensor_dim == expected,
        error_string=f"equal to '{expected}'",
    )


def assert_axis_dimension_gt(
    tensor: "jax.Array | numpy.ndarray | numpy.bool | numpy.number",
    axis: "int",
    val: "int",
) -> "NoneType":
    """Checks that ``tensor.shape[axis] > val``."""
    assert_axis_dimension_comparator(
        tensor,
        axis,
        pass_fn=lambda tensor_dim: tensor_dim > val,
        error_string=f"greater than '{val}'",
    )


def assert_axis_dimension_gteq(
    tensor: "jax.Array | numpy.ndarray | numpy.bool | numpy.number",
    axis: "int",
    val: "int",
) -> "NoneType":
    """Checks that ``tensor.shape[axis] >= val``."""
    assert_axis_dimension_comparator(
        tensor,
        axis,
        pass_fn=lambda tensor_dim: tensor_dim >= val,
        error_string=f"greater than or equal to '{val}'",
    )


def assert_axis_dimension_lt(
    tensor: "jax.Array | numpy.ndarray | numpy.bool | numpy.number",
    axis: "int",
    val: "int",
) -> "NoneType":
    """Checks that ``tensor.shape[axis] < val``."""
    assert_axis_dimension_comparator(
        tensor,
        axis,
        pass_fn=lambda tensor_dim: tensor_dim < val,
        error_string=f"less than '{val}'",
    )


def assert_axis_dimension_lteq(
    tensor: "jax.Array | numpy.ndarray | numpy.bool | numpy.number",
    axis: "int",
    val: "int",
) -> "NoneType":
    """Checks that ``tensor.shape[axis] <= val``."""
    assert_axis_dimension_comparator(
        tensor,
        axis,
        pass_fn=lambda tensor_dim: tensor_dim <= val,
        error_string=f"less than or equal to '{val}'",
    )


def assert_equal_rank(
    inputs: "Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
) -> "NoneType":
    """Checks that all arrays have the same rank."""
    _assert_collection_of_arrays(inputs)
    rank = len(inputs[0].shape)
    expected_ranks = [rank] * len(inputs)
    ranks = [len(x.shape) for x in inputs]
    if ranks != expected_ranks:
        raise AssertionError(f"Arrays have different rank: {ranks}.")


def assert_equal_shape(
    inputs: "Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    dims: "int | Sequence[int] | NoneType" = None,
) -> "NoneType":
    """Checks that all arrays have the same shape."""
    _assert_collection_of_arrays(inputs)

    def extract_relevant_dims(shape: Any, dims_to_extract: Any) -> Any:
        """Docstring."""
        try:
            if dims_to_extract is None:
                return shape
            elif isinstance(dims_to_extract, int):
                return shape[dims_to_extract]
            else:
                return [shape[d] for d in dims_to_extract]
        except IndexError as err:
            raise ValueError(
                f"Indexing error when trying to extra dim(s) {dims_to_extract} from array shape "
                f"{shape}"
            ) from err

    shape = extract_relevant_dims(inputs[0].shape, dims)
    expected_shapes = [shape] * len(inputs)
    shapes = [extract_relevant_dims(x.shape, dims) for x in inputs]
    if shapes != expected_shapes:
        if dims is not None:
            msg = f"Arrays have different shapes at dims {dims}: {shapes}"
        else:
            msg = f"Arrays have different shapes: {shapes}."
        raise AssertionError(msg)


def assert_equal_shape_prefix(
    inputs: "Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    prefix_len: "int",
) -> "NoneType":
    """Checks that the leading ``prefix_dims`` dims of all inputs have same shape."""
    _assert_collection_of_arrays(inputs)
    shapes = [array.shape[:prefix_len] for array in inputs]
    if shapes != [shapes[0]] * len(shapes):
        raise AssertionError(f"Arrays have different shape prefixes: {shapes}")


def assert_equal_shape_suffix(
    inputs: "Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    suffix_len: "int",
) -> "NoneType":
    """Checks that the final ``suffix_len`` dims of all inputs have same shape."""
    _assert_collection_of_arrays(inputs)
    shapes = [
        array.shape[-suffix_len:] if suffix_len > 0 else array.shape[0:0]
        for array in inputs
    ]
    if shapes != [shapes[0]] * len(shapes):
        raise AssertionError(f"Arrays have different shape suffixes: {shapes}")


def assert_equal_size(
    inputs: "Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
) -> "NoneType":
    """Checks that all arrays have the same size."""
    _assert_collection_of_arrays(inputs)
    size = inputs[0].size
    expected_sizes = [size] * len(inputs)
    sizes = [x.size for x in inputs]
    if sizes != expected_sizes:
        raise AssertionError(f"Arrays have different sizes: {sizes}")


def assert_rank(
    inputs: "float | int | jax.Array | numpy.ndarray | numpy.bool | numpy.number | Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    expected_ranks: "int | set[int] | Sequence[int | set[int]]",
) -> "NoneType":
    """Checks that the rank of all inputs matches specified ``expected_ranks``."""
    if not isinstance(expected_ranks, (collections.abc.Collection, int)):
        raise ValueError(
            f"Error in rank compatibility check: expected ranks should be a single "
            f"integer or a collection of integers, got {expected_ranks}."
        )

    if isinstance(expected_ranks, np.ndarray):
        raise ValueError(
            f"Error in rank compatibility check: expected ranks should be a single "
            f"integer or a collection of integers, but was an array: "
            f"{expected_ranks}."
        )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]
    if not isinstance(expected_ranks, collections.abc.Sequence) or isinstance(
        expected_ranks, collections.abc.Set
    ):
        expected_ranks = [expected_ranks] * len(inputs)
    if len(inputs) != len(expected_ranks):
        raise AssertionError(
            "Length of inputs and expected_ranks must match: inputs has length "
            f"{len(inputs)}, expected_ranks has length {len(expected_ranks)}."
        )

    errors = []
    for idx, (x, expected) in enumerate(zip(inputs, expected_ranks)):
        if hasattr(x, "shape"):
            shape = x.shape
        else:
            shape = ()
        rank = len(shape)

        if isinstance(expected, collections.abc.Sequence) and not isinstance(
            expected, collections.abc.Set
        ):
            raise ValueError(
                "Error in rank compatibility check: "
                "Expected ranks should be integers or sets of integers."
            )

        options = expected if isinstance(expected, collections.abc.Set) else {expected}

        if rank not in options:
            errors.append((idx, rank, shape, expected))

    if errors:
        msg = "; ".join(
            f"input {e[0]} has rank {e[1]} (shape {e[2]}) but expected {e[3]}"
            for e in errors
        )
        raise AssertionError(f"Error in rank compatibility check: {msg}.")


def assert_shape(
    inputs: "float | int | jax.Array | numpy.ndarray | numpy.bool | numpy.number | Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    expected_shapes: "Sequence[int | set[int] | ellipsis | NoneType] | Sequence[Sequence[int | set[int] | ellipsis | NoneType]]",
) -> "NoneType":
    """Checks that the shape of all inputs matches specified ``expected_shapes``."""
    if not isinstance(expected_shapes, (list, tuple)):
        raise AssertionError(
            "Error in shape compatibility check: expected shapes should be a list "
            f"or tuple of ints, got {expected_shapes}."
        )

    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    if not expected_shapes or not isinstance(expected_shapes[0], (list, tuple)):
        expected_shapes = [expected_shapes] * len(inputs)
    if len(inputs) != len(expected_shapes):
        raise AssertionError(
            "Length of `inputs` and `expected_shapes` must match: "
            f"{len(inputs)} is not equal to {len(expected_shapes)}."
        )

    errors = []
    for idx, (x, expected) in enumerate(zip(inputs, expected_shapes)):
        shape = getattr(x, "shape", ())
        if not _shape_matches(shape, expected):
            errors.append((idx, shape, _format_shape_matcher(expected)))

    if errors:
        msg = "; ".join(
            f"input {e[0]} has shape {e[1]} but expected {e[2]}" for e in errors
        )
        raise AssertionError(f"Error in shape compatibility check: {msg}.")


def assert_size(
    inputs: "float | int | jax.Array | numpy.ndarray | numpy.bool | numpy.number | Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    expected_sizes: "Sequence[int | set[int] | ellipsis | NoneType] | Sequence[Sequence[int | set[int] | ellipsis | NoneType]]",
) -> "NoneType":
    """Checks that the size of all inputs matches specified ``expected_sizes``."""
    if not isinstance(inputs, collections.abc.Sequence):
        inputs = [inputs]

    if isinstance(expected_sizes, int):
        expected_sizes = [expected_sizes] * len(inputs)

    if not isinstance(expected_sizes, (list, tuple)):
        raise AssertionError(
            "Error in size compatibility check: expected sizes should be an int, "
            f"list, or tuple of ints, got {expected_sizes}."
        )

    if len(inputs) != len(expected_sizes):
        raise AssertionError(
            "Length of `inputs` and `expected_sizes` must match: "
            f"{len(inputs)} is not equal to {len(expected_sizes)}."
        )

    errors = []
    for idx, (x, expected) in enumerate(zip(inputs, expected_sizes)):
        size = getattr(x, "size", 1)
        int_condition = expected in {Ellipsis, None} or size == expected
        set_condition = (
            isinstance(expected, collections.abc.Collection) and size in expected
        )
        if not (int_condition or set_condition):
            errors.append((idx, size, expected))

    if errors:
        msg = "; ".join(
            f"input {e[0]} has size {e[1]} but expected {e[2]}" for e in errors
        )
        raise AssertionError(f"Error in size compatibility check: {msg}.")


def assert_type(
    inputs: "float | int | jax.Array | numpy.ndarray | numpy.bool | numpy.number | Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    expected_types: "str | type[Any] | numpy.dtype | jax._src.typing.SupportsDType | Sequence[str | type[Any] | numpy.dtype | jax._src.typing.SupportsDType]",
) -> "NoneType":
    """Checks that the type of all inputs matches specified ``expected_types``."""
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if not isinstance(expected_types, (list, tuple)):
        expected_types = [expected_types] * len(inputs)

    errors = []
    if len(inputs) != len(expected_types):
        raise AssertionError(
            "Length of `inputs` and `expected_types` must match, "
            f"got {len(inputs)} != {len(expected_types)}."
        )
    for idx, (x, expected) in enumerate(zip(inputs, expected_types)):
        dtype = np.result_type(x)
        if expected in {float, np.floating}:
            if not np.issubdtype(dtype, np.floating):
                errors.append((idx, dtype, expected))
        elif expected in {int, np.integer}:
            if not np.issubdtype(dtype, np.integer):
                errors.append((idx, dtype, expected))
        else:
            expected = np.dtype(expected)
            if dtype != expected:
                errors.append((idx, dtype, expected))

    if errors:
        msg = "; ".join(
            f"input {e[0]} has type {e[1]} but expected {e[2]}" for e in errors
        )
        raise AssertionError(f"Error in type compatibility check: {msg}.")
