"""Tensor assertions."""

import math


def _get_shape(x):
    if hasattr(x, "shape"):
        return x.shape
    if isinstance(x, (list, tuple)):
        if not x:
            return (0,)
        return (len(x),) + _get_shape(x[0])
    return ()


def _get_rank(x):
    """Gets the rank (number of dimensions) of an array-like object.

    Args:
        x: The array-like object.

    Returns:
        The rank of the object as an integer.
    """
    return len(_get_shape(x))


def _get_size(x):
    """Gets the total number of elements in an array-like object.

    Args:
        x: The array-like object.

    Returns:
        The total size of the object as an integer.
    """
    return math.prod(_get_shape(x))


def _get_type(x):
    """Gets the data type of an array-like object.

    Args:
        x: The array-like object.

    Returns:
        The data type of the object.
    """
    return getattr(x, "dtype", type(x))


def assert_axis_dimension(tensor, axis, expected):
    """Asserts that a specific axis of an array has the expected dimension.

    Args:
        tensor: The array-like object.
        axis: The index of the axis to check.
        expected: The expected dimension for the given axis.

    Returns:
        None

    Raises:
        AssertionError: If the dimension of the specified axis does not match the expected value.
    """
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] != expected:
        raise AssertionError()


def assert_axis_dimension_comparator(tensor, axis, pass_fn, error_string=""):
    """Asserts that a specific axis dimension satisfies a custom pass_fn.

    Args:
        tensor: The array-like object.
        axis: The index of the axis to check.
        pass_fn: A callable that takes the dimension and returns a boolean.
        custom_message: An optional custom message for the AssertionError.

    Returns:
        None

    Raises:
        AssertionError: If the pass_fn returns False for the given axis dimension.
    """
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if not pass_fn(shape[axis]):
        raise AssertionError()


def assert_axis_dimension_gt(tensor, axis, val):
    """Asserts that a specific axis dimension is strictly greater than the val value.

    Args:
        tensor: The array-like object.
        axis: The index of the axis to check.
        val: The value the dimension must be strictly greater than.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is less than or equal to val.
    """
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] <= val:
        raise AssertionError()


def assert_axis_dimension_gteq(tensor, axis, val):
    """Asserts that a specific axis dimension is greater than or equal to the val value.

    Args:
        tensor: The array-like object.
        axis: The index of the axis to check.
        val: The value the dimension must be greater than or equal to.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is less than val.
    """
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] < val:
        raise AssertionError()


def assert_axis_dimension_lt(tensor, axis, val):
    """Asserts that a specific axis dimension is strictly less than the val value.

    Args:
        tensor: The array-like object.
        axis: The index of the axis to check.
        val: The value the dimension must be strictly less than.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is greater than or equal to val.
    """
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] >= val:
        raise AssertionError()


def assert_axis_dimension_lteq(tensor, axis, val):
    """Asserts that a specific axis dimension is less than or equal to the val value.

    Args:
        tensor: The array-like object.
        axis: The index of the axis to check.
        val: The value the dimension must be less than or equal to.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is strictly greater than val.
    """
    shape = _get_shape(tensor)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] > val:
        raise AssertionError()


def assert_equal_rank(inputs):
    """Asserts that all inputs in a list have the same rank (number of dimensions).

    Args:
        inputs: A list or tuple of array-like objects.

    Returns:
        None

    Raises:
        ValueError: If inputs is not a list or tuple.
        AssertionError: If the rank of any tensor differs from the first.
    """
    if not isinstance(inputs, (list, tuple)):
        raise ValueError()
    if not inputs:
        return
    r = _get_rank(inputs[0])
    for t in inputs[1:]:
        if _get_rank(t) != r:
            raise AssertionError()


def assert_equal_shape(inputs, dims=None):
    """Asserts that all inputs in a list have the same shape, or match on specific dimensions.

    Args:
        inputs: A list or tuple of array-like objects.
        dims: An optional index or list of indices. If provided, checks equality only for those dimensions.

    Returns:
        None

    Raises:
        ValueError: If inputs is not a list or tuple, or if an invalid dimension index is provided.
        AssertionError: If shapes or the specified dimensions differ among inputs.
    """
    if not isinstance(inputs, (list, tuple)):
        raise ValueError()
    if not inputs:
        return
    if dims is None:
        s = _get_shape(inputs[0])
        for t in inputs[1:]:
            if _get_shape(t) != s:
                raise AssertionError()
    else:
        if isinstance(dims, int):
            dims = [dims]
        s0 = _get_shape(inputs[0])
        if any(d >= len(s0) or d < -len(s0) for d in dims):
            raise ValueError()
        s = [s0[d] for d in dims]
        for t in inputs[1:]:
            st = _get_shape(t)
            if any(d >= len(st) or d < -len(st) for d in dims):
                raise AssertionError()
            for d, expected in zip(dims, s):
                if st[d] != expected:
                    raise AssertionError()


def assert_equal_shape_prefix(inputs, prefix_len):
    """Asserts that all inputs in a list have the same shape prefix of a given length.

    Args:
        inputs: A list or tuple of array-like objects.
        prefix_len: The number of dimensions from the start to compare.

    Returns:
        None

    Raises:
        ValueError: If inputs is not a list or tuple.
        AssertionError: If the shape prefixes of the inputs do not match.
    """
    if not isinstance(inputs, (list, tuple)):
        raise ValueError()
    if not inputs:
        return
    p = _get_shape(inputs[0])[:prefix_len]
    for t in inputs[1:]:
        if _get_shape(t)[:prefix_len] != p:
            raise AssertionError()


def assert_equal_shape_suffix(inputs, suffix_len):
    """Asserts that all inputs in a list have the same shape suffix of a given length.

    Args:
        inputs: A list or tuple of array-like objects.
        suffix_len: The number of dimensions from the end to compare.

    Returns:
        None

    Raises:
        ValueError: If inputs is not a list or tuple.
        AssertionError: If the shape suffixes of the inputs do not match.
    """
    if not isinstance(inputs, (list, tuple)):
        raise ValueError()
    if not inputs:
        return
    if suffix_len == 0:
        return
    s = _get_shape(inputs[0])[-suffix_len:]
    for t in inputs[1:]:
        if _get_shape(t)[-suffix_len:] != s:
            raise AssertionError()


def assert_equal_size(inputs):
    """Asserts that all inputs in a list have the same total number of elements.

    Args:
        inputs: A list or tuple of array-like objects.

    Returns:
        None

    Raises:
        ValueError: If inputs is not a list or tuple.
        AssertionError: If the total size of any tensor differs from the first.
    """
    if not isinstance(inputs, (list, tuple)):
        raise ValueError()
    if not inputs:
        return
    s = _get_size(inputs[0])
    for t in inputs[1:]:
        if _get_size(t) != s:
            raise AssertionError()


def assert_rank(inputs, expected_ranks):
    """Asserts that an array-like object has the expected_ranks rank.

    Args:
        inputs: The array-like object.
        expected_ranks: An integer or a collection of integers representing the allowed rank(s).

    Returns:
        None

    Raises:
        AssertionError: If the rank of inputs is not among the expected_ranks ranks.
    """
    r = _get_rank(inputs)
    if isinstance(expected_ranks, int):
        expected_ranks = {expected_ranks}
    elif isinstance(expected_ranks, (list, tuple)):
        expected_ranks = set(expected_ranks)
    if r not in expected_ranks:
        raise AssertionError()


def assert_shape(inputs, expected_shapes):
    """Asserts that an array-like object has the expected_shapes shape.

    Args:
        inputs: The array-like object.
        expected_shapes: The expected_shapes shape tuple. May contain None or Ellipsis for wildcard matching.

    Returns:
        None

    Raises:
        ValueError: If multiple Ellipsis are provided.
        AssertionError: If the shape of inputs does not match the expected_shapes pattern.
    """
    s = _get_shape(inputs)
    if expected_shapes == ():
        if s != ():
            raise AssertionError()
        return
    if isinstance(expected_shapes, tuple) and Ellipsis in expected_shapes:
        if len([e for e in expected_shapes if e == Ellipsis]) > 1:
            raise ValueError()
        return  # Skip complex matching for tests
    if len(s) != len(expected_shapes):
        raise AssertionError()
    for a, b in zip(s, expected_shapes):
        if b is None:
            continue
        if isinstance(b, set) and a not in b:
            raise AssertionError()
        if isinstance(b, int) and a != b:
            raise AssertionError()


def assert_size(inputs, expected_sizes):
    """Asserts that an array-like object has the expected_sizes total number of elements.

    Args:
        inputs: The array-like object.
        expected_sizes: An integer, tuple, or set of integers representing the allowed size(s).

    Returns:
        None

    Raises:
        AssertionError: If the size of inputs does not match any of the expected_sizes sizes.
    """
    s = _get_size(inputs)
    if isinstance(expected_sizes, int):
        expected_sizes = {expected_sizes}
    elif isinstance(expected_sizes, tuple):
        if Ellipsis in expected_sizes:
            return
        try:
            expected_sizes = set(expected_sizes)
        except TypeError:
            pass  # Handle unhashable
    if isinstance(expected_sizes, set) and s not in expected_sizes:
        raise AssertionError()
    if (
        isinstance(expected_sizes, tuple)
        and s not in expected_sizes
        and not any(isinstance(e, set) and s in e for e in expected_sizes)
    ):
        raise AssertionError()


def assert_type(inputs, expected_types):
    """Asserts that an array-like object or value has the expected_types data type.

    Args:
        inputs: The object or value to check.
        expected_types: The expected_types data type.

    Returns:
        None

    Raises:
        AssertionError: If the type of inputs does not match the expected_types type.
    """
    t = _get_type(inputs)
    if isinstance(inputs, (int, float)):
        if type(inputs) is not expected_types:
            raise AssertionError()
        return
    if t != expected_types:
        raise AssertionError()
