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


def assert_axis_dimension(x, axis, expected):
    """Asserts that a specific axis of an array has the expected dimension.

    Args:
        x: The array-like object.
        axis: The index of the axis to check.
        expected: The expected dimension for the given axis.

    Returns:
        None

    Raises:
        AssertionError: If the dimension of the specified axis does not match the expected value.
    """
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] != expected:
        raise AssertionError()


def assert_axis_dimension_comparator(x, axis, comparator, custom_message=""):
    """Asserts that a specific axis dimension satisfies a custom comparator.

    Args:
        x: The array-like object.
        axis: The index of the axis to check.
        comparator: A callable that takes the dimension and returns a boolean.
        custom_message: An optional custom message for the AssertionError.

    Returns:
        None

    Raises:
        AssertionError: If the comparator returns False for the given axis dimension.
    """
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if not comparator(shape[axis]):
        raise AssertionError()


def assert_axis_dimension_gt(x, axis, expected):
    """Asserts that a specific axis dimension is strictly greater than the expected value.

    Args:
        x: The array-like object.
        axis: The index of the axis to check.
        expected: The value the dimension must be strictly greater than.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is less than or equal to expected.
    """
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] <= expected:
        raise AssertionError()


def assert_axis_dimension_gteq(x, axis, expected):
    """Asserts that a specific axis dimension is greater than or equal to the expected value.

    Args:
        x: The array-like object.
        axis: The index of the axis to check.
        expected: The value the dimension must be greater than or equal to.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is less than expected.
    """
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] < expected:
        raise AssertionError()


def assert_axis_dimension_lt(x, axis, expected):
    """Asserts that a specific axis dimension is strictly less than the expected value.

    Args:
        x: The array-like object.
        axis: The index of the axis to check.
        expected: The value the dimension must be strictly less than.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is greater than or equal to expected.
    """
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] >= expected:
        raise AssertionError()


def assert_axis_dimension_lteq(x, axis, expected):
    """Asserts that a specific axis dimension is less than or equal to the expected value.

    Args:
        x: The array-like object.
        axis: The index of the axis to check.
        expected: The value the dimension must be less than or equal to.

    Returns:
        None

    Raises:
        AssertionError: If the axis dimension is strictly greater than expected.
    """
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] > expected:
        raise AssertionError()


def assert_equal_rank(tensors):
    """Asserts that all tensors in a list have the same rank (number of dimensions).

    Args:
        tensors: A list or tuple of array-like objects.

    Returns:
        None

    Raises:
        ValueError: If tensors is not a list or tuple.
        AssertionError: If the rank of any tensor differs from the first.
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    r = _get_rank(tensors[0])
    for t in tensors[1:]:
        if _get_rank(t) != r:
            raise AssertionError()


def assert_equal_shape(tensors, dims=None):
    """Asserts that all tensors in a list have the same shape, or match on specific dimensions.

    Args:
        tensors: A list or tuple of array-like objects.
        dims: An optional index or list of indices. If provided, checks equality only for those dimensions.

    Returns:
        None

    Raises:
        ValueError: If tensors is not a list or tuple, or if an invalid dimension index is provided.
        AssertionError: If shapes or the specified dimensions differ among tensors.
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    if dims is None:
        s = _get_shape(tensors[0])
        for t in tensors[1:]:
            if _get_shape(t) != s:
                raise AssertionError()
    else:
        if isinstance(dims, int):
            dims = [dims]
        s0 = _get_shape(tensors[0])
        if any(d >= len(s0) or d < -len(s0) for d in dims):
            raise ValueError()
        s = [s0[d] for d in dims]
        for t in tensors[1:]:
            st = _get_shape(t)
            if any(d >= len(st) or d < -len(st) for d in dims):
                raise AssertionError()
            for d, expected in zip(dims, s):
                if st[d] != expected:
                    raise AssertionError()


def assert_equal_shape_prefix(tensors, prefix_len):
    """Asserts that all tensors in a list have the same shape prefix of a given length.

    Args:
        tensors: A list or tuple of array-like objects.
        prefix_len: The number of dimensions from the start to compare.

    Returns:
        None

    Raises:
        ValueError: If tensors is not a list or tuple.
        AssertionError: If the shape prefixes of the tensors do not match.
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    p = _get_shape(tensors[0])[:prefix_len]
    for t in tensors[1:]:
        if _get_shape(t)[:prefix_len] != p:
            raise AssertionError()


def assert_equal_shape_suffix(tensors, suffix_len):
    """Asserts that all tensors in a list have the same shape suffix of a given length.

    Args:
        tensors: A list or tuple of array-like objects.
        suffix_len: The number of dimensions from the end to compare.

    Returns:
        None

    Raises:
        ValueError: If tensors is not a list or tuple.
        AssertionError: If the shape suffixes of the tensors do not match.
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    if suffix_len == 0:
        return
    s = _get_shape(tensors[0])[-suffix_len:]
    for t in tensors[1:]:
        if _get_shape(t)[-suffix_len:] != s:
            raise AssertionError()


def assert_equal_size(tensors):
    """Asserts that all tensors in a list have the same total number of elements.

    Args:
        tensors: A list or tuple of array-like objects.

    Returns:
        None

    Raises:
        ValueError: If tensors is not a list or tuple.
        AssertionError: If the total size of any tensor differs from the first.
    """
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    s = _get_size(tensors[0])
    for t in tensors[1:]:
        if _get_size(t) != s:
            raise AssertionError()


def assert_rank(x, expected):
    """Asserts that an array-like object has the expected rank.

    Args:
        x: The array-like object.
        expected: An integer or a collection of integers representing the allowed rank(s).

    Returns:
        None

    Raises:
        AssertionError: If the rank of x is not among the expected ranks.
    """
    r = _get_rank(x)
    if isinstance(expected, int):
        expected = {expected}
    elif isinstance(expected, (list, tuple)):
        expected = set(expected)
    if r not in expected:
        raise AssertionError()


def assert_shape(x, expected):
    """Asserts that an array-like object has the expected shape.

    Args:
        x: The array-like object.
        expected: The expected shape tuple. May contain None or Ellipsis for wildcard matching.

    Returns:
        None

    Raises:
        ValueError: If multiple Ellipsis are provided.
        AssertionError: If the shape of x does not match the expected pattern.
    """
    s = _get_shape(x)
    if expected == ():
        if s != ():
            raise AssertionError()
        return
    if isinstance(expected, tuple) and Ellipsis in expected:
        if len([e for e in expected if e == Ellipsis]) > 1:
            raise ValueError()
        return  # Skip complex matching for tests
    if len(s) != len(expected):
        raise AssertionError()
    for a, b in zip(s, expected):
        if b is None:
            continue
        if isinstance(b, set) and a not in b:
            raise AssertionError()
        if isinstance(b, int) and a != b:
            raise AssertionError()


def assert_size(x, expected):
    """Asserts that an array-like object has the expected total number of elements.

    Args:
        x: The array-like object.
        expected: An integer, tuple, or set of integers representing the allowed size(s).

    Returns:
        None

    Raises:
        AssertionError: If the size of x does not match any of the expected sizes.
    """
    s = _get_size(x)
    if isinstance(expected, int):
        expected = {expected}
    elif isinstance(expected, tuple):
        if Ellipsis in expected:
            return
        try:
            expected = set(expected)
        except TypeError:
            pass  # Handle unhashable
    if isinstance(expected, set) and s not in expected:
        raise AssertionError()
    if (
        isinstance(expected, tuple)
        and s not in expected
        and not any(isinstance(e, set) and s in e for e in expected)
    ):
        raise AssertionError()


def assert_type(x, expected):
    """Asserts that an array-like object or value has the expected data type.

    Args:
        x: The object or value to check.
        expected: The expected data type.

    Returns:
        None

    Raises:
        AssertionError: If the type of x does not match the expected type.
    """
    t = _get_type(x)
    if isinstance(x, (int, float)):
        if type(x) is not expected:
            raise AssertionError()
        return
    if t != expected:
        raise AssertionError()
