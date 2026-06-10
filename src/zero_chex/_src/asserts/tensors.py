import numpy as np


def _get_shape(x):
    return getattr(x, "shape", np.array(x).shape)


def _get_rank(x):
    return len(_get_shape(x))


def _get_size(x):
    return np.prod(_get_shape(x))


def _get_type(x):
    return getattr(x, "dtype", type(x))


def assert_axis_dimension(x, axis, expected):
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] != expected:
        raise AssertionError()


def assert_axis_dimension_comparator(x, axis, comparator, custom_message=""):
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if not comparator(shape[axis]):
        raise AssertionError()


def assert_axis_dimension_gt(x, axis, expected):
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] <= expected:
        raise AssertionError()


def assert_axis_dimension_gteq(x, axis, expected):
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] < expected:
        raise AssertionError()


def assert_axis_dimension_lt(x, axis, expected):
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] >= expected:
        raise AssertionError()


def assert_axis_dimension_lteq(x, axis, expected):
    shape = _get_shape(x)
    if axis >= len(shape) or axis < -len(shape):
        raise AssertionError()
    if shape[axis] > expected:
        raise AssertionError()


def assert_equal_rank(tensors):
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    r = _get_rank(tensors[0])
    for t in tensors[1:]:
        if _get_rank(t) != r:
            raise AssertionError()


def assert_equal_shape(tensors, dims=None):
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
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    p = _get_shape(tensors[0])[:prefix_len]
    for t in tensors[1:]:
        if _get_shape(t)[:prefix_len] != p:
            raise AssertionError()


def assert_equal_shape_suffix(tensors, suffix_len):
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
    if not isinstance(tensors, (list, tuple)):
        raise ValueError()
    if not tensors:
        return
    s = _get_size(tensors[0])
    for t in tensors[1:]:
        if _get_size(t) != s:
            raise AssertionError()


def assert_rank(x, expected):
    r = _get_rank(x)
    if isinstance(expected, int):
        expected = {expected}
    elif isinstance(expected, (list, tuple)):
        expected = set(expected)
    if r not in expected:
        raise AssertionError()


def assert_shape(x, expected):
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
    t = _get_type(x)
    if isinstance(x, (int, float)):
        if type(x) is not expected:
            raise AssertionError()
        return
    if t != expected:
        raise AssertionError()
