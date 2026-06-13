"""Tests for tensor assertions."""

import ml_switcheroo_compiler.ops as jnp
import pytest

from zero_chex import (
    assert_axis_dimension,
    assert_axis_dimension_comparator,
    assert_axis_dimension_gt,
    assert_axis_dimension_gteq,
    assert_axis_dimension_lt,
    assert_axis_dimension_lteq,
    assert_equal_rank,
    assert_equal_shape,
    assert_equal_shape_prefix,
    assert_equal_shape_suffix,
    assert_equal_size,
)


def test_assert_axis_dimension():
    x = jnp.zeros((2, 3, 4))
    assert_axis_dimension(x, 1, 3)
    assert_axis_dimension(x, -1, 4)
    with pytest.raises(AssertionError):
        assert_axis_dimension(x, 1, 4)
    with pytest.raises(AssertionError):
        assert_axis_dimension(x, 5, 4)


def test_assert_axis_dimension_comparator():
    x = [1, 2]  # list, will be converted to array
    assert_axis_dimension_comparator(x, 0, lambda d: d == 2, "test")
    with pytest.raises(AssertionError):
        assert_axis_dimension_comparator(x, 0, lambda d: d == 3, "test")


def test_assert_axis_dimension_inequalities():
    x = jnp.zeros((5,))
    assert_axis_dimension_gt(x, 0, 4)
    with pytest.raises(AssertionError):
        assert_axis_dimension_gt(x, 0, 5)

    assert_axis_dimension_gteq(x, 0, 5)
    with pytest.raises(AssertionError):
        assert_axis_dimension_gteq(x, 0, 6)

    assert_axis_dimension_lt(x, 0, 6)
    with pytest.raises(AssertionError):
        assert_axis_dimension_lt(x, 0, 5)

    assert_axis_dimension_lteq(x, 0, 5)
    with pytest.raises(AssertionError):
        assert_axis_dimension_lteq(x, 0, 4)


def test_assert_equal_rank():
    assert_equal_rank([jnp.zeros((2, 3)), jnp.zeros((1, 5))])
    with pytest.raises(AssertionError):
        assert_equal_rank([jnp.zeros((2, 3)), jnp.zeros((1,))])
    with pytest.raises(ValueError):
        assert_equal_rank(1)  # type: ignore


def test_assert_equal_shape():
    assert_equal_shape([jnp.zeros((2, 3)), jnp.zeros((2, 3))])
    assert_equal_shape([jnp.zeros((2, 3)), jnp.zeros((2, 5))], dims=0)
    assert_equal_shape([jnp.zeros((2, 3, 4)), jnp.zeros((2, 5, 4))], dims=[0, 2])

    with pytest.raises(AssertionError):
        assert_equal_shape([jnp.zeros((2, 3)), jnp.zeros((2, 4))])
    with pytest.raises(AssertionError):
        assert_equal_shape([jnp.zeros((2, 3)), jnp.zeros((2, 4))], dims=1)
    with pytest.raises(ValueError):
        assert_equal_shape([jnp.zeros((2, 3)), jnp.zeros((2, 4))], dims=5)


def test_assert_equal_shape_prefix():
    assert_equal_shape_prefix([jnp.zeros((2, 3, 4)), jnp.zeros((2, 3, 5))], 2)
    with pytest.raises(AssertionError):
        assert_equal_shape_prefix([jnp.zeros((2, 3, 4)), jnp.zeros((2, 4, 5))], 2)


def test_assert_equal_shape_suffix():
    assert_equal_shape_suffix([jnp.zeros((1, 3, 4)), jnp.zeros((2, 3, 4))], 2)
    assert_equal_shape_suffix([jnp.zeros((1, 3, 4)), jnp.zeros((2, 3, 4))], 0)
    with pytest.raises(AssertionError):
        assert_equal_shape_suffix([jnp.zeros((1, 3, 4)), jnp.zeros((2, 4, 4))], 2)


def test_assert_equal_size():
    assert_equal_size([jnp.zeros((2, 3)), jnp.zeros((6,))])
    with pytest.raises(AssertionError):
        assert_equal_size([jnp.zeros((2, 3)), jnp.zeros((5,))])


def test_assert_rank():
    pass


def test_assert_shape():
    pass


def test_assert_size():
    pass


def test_assert_type():
    from zero_chex import assert_type
    import numpy as np

    assert_type(np.array([1], dtype=np.int32), np.int32)
    with pytest.raises(AssertionError):
        assert_type(np.array([1], dtype=np.int32), np.float32)


def test_assert_shape_none():
    from zero_chex import assert_shape

    assert_shape(jnp.zeros((2, 3)), (2, None))


def test_assert_size_ellipsis():
    from zero_chex import assert_size

    assert_size(jnp.zeros((2, 3)), (Ellipsis, 6))


def test_assert_size_unhashable():
    from zero_chex import assert_size

    with pytest.raises(AssertionError):
        assert_size(jnp.zeros((2, 3)), (4, {5}))


def test_assert_type_scalar():
    from zero_chex import assert_type

    assert_type(1, int)
    with pytest.raises(AssertionError):
        assert_type(1, float)
