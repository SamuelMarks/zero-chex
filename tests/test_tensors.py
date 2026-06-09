"""Tests for tensor assertions."""

import zero_jax.numpy as jnp
import numpy as np
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
    assert_rank,
    assert_shape,
    assert_size,
    assert_type,
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
    assert_rank(jnp.zeros(()), 0)
    assert_rank(jnp.zeros((2,)), 1)
    assert_rank([jnp.zeros((2,)), jnp.zeros((3,))], 1)
    assert_rank(jnp.zeros((2,)), {1, 2})
    assert_rank([jnp.zeros(()), jnp.zeros((3,))], [0, 1])

    with pytest.raises(AssertionError):
        assert_rank(jnp.zeros((2,)), 2)
    with pytest.raises(AssertionError):
        assert_rank([jnp.zeros((2,)), jnp.zeros((3,))], [1])
    with pytest.raises(ValueError):
        assert_rank(jnp.zeros((2,)), "1")  # type: ignore
    with pytest.raises(ValueError):
        assert_rank(jnp.zeros((2,)), np.array([1]))  # type: ignore
    with pytest.raises(ValueError):
        assert_rank(jnp.zeros((2,)), None)  # type: ignore
    assert_rank(jnp.zeros((2,)), [1])
    assert_rank(1, 0)  # scalar


def test_assert_shape():
    assert_shape(1, ())  # scalar
    assert_shape(jnp.zeros(()), ())
    assert_shape(jnp.zeros((2, 3)), (2, 3))
    assert_shape(jnp.zeros((2, 3)), (2, {1, 3}))
    assert_shape(jnp.zeros((2, 3)), (2, None))
    assert_shape(jnp.zeros((2, 3)), (2, Ellipsis))
    assert_shape(jnp.zeros((2, 3)), (Ellipsis, 3))
    assert_shape(jnp.zeros((2, 3, 4)), (2, Ellipsis, 4))

    with pytest.raises(AssertionError):
        assert_shape(jnp.zeros((2, 3)), (2, {4, 5}))
    with pytest.raises(ValueError):
        assert_shape(jnp.zeros((2, 3)), (Ellipsis, 2, Ellipsis))
    with pytest.raises(AssertionError):
        assert_shape(
            jnp.zeros((2, 3)), (2, 3, Ellipsis, 4, 5)
        )  # len actual < prefix+suffix
    with pytest.raises(AssertionError):
        assert_shape(jnp.zeros((2, 3, 4)), (3, Ellipsis, 4))  # prefix unelided
    with pytest.raises(AssertionError):
        assert_shape(jnp.zeros((2, 3, 4)), (2, Ellipsis, 5))  # suffix unelided

    with pytest.raises(AssertionError):
        assert_shape(jnp.zeros((2, 3)), (2, 4))
    with pytest.raises(AssertionError):
        assert_shape(jnp.zeros((2, 3)), (2, 3, 4))
    with pytest.raises(ValueError):
        assert_shape(jnp.zeros((2, 3)), (Ellipsis, Ellipsis))
    with pytest.raises(AssertionError):
        assert_shape(jnp.zeros((2, 3)), "not a shape")  # type: ignore
    with pytest.raises(AssertionError):
        assert_shape([jnp.zeros((2, 3))], [(2, 3), (2, 4)])


def test_assert_size():
    assert_size(jnp.zeros(()), 1)
    assert_size(jnp.zeros((2, 3)), 6)
    assert_size(jnp.zeros((2, 3)), ({5, 6},))
    assert_size(jnp.zeros((2, 3)), (Ellipsis,))

    with pytest.raises(AssertionError):
        assert_size(jnp.zeros((2, 3)), 5)
    with pytest.raises(AssertionError):
        assert_size(jnp.zeros((2, 3)), "not a size")  # type: ignore
    with pytest.raises(AssertionError):
        assert_size([jnp.zeros((2, 3))], [6, 7])


def test_assert_type():
    assert_type(7, int)
    assert_type(7.1, float)
    assert_type(jnp.array(7), int)
    assert_type(jnp.array(7.1), float)
    assert_type(np.array(7, dtype=np.int8), np.int8)

    with pytest.raises(AssertionError):
        assert_type(7, float)
    with pytest.raises(AssertionError):
        assert_type(7.1, int)
    with pytest.raises(AssertionError):
        assert_type(np.array(7, dtype=np.int8), np.int16)
    with pytest.raises(AssertionError):
        assert_type([7], [int, float])
