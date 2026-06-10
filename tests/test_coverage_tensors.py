import pytest
import ml_switcheroo.ops as np
import ml_switcheroo.core.dtype as dt

import zero_chex as chex


def test_assert_axis_dimension_comparator_fail():
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_comparator(np.ones((2, 3)), 0, lambda x: x == 3)
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_comparator(np.ones((2, 3)), 5, lambda x: True)


def test_assert_axis_dimension_gt_fail():
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_gt(np.ones((2, 3)), 0, 2)
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_gt(np.ones((2, 3)), 5, 2)


def test_assert_axis_dimension_gteq_fail():
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_gteq(np.ones((2, 3)), 0, 3)
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_gteq(np.ones((2, 3)), 5, 2)


def test_assert_axis_dimension_lt_fail():
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_lt(np.ones((2, 3)), 0, 2)
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_lt(np.ones((2, 3)), 5, 2)


def test_assert_axis_dimension_lteq_fail():
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_lteq(np.ones((2, 3)), 0, 1)
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension_lteq(np.ones((2, 3)), 5, 2)


def test_assert_axis_dimension_fail():
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension(np.ones((2, 3)), 0, 3)
    with pytest.raises(AssertionError):
        chex.assert_axis_dimension(np.ones((2, 3)), 5, 2)


def test_assert_shape_fail():
    with pytest.raises(AssertionError):
        chex.assert_shape(np.ones((2, 3)), (2, 4))
    with pytest.raises(AssertionError):
        chex.assert_shape(np.ones((2, 3)), (2, 3, 4))


def test_assert_equal_shape_fail():
    with pytest.raises(AssertionError):
        chex.assert_equal_shape([np.ones((2, 3)), np.ones((2, 4))])


def test_assert_equal_shape_prefix_fail():
    with pytest.raises(AssertionError):
        chex.assert_equal_shape_prefix([np.ones((2, 3, 4)), np.ones((3, 3, 5))], 1)
    with pytest.raises(AssertionError):
        chex.assert_equal_shape_prefix([np.ones((2, 3)), np.ones((2,))], 2)


def test_assert_equal_shape_suffix_fail():
    with pytest.raises(AssertionError):
        chex.assert_equal_shape_suffix([np.ones((4, 2, 3)), np.ones((5, 3, 3))], 2)
    with pytest.raises(AssertionError):
        chex.assert_equal_shape_suffix([np.ones((2, 3)), np.ones((3,))], 2)


def test_assert_rank_fail():
    with pytest.raises(AssertionError):
        chex.assert_rank(np.ones((2, 3)), 3)


def test_assert_equal_rank_fail():
    with pytest.raises(AssertionError):
        chex.assert_equal_rank([np.ones((2, 3)), np.ones((2,))])


def test_assert_type_fail():
    with pytest.raises(AssertionError):
        chex.assert_type(np.ones((2, 3), dtype=dt.DType.Float32), dt.DType.Int32)


def test_assert_equal_rank_pass():
    chex.assert_equal_rank([])
    chex.assert_equal_rank([np.ones((2, 3)), np.ones((4, 5))])


def test_assert_equal_shape_pass():
    chex.assert_equal_shape([])
    chex.assert_equal_shape([np.ones((2, 3)), np.ones((2, 3))])


def test_assert_equal_shape_valueerror():
    with pytest.raises(ValueError):
        chex.assert_equal_shape(np.ones((2, 3)))


def test_assert_equal_shape_prefix_pass():
    chex.assert_equal_shape_prefix([], 1)
    chex.assert_equal_shape_prefix([np.ones((2, 3)), np.ones((2, 4))], 1)


def test_assert_equal_shape_prefix_valueerror():
    with pytest.raises(ValueError):
        chex.assert_equal_shape_prefix(np.ones((2, 3)), 1)


def test_assert_equal_shape_suffix_pass():
    chex.assert_equal_shape_suffix([], 1)
    chex.assert_equal_shape_suffix([np.ones((2, 3)), np.ones((4, 3))], 1)


def test_assert_equal_shape_suffix_valueerror():
    with pytest.raises(ValueError):
        chex.assert_equal_shape_suffix(np.ones((2, 3)), 1)


def test_assert_equal_size_pass():
    chex.assert_equal_size([])
    chex.assert_equal_size([np.ones((2, 3)), np.ones((6,))])


def test_assert_equal_size_valueerror():
    with pytest.raises(ValueError):
        chex.assert_equal_size(np.ones((2, 3)))


def test_assert_shape_empty_fail():
    with pytest.raises(AssertionError):
        chex.assert_shape(np.ones((2,)), ())


def test_assert_shape_tuple_set_fail():
    with pytest.raises(AssertionError):
        chex.assert_shape(np.ones((2,)), ((3,), {4}))


def test_assert_equal_shape_dims_assertionerror():
    with pytest.raises(AssertionError):
        chex.assert_equal_shape([np.ones((2, 3)), np.ones((2,))], dims=[1])


def test_assert_shape_ellipsis_valueerror():
    with pytest.raises(ValueError):
        chex.assert_shape(np.ones((2, 3)), (Ellipsis, Ellipsis))


def test_assert_shape_set_assertionerror():
    with pytest.raises(AssertionError):
        chex.assert_shape(np.ones((2, 3)), (2, {4}))


def test_assert_size_assertionerror():
    with pytest.raises(AssertionError):
        chex.assert_size(np.ones((2, 3)), 5)


def test_assert_shape_ellipsis_return():
    chex.assert_shape(np.ones((2, 3)), (Ellipsis, 3))


def test_assert_shape_tuple_assertionerror():
    with pytest.raises(AssertionError):
        chex.assert_shape(np.ones((2, 3)), (2, 4))
