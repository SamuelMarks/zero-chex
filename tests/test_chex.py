"""Tests for disabled assertion branches across all zero_chex modules."""

from __future__ import annotations

from ml_switcheroo_ir import DType

import zero_chex as chex
from zero_chex.types import ZeroArray


def test_all_assertions_disabled() -> None:
    """Verify every assertion function returns early without raising when disable_asserts() is active."""
    chex.disable_asserts()
    try:
        x = ZeroArray.zeros((2, 3), dtype=DType.float32)
        y = ZeroArray.ones((4, 5, 6), dtype=DType.int32)

        chex.assert_devices_available(99, "cpu")
        chex.assert_gpu_available()
        chex.assert_tpu_available()
        chex.assert_equal(1, 2)
        chex.assert_exactly_one_is_none(None, None)
        chex.assert_not_both_none(None, None)
        chex.assert_is_broadcastable((2, 3), (4, 5))
        chex.assert_is_divisible(10, 3)

        chex.assert_scalar([1, 2])
        chex.assert_scalar_in(10, 0, 5)
        chex.assert_scalar_negative(10)
        chex.assert_scalar_non_negative(-10)
        chex.assert_scalar_positive(-10)

        chex.assert_axis_dimension(x, 0, 99)
        chex.assert_axis_dimension_comparator(x, 0, lambda d: False)
        chex.assert_axis_dimension_gt(x, 0, 99)
        chex.assert_axis_dimension_gteq(x, 0, 99)
        chex.assert_axis_dimension_lt(x, 0, 0)
        chex.assert_axis_dimension_lteq(x, 0, 0)
        chex.assert_equal_rank([x, y])
        chex.assert_equal_shape([x, y])
        chex.assert_equal_shape_prefix([x, y], 2)
        chex.assert_equal_shape_suffix([x, y], 2)
        chex.assert_equal_size([x, y])
        chex.assert_rank(x, 9)
        chex.assert_shape(x, (9, 9))
        chex.assert_size(x, 99)
        chex.assert_type(x, DType.int32)

        chex.assert_tree_all_finite({"a": float("nan")})
        chex.assert_tree_has_only_ndarrays({"a": 1})
        chex.assert_tree_is_on_device({"a": x}, platform="tpu")
        chex.assert_tree_is_on_host({"a": x}, allow_cpu_device=False)
        chex.assert_tree_is_sharded({"a": x})
        chex.assert_tree_no_nans({"a": float("nan")})
        chex.assert_tree_no_nones({"a": None})
        chex.assert_tree_shape_prefix({"a": x}, (9, 9))
        chex.assert_tree_shape_suffix({"a": x}, (9, 9))
        chex.assert_trees_all_close({"a": 1.0}, {"a": 99.0})
        chex.assert_trees_all_close_ulp({"a": 1.0}, {"a": 99.0})
        chex.assert_trees_all_equal({"a": 1}, {"a": 2})
        chex.assert_trees_all_equal_comparator(lambda a, b: False, lambda a, b: "err", {"a": 1}, {"a": 2})
        chex.assert_trees_all_equal_dtypes({"a": x}, {"a": y})
        chex.assert_trees_all_equal_shapes({"a": x}, {"a": y})
        chex.assert_trees_all_equal_shapes_and_dtypes({"a": x}, {"a": y})
        chex.assert_trees_all_equal_sizes({"a": x}, {"a": y})
        chex.assert_trees_all_equal_structs({"a": 1}, {"b": 2})

        chex.assert_numerical_grads(lambda v: v, [1.0], order=0)
    finally:
        chex.enable_asserts()
