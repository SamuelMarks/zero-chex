"""Internal tree assertions module re-exporting from zero_chex.trees and zero_chex.devices."""

from __future__ import annotations

from zero_chex.devices import (
    _check_sharding,
    assert_tree_is_on_device,
    assert_tree_is_on_host,
    assert_tree_is_sharded,
)
from zero_chex.trees import (
    _get_leaves,
    assert_tree_all_finite,
    assert_tree_has_only_ndarrays,
    assert_tree_no_nones,
    assert_tree_shape_prefix,
    assert_tree_shape_suffix,
    assert_trees_all_close,
    assert_trees_all_close_ulp,
    assert_trees_all_equal,
    assert_trees_all_equal_comparator,
    assert_trees_all_equal_dtypes,
    assert_trees_all_equal_shapes,
    assert_trees_all_equal_shapes_and_dtypes,
    assert_trees_all_equal_sizes,
    assert_trees_all_equal_structs,
)

__all__ = [
    "_check_sharding",
    "_get_leaves",
    "assert_tree_all_finite",
    "assert_tree_has_only_ndarrays",
    "assert_tree_is_on_device",
    "assert_tree_is_on_host",
    "assert_tree_is_sharded",
    "assert_tree_no_nones",
    "assert_tree_shape_prefix",
    "assert_tree_shape_suffix",
    "assert_trees_all_close",
    "assert_trees_all_close_ulp",
    "assert_trees_all_equal",
    "assert_trees_all_equal_comparator",
    "assert_trees_all_equal_dtypes",
    "assert_trees_all_equal_shapes",
    "assert_trees_all_equal_shapes_and_dtypes",
    "assert_trees_all_equal_sizes",
    "assert_trees_all_equal_structs",
]
