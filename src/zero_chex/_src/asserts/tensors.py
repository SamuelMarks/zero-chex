"""Internal tensor assertions module re-exporting from zero_chex.assertions."""

from __future__ import annotations

from zero_chex.assertions import (
    _get_rank,
    _get_shape,
    _get_size,
    _get_type,
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

__all__ = [
    "_get_rank",
    "_get_shape",
    "_get_size",
    "_get_type",
    "assert_axis_dimension",
    "assert_axis_dimension_comparator",
    "assert_axis_dimension_gt",
    "assert_axis_dimension_gteq",
    "assert_axis_dimension_lt",
    "assert_axis_dimension_lteq",
    "assert_equal_rank",
    "assert_equal_shape",
    "assert_equal_shape_prefix",
    "assert_equal_shape_suffix",
    "assert_equal_size",
    "assert_rank",
    "assert_shape",
    "assert_size",
    "assert_type",
]
