"""Tests for tensor shape, rank, axis dimension, size, and dtype assertions."""

from __future__ import annotations

import pytest
import zero_jax.numpy as jnp
from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_ir import DType, TensorShape, TensorSpec
from zero_flax.nnx.variables import Param
from zero_grain import SharedMemoryArray

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
    assert_is_broadcastable,
    assert_rank,
    assert_shape,
    assert_size,
    assert_type,
)
from zero_chex.assertions import _dim_matches, _dtypes_match, _get_shape, _get_type, _normalize_dtype
from zero_chex.types import ZeroArray


class _CustomDTypeName:
    """Helper class implementing DTypeLikeProtocol with a .name property."""

    def __init__(self, name: str) -> None:
        """Initialize with dtype name.

        Args:
            name: The dtype name string.
        """
        self._name = name

    @property
    def name(self) -> str:
        """Return the dtype name."""
        return self._name


class _CustomTensorShapeHolder:
    """Helper class exposing a TensorShape on .shape."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        """Initialize with dims tuple.

        Args:
            dims: Dimension sizes.
        """
        self._shape = TensorShape(dims)

    @property
    def shape(self) -> TensorShape:
        """Return TensorShape."""
        return self._shape


class _CustomTupleShapeHolder:
    """Helper class exposing a plain tuple on .shape."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        """Initialize with dims tuple.

        Args:
            dims: Dimension sizes.
        """
        self._shape = dims

    @property
    def shape(self) -> tuple[int, ...]:
        """Return shape tuple."""
        return self._shape


def test_internal_tensor_helpers() -> None:
    """Test _get_shape, _normalize_dtype, _get_type, and _dim_matches edge cases."""
    assert _get_shape([]) == (0,)
    assert _get_shape([[1, 2], [3, 4]]) == (2, 2)
    assert _get_shape(42) == ()
    assert _get_shape(TensorSpec(shape=(3, 4), dtype=DType.float32)) == (3, 4)
    assert _get_shape(_CustomTensorShapeHolder((5, 6))) == (5, 6)
    assert _get_shape(_CustomTupleShapeHolder((7, 8))) == (7, 8)
    with pytest.raises(AssertionError, match="Ragged sequence"):
        _get_shape([[1, 2], [3]])

    assert _normalize_dtype(DType.float32) == "float32"
    assert _normalize_dtype(_CustomDTypeName("Float64")) == "float64"
    assert _normalize_dtype(_CustomDTypeName("DType.bool_")) == "bool"
    with pytest.raises(ValueError, match="Unrecognized dtype name"):
        _normalize_dtype(_CustomDTypeName("not_a_real_dtype"))
    assert _normalize_dtype(bool) == "bool"
    assert _normalize_dtype(int) == "int"
    assert _normalize_dtype(float) == "float"
    assert _normalize_dtype(complex) == "complex"
    assert _normalize_dtype(str) == "str"
    assert _normalize_dtype("DType.int32") == "int32"
    assert _normalize_dtype("bool_") == "bool"

    assert _get_type("hello") == "str"
    assert _get_type(TensorSpec(shape=(2,), dtype=DType.int64)) == DType.int64
    assert _dtypes_match("float32", float)
    assert _dtypes_match(float, "float32")
    assert not _dtypes_match("int32", float)
    assert not _dtypes_match("float32", "float64")
    assert not _dim_matches(3, ...)
    assert _dim_matches(3, "B", None)


def test_assert_axis_dimensions() -> None:
    """Test all axis dimension assertions and comparator failure cases."""
    x = ZeroArray.zeros((2, 3, 4))
    assert_axis_dimension(x, 0, 2)
    assert_axis_dimension(x, 1, 3)
    assert_axis_dimension(x, -1, 4)

    with pytest.raises(AssertionError, match="out of bounds"):
        assert_axis_dimension(x, 3, 2)
    with pytest.raises(AssertionError, match="out of bounds"):
        assert_axis_dimension(x, -4, 2)
    with pytest.raises(AssertionError, match="equal to 5"):
        assert_axis_dimension(x, 0, 5)
    with pytest.raises(AssertionError, match="Expected tensor to have dimension on axis 0"):
        assert_axis_dimension_comparator(x, 0, lambda d: d == 99, error_string="")

    assert_axis_dimension_gt(x, 1, 2)
    with pytest.raises(AssertionError, match="greater than 3"):
        assert_axis_dimension_gt(x, 1, 3)

    assert_axis_dimension_gteq(x, 1, 3)
    assert_axis_dimension_gteq(x, 1, 2)
    with pytest.raises(AssertionError, match="greater than or equal to 4"):
        assert_axis_dimension_gteq(x, 1, 4)

    assert_axis_dimension_lt(x, 1, 4)
    with pytest.raises(AssertionError, match="less than 3"):
        assert_axis_dimension_lt(x, 1, 3)

    assert_axis_dimension_lteq(x, 1, 3)
    assert_axis_dimension_lteq(x, 1, 4)
    with pytest.raises(AssertionError, match="less than or equal to 2"):
        assert_axis_dimension_lteq(x, 1, 2)


def test_assert_equal_rank_shape_prefix_suffix_size() -> None:
    """Test multi-array equality assertions for rank, shape, prefix, suffix, and size."""
    x = ZeroArray.zeros((2, 3))
    y = ZeroArray.ones((4, 5))
    z = ZeroArray.zeros((2, 3, 4))

    assert_equal_rank([])
    assert_equal_rank([x, y])
    with pytest.raises(ValueError, match="must be a list or tuple"):
        assert_equal_rank(x)
    with pytest.raises(AssertionError, match="different ranks"):
        assert_equal_rank([x, z])

    assert_equal_shape([])
    assert_equal_shape([x, ZeroArray.ones((2, 3))])
    assert_equal_shape([x, z], dims=0)
    assert_equal_shape([x, z], dims=[0, 1])
    with pytest.raises(ValueError, match="must be a list or tuple"):
        assert_equal_shape(x)
    with pytest.raises(AssertionError, match="different shapes"):
        assert_equal_shape([x, y])
    with pytest.raises(ValueError, match="out of bounds"):
        assert_equal_shape([x, y], dims=5)
    with pytest.raises(AssertionError, match="out of bounds"):
        assert_equal_shape([z, x], dims=2)
    with pytest.raises(AssertionError, match="Dimension 0 mismatch"):
        assert_equal_shape([x, y], dims=0)

    assert_equal_shape_prefix([], 1)
    assert_equal_shape_prefix([x, z], 2)
    with pytest.raises(ValueError, match="must be a list or tuple"):
        assert_equal_shape_prefix(x, 1)
    with pytest.raises(AssertionError, match="smaller than prefix_len"):
        assert_equal_shape_prefix([x, z], 3)
    with pytest.raises(AssertionError, match="different shape prefixes"):
        assert_equal_shape_prefix([x, y], 1)

    assert_equal_shape_suffix([], 1)
    assert_equal_shape_suffix([x, y], 0)
    assert_equal_shape_suffix([ZeroArray.zeros((1, 3, 4)), z], 2)
    with pytest.raises(ValueError, match="must be a list or tuple"):
        assert_equal_shape_suffix(x, 1)
    with pytest.raises(AssertionError, match="smaller than suffix_len"):
        assert_equal_shape_suffix([x, z], 3)
    with pytest.raises(AssertionError, match="different shape suffixes"):
        assert_equal_shape_suffix([x, y], 1)

    assert_equal_size([])
    assert_equal_size([ZeroArray.zeros((2, 3)), ZeroArray.zeros((6,))])
    with pytest.raises(ValueError, match="must be a list or tuple"):
        assert_equal_size(x)
    with pytest.raises(AssertionError, match="different sizes"):
        assert_equal_size([x, y])


def test_assert_rank_shape_size_type() -> None:
    """Test pattern-matching assertions for rank, shape, size, and type including PyTree inputs."""
    x = ZeroArray.zeros((2, 3), dtype=DType.float32)
    y = ZeroArray.zeros((4, 5, 6), dtype=DType.int32)
    s = ZeroArray(1.0)

    assert_rank(x, 2)
    assert_rank(x, {1, 2})
    assert_rank(x, [1, 2])
    assert_rank([x, y], [2, 3])
    assert_rank([x, y], {2, 3})
    assert_rank([x, ZeroArray.ones((4, 5))], 2)
    assert_rank({"a": x, "b": y}, [2, 3])
    assert_rank({"a": x, "b": ZeroArray.ones((4, 5))}, 2)
    assert_rank([1, 2], [0, 0])

    with pytest.raises(AssertionError, match="Expected rank"):
        assert_rank(x, 3)
    with pytest.raises(ValueError, match="Length of inputs"):
        assert_rank([x, y], [2])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_rank([x, y], [2, 2])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_rank([x, y], 2)
    with pytest.raises(ValueError, match="Invalid expected_ranks"):
        assert_rank(10, [{1}])

    assert_shape(s, ())
    with pytest.raises(AssertionError, match="Expected scalar shape"):
        assert_shape(x, ())

    assert_shape(x, (2, 3))
    assert_shape(x, (2, None))
    assert_shape(x, (2, {3, 4}))
    assert_shape(x, (2, (3, 4)))
    assert_shape(x, ("B", "N"))
    assert_shape([x, ZeroArray.ones((2, 5))], [("B", 3), ("B", 5)])
    with pytest.raises(AssertionError, match="does not match expected pattern"):
        assert_shape([x, ZeroArray.ones((4, 5))], [("B", 3), ("B", 5)])
    assert_shape(y, (4, ...))
    assert_shape(y, (..., 6))
    assert_shape(y, (4, ..., 6))
    assert_shape([x, ZeroArray.ones((2, 3))], (2, 3))
    assert_shape([x, y], [(2, 3), (4, 5, 6)])
    assert_shape({"a": x, "b": y}, [(2, 3), (4, 5, 6)])

    with pytest.raises(ValueError, match="multiple Ellipsis"):
        assert_shape(y, (..., 5, ...))
    with pytest.raises(ValueError, match="Invalid shape dimension matcher"):
        assert_shape(x, (True, 3))
    with pytest.raises(ValueError, match="Invalid boolean in shape dimension matcher set"):
        assert_shape(x, ({True, 2}, 3))
    with pytest.raises(ValueError, match="Invalid shape dimension matcher tuple"):
        assert_shape(x, ((True, 2), 3))
    with pytest.raises(AssertionError, match="shorter than pattern"):
        assert_shape(x, (2, 3, ..., 4))
    with pytest.raises(AssertionError, match="prefix does not match"):
        assert_shape(y, (9, ..., 6))
    with pytest.raises(AssertionError, match="suffix does not match"):
        assert_shape(y, (4, ..., 9))
    with pytest.raises(AssertionError, match="rank does not match"):
        assert_shape(x, (2, 3, 4))
    with pytest.raises(AssertionError, match="does not match expected pattern"):
        assert_shape(x, (2, 9))
    with pytest.raises(AssertionError, match="does not match expected pattern"):
        assert_shape(x, (2, {8, 9}))
    with pytest.raises(ValueError, match="Length of inputs"):
        assert_shape([x, y], [(2, 3)])

    assert_size(x, 6)
    assert_size(x, ...)
    assert_size(x, {6, 12})
    assert_size(x, (6, 12))
    assert_size(x, (4, {6}))
    assert_size(x, (...,))
    assert_size([x, y], [6, 120])
    assert_size([x, y], [(6, 12), (100, 120)])
    assert_size([x, ZeroArray.ones((6,))], 6)
    assert_size({"a": x, "b": y}, [6, 120])

    with pytest.raises(AssertionError, match="Expected size"):
        assert_size(x, 10)
    with pytest.raises(AssertionError, match="Expected size"):
        assert_size(x, (4, {5}))
    with pytest.raises(AssertionError, match="Length of inputs"):
        assert_size([x, y], [6])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_size([x, y], [6, 99])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_size([x, y], [(6, 12), (99, 100)])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_size([x, y], 6)

    assert_type(1, int)
    assert_type(1.0, DType.float32)
    assert_type(True, DType.bool)
    assert_type(1, DType.int32)
    assert_type(1, [int, float])
    assert_type(x, DType.float32)
    assert_type(x, float)
    assert_type(x, (DType.float32, DType.float64))
    assert_type([x, y], [DType.float32, DType.int32])
    assert_type({"a": x, "b": y}, [DType.float32, DType.int32])
    assert_type([x, ZeroArray.ones((2,), dtype=DType.float32)], DType.float32)

    with pytest.raises(AssertionError, match="Expected type"):
        assert_type(1, float)
    with pytest.raises(AssertionError, match="Expected type"):
        assert_type(x, DType.int32)
    with pytest.raises(AssertionError, match="Length of inputs"):
        assert_type([x, y], [DType.float32])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_type([x, y], [DType.float32, DType.float32])
    with pytest.raises(AssertionError, match="does not match expected"):
        assert_type([x, y], DType.float32)


def test_cross_ecosystem_tensor_assertions() -> None:
    """Test tensor assertions on zero_jax, ml_switcheroo_compiler, zero_flax, and zero_grain objects."""
    j_arr = jnp.zeros((2, 3, 4), dtype="float32")
    c_ten = CompilerTensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        TensorConfig(shape=(2, 4), dtype=CompilerDType.float32),
    )
    p_var = Param(j_arr)

    assert_shape(j_arr, (2, 3, 4))
    assert_shape(c_ten, (2, 4))
    assert_shape(p_var, (2, 3, 4))

    assert_rank(j_arr, 3)
    assert_rank(c_ten, 2)
    assert_rank(p_var, 3)

    assert_type(j_arr, DType.float32)
    assert_type(c_ten, CompilerDType.float32)
    assert_type(c_ten, [CompilerDType.int32, CompilerDType.float32])
    assert_type(p_var, DType.float32)

    assert_axis_dimension(j_arr, 0, 2)
    assert_axis_dimension(c_ten, 1, 4)
    assert_axis_dimension(p_var, 2, 4)

    assert_is_broadcastable(j_arr.shape, (1, 3, 4))
    assert_is_broadcastable(c_ten.shape, (2, 1))

    shm_arr = SharedMemoryArray((2, 3), "float32")
    try:
        assert_shape(shm_arr, (2, 3))
        assert_type(shm_arr, DType.float32)
    finally:
        shm_arr.unlink_on_del()

    with pytest.raises(ValueError, match=r"\[Chex\] Assertion failed: shape mismatch on batch 4"):
        assert_shape(
            ZeroArray.zeros((2, 3)),
            (4, 3),
            custom_message="shape mismatch on batch {}",
            custom_message_format_vars=(4,),
            include_default_message=False,
            exception_type=ValueError,
        )
