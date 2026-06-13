"""Tests for module."""

import pytest
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
import numpy as np
from zero_chex.core import (
    assert_axis_dimension,
    assert_axis_dimension_comparator,
    assert_axis_dimension_gt,
    assert_axis_dimension_gteq,
    assert_axis_dimension_lt,
    assert_axis_dimension_lteq,
    assert_equal_shape,
    assert_equal_rank,
    assert_equal_shape_prefix,
    assert_equal_shape_suffix,
    assert_equal_size,
    assert_rank,
    assert_shape,
    assert_size,
    assert_type,
    assert_tree_all_finite,
    assert_tree_has_only_ndarrays,
    assert_tree_shape_prefix,
    assert_tree_shape_suffix,
    assert_tree_no_nones,
    assert_trees_all_close,
    assert_trees_all_close_ulp,
    assert_trees_all_equal,
    assert_trees_all_equal_comparator,
    assert_trees_all_equal_dtypes,
    assert_trees_all_equal_shapes,
    assert_trees_all_equal_shapes_and_dtypes,
    assert_trees_all_equal_sizes,
    assert_trees_all_equal_structs,
    assert_devices_available,
    assert_gpu_available,
    assert_tpu_available,
    assert_tree_is_on_device,
    assert_tree_is_on_host,
    assert_tree_is_sharded,
    chexify,
    block_until_chexify_assertions_complete,
    assert_max_traces,
    assert_numerical_grads,
    clear_trace_counter,
    disable_asserts,
    enable_asserts,
    fake_jit,
    fake_pmap,
    fake_pmap_and_jit,
    restrict_backends,
    set_n_cpu_devices,
    assert_scalar,
    assert_scalar_in,
    assert_scalar_negative,
    assert_scalar_non_negative,
    assert_scalar_positive,
    assert_equal,
    assert_exactly_one_is_none,
    assert_not_both_none,
    assert_is_broadcastable,
    assert_is_divisible,
    dataclass,
    mappable_dataclass,
    params_product,
    warn_deprecated_function,
    warn_only_n_pos_args_in_future,
    if_args_not_none,
    all_variants,
    register_dataclass_type_with_jax_tree_util,
    variants,
    warn_keyword_args_only_in_future,
)


def test_assert_axis_dimension() -> None:
    """Test assert axis dimension."""
    t = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_axis_dimension(t, 0, 2)
    with pytest.raises(AssertionError):
        assert_axis_dimension(t, 0, 3)


def test_assert_axis_dimension_comparator() -> None:
    """Test assert axis dimension comparator."""
    t = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_axis_dimension_comparator(t, 0, lambda x: x == 2, "error")
    with pytest.raises(AssertionError, match="error"):
        assert_axis_dimension_comparator(t, 0, lambda x: x == 3, "error")


def test_assert_axis_dimension_gt() -> None:
    """Test assert axis dimension gt."""
    t = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_axis_dimension_gt(t, 1, 2)
    with pytest.raises(AssertionError):
        assert_axis_dimension_gt(t, 1, 3)


def test_assert_axis_dimension_gteq() -> None:
    """Test assert axis dimension gteq."""
    t = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_axis_dimension_gteq(t, 1, 3)
    with pytest.raises(AssertionError):
        assert_axis_dimension_gteq(t, 1, 4)


def test_assert_axis_dimension_lt() -> None:
    """Test assert axis dimension lt."""
    t = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_axis_dimension_lt(t, 1, 4)
    with pytest.raises(AssertionError):
        assert_axis_dimension_lt(t, 1, 3)


def test_assert_axis_dimension_lteq() -> None:
    """Test assert axis dimension lteq."""
    t = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_axis_dimension_lteq(t, 1, 3)
    with pytest.raises(AssertionError):
        assert_axis_dimension_lteq(t, 1, 2)


def test_assert_equal_shape() -> None:
    """Test assert equal shape."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t3 = Tensor(np.zeros((2, 4)), shape=(2, 4), dtype=DType.Float32, device="cpu")
    assert_equal_shape([t1, t2])
    with pytest.raises(AssertionError):
        assert_equal_shape([t1, t3])
    assert_equal_shape([])


def test_assert_equal_rank() -> None:
    """Test assert equal rank."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((4, 5)), shape=(4, 5), dtype=DType.Float32, device="cpu")
    t3 = Tensor(np.zeros((2,)), shape=(2,), dtype=DType.Float32, device="cpu")
    assert_equal_rank([t1, t2])
    with pytest.raises(AssertionError):
        assert_equal_rank([t1, t3])


def test_assert_equal_shape_prefix() -> None:
    """Test assert equal shape prefix."""
    t1 = Tensor(np.zeros((2, 3, 4)), shape=(2, 3, 4), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((2, 3, 5)), shape=(2, 3, 5), dtype=DType.Float32, device="cpu")
    assert_equal_shape_prefix([t1, t2], 2)
    with pytest.raises(AssertionError):
        assert_equal_shape_prefix([t1, t2], 3)


def test_assert_equal_shape_suffix() -> None:
    """Test assert equal shape suffix."""
    t1 = Tensor(np.zeros((5, 3, 4)), shape=(5, 3, 4), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((6, 3, 4)), shape=(6, 3, 4), dtype=DType.Float32, device="cpu")
    assert_equal_shape_suffix([t1, t2], 2)
    with pytest.raises(AssertionError):
        assert_equal_shape_suffix([t1, t2], 3)


def test_assert_equal_size() -> None:
    """Test assert equal size."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((6,)), shape=(6,), dtype=DType.Float32, device="cpu")
    t3 = Tensor(np.zeros((2, 4)), shape=(2, 4), dtype=DType.Float32, device="cpu")
    assert_equal_size([t1, t2])
    with pytest.raises(AssertionError):
        assert_equal_size([t1, t3])


def test_assert_rank() -> None:
    """Test assert rank."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_rank([t1], 2)
    assert_rank([t1], [2, 3])
    with pytest.raises(AssertionError):
        assert_rank([t1], 3)


def test_assert_shape() -> None:
    """Test assert shape."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_shape([t1], (2, 3))
    assert_shape([t1], [(2, 3), (4, 5)])
    with pytest.raises(AssertionError):
        assert_shape([t1], (2, 4))


def test_assert_size() -> None:
    """Test assert size."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_size([t1], 6)
    assert_size([t1], [6, 7])
    with pytest.raises(AssertionError):
        assert_size([t1], 7)


def test_assert_type() -> None:
    """Test assert type."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_type([t1], DType.Float32)
    assert_type([t1], [DType.Float32, DType.Float64])
    with pytest.raises(AssertionError):
        assert_type([t1], DType.Int32)


def test_assert_tree_all_finite() -> None:
    """Test assert tree all finite."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.array([1.0, np.nan]), shape=(2,), dtype=DType.Float32, device="cpu")
    assert_tree_all_finite([t1])
    with pytest.raises(AssertionError):
        assert_tree_all_finite([t2])


def test_assert_tree_has_only_ndarrays() -> None:
    """Test assert tree has only ndarrays."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_tree_has_only_ndarrays([t1])
    with pytest.raises(AssertionError):
        assert_tree_has_only_ndarrays([t1, 5])


def test_assert_tree_shape_prefix_suffix() -> None:
    """Test assert tree shape prefix suffix."""
    t1 = Tensor(np.zeros((2, 3, 4)), shape=(2, 3, 4), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((2, 3, 5)), shape=(2, 3, 5), dtype=DType.Float32, device="cpu")
    assert_tree_shape_prefix([t1, t2], (2, 3))
    with pytest.raises(AssertionError):
        assert_tree_shape_prefix([t1, t2], (2, 4))

    t3 = Tensor(np.zeros((5, 3, 4)), shape=(5, 3, 4), dtype=DType.Float32, device="cpu")
    assert_tree_shape_suffix([t1, t3], (3, 4))
    with pytest.raises(AssertionError):
        assert_tree_shape_suffix([t1, t3], (2, 4))


def test_assert_tree_no_nones() -> None:
    """Test assert tree no nones."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_tree_no_nones([t1, {"a": t1}])
    with pytest.raises(AssertionError):
        assert_tree_no_nones([t1, None])


def test_assert_trees_all_close() -> None:
    """Test assert trees all close."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    t2 = Tensor(
        np.array([1.0, 2.0000001]), shape=(2,), dtype=DType.Float32, device="cpu"
    )
    t3 = Tensor(np.array([1.0, 3.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    assert_trees_all_close([t1, t2], rtol=1e-5)
    with pytest.raises(AssertionError):
        assert_trees_all_close([t1, t3])


def test_assert_trees_all_equal() -> None:
    """Test assert trees all equal."""
    from ml_switcheroo_compiler.core.config import config

    config.eager_mode = True
    t1 = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    t3 = Tensor(np.array([1.0, 3.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    assert_trees_all_equal([t1, t2])
    with pytest.raises(AssertionError):
        assert_trees_all_equal([t1, t3])


def test_assert_trees_all_equal_comparator() -> None:
    """Test assert trees all equal comparator."""
    t1 = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device="cpu")
    assert_trees_all_equal_comparator(lambda x, y: True, [t1], [t1])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_comparator(lambda x, y: False, [t1], [t1])


def test_assert_trees_all_equal_dtypes() -> None:
    """Test assert trees all equal dtypes."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float64, device="cpu")
    assert_trees_all_equal_dtypes([t1, t1])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_dtypes([t1, t2])


def test_assert_trees_all_equal_shapes() -> None:
    """Test assert trees all equal shapes."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((2, 4)), shape=(2, 4), dtype=DType.Float32, device="cpu")
    assert_trees_all_equal_shapes([t1, t1])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_shapes([t1, t2])


def test_assert_trees_all_equal_shapes_and_dtypes() -> None:
    """Test assert trees all equal shapes and dtypes."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_trees_all_equal_shapes_and_dtypes([t1, t1])


def test_assert_trees_all_equal_sizes() -> None:
    """Test assert trees all equal sizes."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t2 = Tensor(np.zeros((6,)), shape=(6,), dtype=DType.Float32, device="cpu")
    t3 = Tensor(np.zeros((7,)), shape=(7,), dtype=DType.Float32, device="cpu")
    assert_trees_all_equal_sizes([t1, t2])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_sizes([t1, t3])


def test_assert_trees_all_equal_structs() -> None:
    """Test assert trees all equal structs."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_trees_all_equal_structs([[t1], [t1]])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_structs([[t1], {"a": t1}])


def test_assert_devices_available() -> None:
    """Test assert devices available."""
    assert_devices_available(1, "cpu", not_less_than=True)
    with pytest.raises(AssertionError):
        assert_devices_available(2, "cpu")
    with pytest.raises(AssertionError):
        assert_devices_available(1, "gpu")


def test_assert_gpu_available() -> None:
    """Test assert gpu available."""
    with pytest.raises(AssertionError):
        assert_gpu_available()


def test_assert_tpu_available() -> None:
    """Test assert tpu available."""
    with pytest.raises(AssertionError):
        assert_tpu_available()


def test_assert_tree_is_on_device() -> None:
    """Test assert tree is on device."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="gpu")
    t2 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_tree_is_on_device([t1])
    with pytest.raises(AssertionError):
        assert_tree_is_on_device([t2])


def test_assert_tree_is_on_host() -> None:
    """Test assert tree is on host."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="gpu")
    t2 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_tree_is_on_host([t2])
    with pytest.raises(AssertionError):
        assert_tree_is_on_host([t1])


def test_assert_tree_is_sharded() -> None:
    """Test assert tree is sharded."""
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    t1.sharding = "sharded_mock"
    t2 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    assert_tree_is_sharded([t1])
    with pytest.raises(AssertionError):
        assert_tree_is_sharded([t2])


def test_chex_tracing_utils() -> None:
    """Test chex tracing utils."""
    clear_trace_counter()

    @assert_max_traces(n=1)
    def f() -> None:
        pass

    f()
    with pytest.raises(AssertionError):
        f()

    assert chexify(f)
    block_until_chexify_assertions_complete()
    assert_numerical_grads(f, ())
    disable_asserts()
    enable_asserts()
    with fake_jit():
        pass
    with fake_pmap():
        pass
    with fake_pmap_and_jit():
        pass
    restrict_backends()
    set_n_cpu_devices(1)


def test_scalar_math_logic() -> None:
    """Test scalar math logic."""
    assert_scalar(1.0)
    assert_scalar(Tensor(1.0, shape=(), dtype=DType.Float32, device="cpu"))
    with pytest.raises(AssertionError):
        assert_scalar(Tensor([1.0], shape=(1,), dtype=DType.Float32, device="cpu"))

    assert_scalar_in(1.0, 0.0, 2.0)
    with pytest.raises(AssertionError):
        assert_scalar_in(3.0, 0.0, 2.0)

    assert_scalar_negative(-1.0)
    with pytest.raises(AssertionError):
        assert_scalar_negative(1.0)

    assert_scalar_non_negative(0.0)
    assert_scalar_non_negative(1.0)
    with pytest.raises(AssertionError):
        assert_scalar_non_negative(-1.0)

    assert_scalar_positive(1.0)
    with pytest.raises(AssertionError):
        assert_scalar_positive(0.0)

    assert_equal(1, 1)
    with pytest.raises(AssertionError):
        assert_equal(1, 2)

    assert_exactly_one_is_none(1, None)
    with pytest.raises(AssertionError):
        assert_exactly_one_is_none(None, None)

    assert_not_both_none(1, None)
    with pytest.raises(AssertionError):
        assert_not_both_none(None, None)

    assert_is_broadcastable((1, 3), (2, 3))
    with pytest.raises(AssertionError):
        assert_is_broadcastable((2, 3), (3, 3))

    assert_is_divisible(4, 2)
    with pytest.raises(AssertionError):
        assert_is_divisible(5, 2)


def test_dataclasses_and_utils() -> None:
    """Test dataclasses and utils."""

    @dataclass
    class MyData:
        a: int

    m = MyData(1)
    assert m.a == 1

    @mappable_dataclass
    class MapData:
        b: int

    map_data = MapData(2)
    assert map_data.b == 2

    assert params_product([[1, 2], [3]]) == [(1, 3), (2, 3)]

    @warn_deprecated_function
    def dep() -> None:
        pass

    @warn_only_n_pos_args_in_future(n=1)
    def pos(a: object, b: object = None) -> None:
        pass

    assert if_args_not_none(lambda x: x, [1], {}) == 1
    assert if_args_not_none(lambda x: x, [None], {}) is None

    @all_variants()
    def test_av() -> None:
        pass

    @variants()
    def test_v() -> None:
        pass

    register_dataclass_type_with_jax_tree_util(MyData)

    @warn_keyword_args_only_in_future
    def kw(a: object) -> None:
        pass


def test_chex_missing_branches() -> None:
    """Test chex missing branches."""
    # Empty lists
    assert_equal_shape([])
    assert_equal_rank([])
    assert_equal_shape_prefix([], 1)
    assert_equal_shape_suffix([], 1)
    assert_equal_size([])
    assert_tree_shape_prefix([], (1,))
    assert_tree_shape_suffix([], (1,))
    assert_trees_all_close([])
    assert_trees_all_close_ulp([])
    assert_trees_all_equal([])
    assert_trees_all_equal_comparator(lambda x, y: True, *[])
    assert_trees_all_equal_dtypes([])
    assert_trees_all_equal_shapes([])
    assert_trees_all_equal_sizes([])
    assert_trees_all_equal_structs([])

    # assert_tree_shape_suffix with non-tensor
    with pytest.raises(AssertionError):
        assert_tree_shape_suffix([1], (1,))

    # trees have different structures
    t1 = Tensor(np.zeros((2, 3)), shape=(2, 3), dtype=DType.Float32, device="cpu")
    with pytest.raises(AssertionError):
        assert_trees_all_close([t1, {"a": t1}])
    with pytest.raises(AssertionError):
        assert_trees_all_close_ulp([t1, {"a": t1}])
    with pytest.raises(AssertionError):
        assert_trees_all_equal([t1, {"a": t1}])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_comparator(lambda x, y: True, t1, {"a": t1})
    with pytest.raises(AssertionError):
        assert_trees_all_equal_dtypes([t1, {"a": t1}])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_shapes([t1, {"a": t1}])
    with pytest.raises(AssertionError):
        assert_trees_all_equal_sizes([t1, {"a": t1}])

    # assert_trees_all_close_ulp
    assert_trees_all_close_ulp([t1, t1])
    t2 = Tensor(np.zeros((2, 3)) + 1.0, shape=(2, 3), dtype=DType.Float32, device="cpu")
    with pytest.raises(AssertionError):
        assert_trees_all_close_ulp([t1, t2])

    # assert_max_traces no fn
    @assert_max_traces()
    def f_traces() -> None:
        pass

    clear_trace_counter()
    f_traces()

    # assert_scalar with non-tensor
    with pytest.raises(AssertionError):
        assert_scalar("string")

    # assert_scalar_in with tensor and not included
    t_scalar = Tensor(1.0, shape=(), dtype=DType.Float32, device="cpu")
    assert_scalar_in(t_scalar, 0.0, 2.0, included=False)
    with pytest.raises(AssertionError):
        assert_scalar_in(t_scalar, 1.0, 2.0, included=False)

    # dataclass without args
    from zero_chex.core import (
        dataclass,
        create_deprecated_function_alias,
        warn_deprecated_function,
        get_err_regex,
    )

    @dataclass()
    class D:
        pass

    # params_product named
    assert params_product({"a": [1], "b": [2]}, named=True) == [{"a": 1, "b": 2}]

    # deprecation utilities
    aliased = create_deprecated_function_alias(lambda x: x, "new_name")
    assert aliased(1) == 1

    @warn_deprecated_function
    def dep() -> int:
        return 2

    assert dep() == 2

    from zero_chex.core import warn_only_n_pos_args_in_future

    @warn_only_n_pos_args_in_future(n=1)
    def pos(a: object, b: object = None) -> object:
        return a

    assert pos(1, 2) == 1

    from zero_chex.core import all_variants, variants, warn_keyword_args_only_in_future

    @all_variants()
    def test_av(a: object) -> object:
        return a

    assert test_av(1) == 1

    @variants()
    def test_v(a: object) -> object:
        return a

    assert test_v(1) == 1

    @warn_keyword_args_only_in_future
    def test_kw(a: object = 1) -> object:
        return a

    assert test_kw(1) == 1

    assert get_err_regex(ValueError("err")) == "err"


def test_chexify_call() -> None:
    """Test chexify call."""
    from zero_chex.core import chexify

    @chexify
    def my_fn(a: object) -> object:
        return a

    assert my_fn(5) == 5
