"""Tests for PyTree structural and leaf value assertions."""

from __future__ import annotations

import struct

import pytest
from ml_switcheroo_ir import DType

from zero_chex import (
    Device,
    assert_tree_all_finite,
    assert_tree_has_only_ndarrays,
    assert_tree_is_on_device,
    assert_tree_is_on_host,
    assert_tree_is_sharded,
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
    dataclass,
)
from zero_chex._src.tree_util import (
    _is_leaf_value,
    tree_all,
    tree_flatten,
    tree_flatten_with_path,
    tree_leaves,
    tree_map,
    tree_structure,
    tree_unflatten,
)
from zero_chex.dataclasses import MappableDataclassMixin
from zero_chex.trees import (
    _check_sharding,
    _float32_ulp_diff,
    _format_path,
    _leaf_close,
    _leaf_close_ulp,
    _leaf_equal,
)
from zero_chex.types import LeafValue, PmapSharding, ZeroArray


def test_tree_util_functions() -> None:
    """Test tree_flatten, tree_unflatten, tree_map, tree_all, and tree_structure."""
    assert _format_path(()) == "root"
    assert _format_path(("a", 0)) == "root['a'][0]"

    tree = {"a": [1, 2], "b": (3, 4)}
    leaves_with_path, treedef = tree_flatten_with_path(tree)
    assert leaves_with_path == [
        (("a", 0), 1),
        (("a", 1), 2),
        (("b", 0), 3),
        (("b", 1), 4),
    ]
    assert treedef.num_leaves == 4
    assert tree_leaves(tree) == [1, 2, 3, 4]
    assert tree_structure(tree) == treedef

    reconstructed = tree_unflatten(treedef, [10, 20, 30, 40])
    assert reconstructed == {"a": [10, 20], "b": (30, 40)}

    def _double_leaves(*leaves: LeafValue) -> LeafValue:
        """Double the first integer leaf."""
        first = leaves[0]
        return first * 2 if isinstance(first, int) else first

    def _sum_leaves(*leaves: LeafValue) -> LeafValue:
        """Sum two integer leaves."""
        first, second = leaves[0], leaves[1]
        if isinstance(first, int) and isinstance(second, int):
            return first + second
        return first

    doubled = tree_map(_double_leaves, tree)
    assert doubled == {"a": [2, 4], "b": (6, 8)}

    added = tree_map(_sum_leaves, tree, doubled)
    assert added == {"a": [3, 6], "b": (9, 12)}

    assert tree_all({"a": True, "b": [1, True]})
    assert not tree_all({"a": True, "b": [0, True]})

    leaves_flat, def_flat = tree_flatten(5)
    assert leaves_flat == [5]
    assert tree_unflatten(def_flat, [9]) == 9

    @dataclass
    class CustomNode(MappableDataclassMixin):
        """Dataclass testing _is_leaf_value and custom PyTree traversal."""

        val: int = 1

        def __init__(self, val: int = 1) -> None:
            """Initialize CustomNode.

            Args:
                val: Stored integer value.
            """
            self.val = val

    node_inst = CustomNode(7)
    assert not _is_leaf_value(node_inst)
    nl_leaves, nl_def = tree_flatten(node_inst)
    assert nl_leaves == [7]
    assert tree_unflatten(nl_def, [11]) == CustomNode(11)


def test_tree_device_host_sharding_assertions() -> None:
    """Test assert_tree_is_on_device, assert_tree_is_on_host, and assert_tree_is_sharded."""
    cpu_dev = Device("cpu", 0)
    gpu_dev = Device("gpu", 0)
    sharding = PmapSharding([cpu_dev])

    cpu_arr = ZeroArray.zeros((2,), device=cpu_dev)
    gpu_arr = ZeroArray.zeros((2,), device=gpu_dev)
    sharded_arr = ZeroArray.zeros((2,), device=cpu_dev, sharding=sharding)

    assert not _check_sharding("plain_string")
    assert not _check_sharding(cpu_arr)
    assert _check_sharding(sharded_arr)

    assert_tree_is_on_device({"a": gpu_arr}, platform="gpu")
    assert_tree_is_on_device({"a": gpu_arr}, platform=("gpu", "tpu"), device=gpu_dev)
    with pytest.raises(AssertionError, match="not placed on any device"):
        assert_tree_is_on_device({"a": 42}, platform="gpu")
    with pytest.raises(AssertionError, match="expected"):
        assert_tree_is_on_device({"a": cpu_arr}, platform="gpu")
    with pytest.raises(AssertionError, match="device mismatch"):
        assert_tree_is_on_device({"a": gpu_arr}, platform="gpu", device=Device("gpu", 1))

    assert_tree_is_on_host({"a": cpu_arr, "b": 42})
    with pytest.raises(AssertionError, match="Sharded array"):
        assert_tree_is_on_host({"a": sharded_arr})
    assert_tree_is_on_host({"a": sharded_arr}, allow_sharded_arrays=True)
    with pytest.raises(AssertionError, match="allow_cpu_device=False"):
        assert_tree_is_on_host({"a": cpu_arr}, allow_cpu_device=False)
    with pytest.raises(AssertionError, match="non-CPU device"):
        assert_tree_is_on_host({"a": gpu_arr})

    assert_tree_is_sharded({"a": sharded_arr})
    assert_tree_is_sharded({"a": sharded_arr}, devices=[cpu_dev])
    with pytest.raises(AssertionError, match="not sharded"):
        assert_tree_is_sharded({"a": cpu_arr})
    with pytest.raises(AssertionError, match="Leaf sharded across"):
        assert_tree_is_sharded({"a": sharded_arr}, devices=[gpu_dev])


def test_single_tree_assertions() -> None:
    """Test finite, ndarray, no_nones, shape_prefix, and shape_suffix tree assertions."""
    arr = ZeroArray.zeros((2, 3, 4))
    assert_tree_all_finite({"a": arr, "b": 1.0, "c": 1 + 2j})
    with pytest.raises(AssertionError, match="non-finite"):
        assert_tree_all_finite({"a": float("nan")})
    with pytest.raises(AssertionError, match="non-finite"):
        assert_tree_all_finite({"a": complex(1.0, float("nan"))})
    with pytest.raises(AssertionError, match="non-finite"):
        assert_tree_all_finite({"a": ZeroArray([float("inf")])})

    assert_tree_has_only_ndarrays({"a": arr})
    with pytest.raises(AssertionError, match="not an ndarray"):
        assert_tree_has_only_ndarrays({"a": 1})

    assert_tree_no_nones({"a": arr, "b": 1})
    with pytest.raises(AssertionError, match="Tree is None"):
        assert_tree_no_nones(None)
    with pytest.raises(AssertionError, match="None detected"):
        assert_tree_no_nones({"a": None})

    assert_tree_shape_prefix({"a": arr}, (2, 3))
    with pytest.raises(AssertionError, match="expected prefix"):
        assert_tree_shape_prefix({"a": arr}, (2, 9))

    assert_tree_shape_suffix({"a": arr}, ())
    assert_tree_shape_suffix({"a": arr}, (3, 4))
    with pytest.raises(AssertionError, match="expected suffix"):
        assert_tree_shape_suffix({"a": arr}, (9, 4))


def _next_float32(val: float, steps: int = 1) -> float:
    """Step a Python float by `steps` IEEE-754 float32 ULPs.

    Args:
        val: Starting float value.
        steps: Signed number of float32 ULPs to advance.

    Returns:
        Float value shifted by `steps` float32 ULPs.
    """
    bits = struct.unpack("!i", struct.pack("!f", val))[0]
    return float(struct.unpack("!f", struct.pack("!i", bits + steps))[0])


def test_multi_tree_assertions() -> None:
    """Test pairwise and multi-tree assertions across structures, values, dtypes, shapes, and sizes."""
    v2_1ulp = _next_float32(2.0, 1)
    t1 = {"a": ZeroArray([1.0, 2.0], shape=(2,), dtype=DType.float32)}
    t2 = {"a": ZeroArray([1.0, v2_1ulp], shape=(2,), dtype=DType.float32)}
    t3 = {"a": ZeroArray([1.0, 5.0], shape=(2,), dtype=DType.float32)}
    t_int = {"a": ZeroArray([1, 2], shape=(2,), dtype=DType.int32)}
    t_diff_shape = {"a": ZeroArray([1.0, 2.0, 3.0], shape=(3,), dtype=DType.float32)}

    assert_trees_all_equal_structs(t1)
    assert_trees_all_equal_structs(t1, t2)
    with pytest.raises(ValueError, match="different root types"):
        assert_trees_all_equal_structs(t1, [ZeroArray([1.0, 2.0], shape=(2,))])
    with pytest.raises(AssertionError, match="different structures"):
        assert_trees_all_equal_structs(t1, {"b": ZeroArray([1.0, 2.0], shape=(2,))})

    assert_trees_all_equal_comparator(lambda a, b: a == b, lambda a, b: "err", t1)
    assert_trees_all_equal_comparator(lambda a, b: a == b, lambda a, b: "err", t1, t1)

    assert_trees_all_close(t1, t2, rtol=1e-5, atol=1e-5)
    with pytest.raises(AssertionError, match="Leaves not close"):
        assert_trees_all_close(t1, t3)
    with pytest.raises(AssertionError, match="Leaves not close"):
        assert_trees_all_close({"a": 1 + 2j}, {"a": 1 + 99j})

    assert _leaf_close("abc", "abc")
    assert not _leaf_close("abc", "def")
    assert not _leaf_close(t1["a"], t_diff_shape["a"])
    assert _leaf_close(float("nan"), float("nan"), equal_nan=True)
    assert not _leaf_close(float("nan"), float("nan"), equal_nan=False)
    assert _leaf_close(complex(1.0, float("nan")), complex(1.0, float("nan")), equal_nan=True)
    assert _leaf_close(float("inf"), float("inf"))
    assert _leaf_close(complex(1.0, float("inf")), complex(1.0, float("inf")))
    assert not _leaf_close(float("inf"), float("-inf"))
    assert _leaf_close(1 + 2j, 1 + 2j)
    assert not _leaf_close(1 + 2j, 1 + 3j)

    # ULP tests across magnitudes (1e-20, 1.0, 1e10, negative, float64 overflow, complex)
    assert_trees_all_close_ulp(t1, t2, maxulp=1)
    assert_trees_all_close_ulp({"a": 1e10}, {"a": _next_float32(1e10, 1)}, maxulp=1)
    assert_trees_all_close_ulp({"a": 1e-20}, {"a": _next_float32(1e-20, 1)}, maxulp=1)
    assert_trees_all_close_ulp({"a": -2.0}, {"a": _next_float32(-2.0, 1)}, maxulp=1)
    assert _float32_ulp_diff(1e100, 1e100) == 0
    assert _float32_ulp_diff(1e100, 1e100 * (1.0 + 1e-15)) > 0
    assert _float32_ulp_diff(-1e100, -1e100 * (1.0 + 1e-15)) > 0
    assert _leaf_close_ulp("abc", "abc")
    assert not _leaf_close_ulp(t1["a"], t_diff_shape["a"])
    assert not _leaf_close_ulp(float("nan"), 1.0)
    assert _leaf_close_ulp(float("inf"), float("inf"))
    assert not _leaf_close_ulp(float("inf"), 1.0)
    assert _leaf_close_ulp(1 + 2j, complex(1.0, _next_float32(2.0, 1)), maxulp=1)
    assert not _leaf_close_ulp(1 + 2j, 1 + 5j, maxulp=1)
    with pytest.raises(AssertionError, match="ULPs"):
        assert_trees_all_close_ulp(t1, t3, maxulp=1)

    assert_trees_all_equal(t1, t1)
    assert_trees_all_equal(t1, t_int, strict=False)
    with pytest.raises(AssertionError, match="Leaves are not equal"):
        assert_trees_all_equal(t1, t_int, strict=True)
    with pytest.raises(AssertionError, match="Leaves are not equal"):
        assert_trees_all_equal(t1, t3)
    assert not _leaf_equal(t1["a"], t_diff_shape["a"])
    assert _leaf_equal("same", "same")
    assert _leaf_equal("same", "same", strict=True)

    assert_trees_all_equal_dtypes(t1)
    assert_trees_all_equal_dtypes(t1, t2)
    with pytest.raises(AssertionError, match=r"not a \(j-\)np array"):
        assert_trees_all_equal_dtypes({"a": 1}, {"a": 2})
    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_trees_all_equal_dtypes(t1, t_int)

    assert_trees_all_equal_shapes(t1, t2)
    with pytest.raises(AssertionError, match="shapes do not match"):
        assert_trees_all_equal_shapes(t1, t_diff_shape)

    assert_trees_all_equal_sizes(t1, t2)
    with pytest.raises(AssertionError, match="sizes do not match"):
        assert_trees_all_equal_sizes(t1, t_diff_shape)

    assert_trees_all_equal_shapes_and_dtypes(t1, t2)
    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_trees_all_equal_shapes_and_dtypes(t1, t_int)
