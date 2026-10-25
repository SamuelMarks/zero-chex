"""Tests for PyTree structural and leaf value assertions."""

from __future__ import annotations

import pathlib
import struct

import pytest
import zero_jax.numpy as jnp
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_compiler.core.tensor import TensorConfig, as_tensor_obj
from ml_switcheroo_ir import DType
from zero_flax.core import FrozenDict
from zero_flax.nnx.state import State
from zero_flax.nnx.variables import BatchStat, Param
from zero_grain import Batch, DataLoader, InMemoryDataSource, SequentialSampler, SharedMemoryArray
from zero_grain._types import ElementTree
from zero_optax import ScaleByScheduleState, TraceState
from zero_optax.optimizers.adam import adam
from zero_orbax.checkpoint import PyTreeCheckpointer, PyTreeCheckpointHandler

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
from zero_chex.devices import _check_sharding
from zero_chex.trees import (
    _float32_ulp_diff,
    _format_path,
    _leaf_close,
    _leaf_close_ulp,
    _leaf_equal,
)
from zero_chex.types import DeviceProtocol, LeafValue, PmapSharding, ZeroArray


class _EmptyDevicesHolder:
    """Helper class implementing DevicesMethodProtocol with an empty devices() tuple."""

    def devices(self) -> tuple[DeviceProtocol, ...]:
        """Return empty devices tuple."""
        return ()


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

    with pytest.raises(ValueError, match="Too few leaves"):
        tree_unflatten(treedef, [10, 20])
    with pytest.raises(ValueError, match="Too many leaves"):
        tree_unflatten(treedef, [10, 20, 30, 40, 50])

    # Non-string dictionary keys (int and tuple) must be preserved
    int_key_tree: dict[int | tuple[int, str], int] = {1: 10, (2, "k"): 20}
    ik_leaves, ik_def = tree_flatten(int_key_tree)
    assert tree_unflatten(ik_def, ik_leaves) == int_key_tree

    def _double_leaves(*leaves: LeafValue) -> LeafValue:
        """Double the first integer leaf.

        Args:
            *leaves: Input leaf values across mapped trees.

        Returns:
            Doubled integer leaf or original leaf value.
        """
        first = leaves[0]
        return first * 2 if isinstance(first, int) else first

    def _sum_leaves(*leaves: LeafValue) -> LeafValue:
        """Sum two integer leaves.

        Args:
            *leaves: Input leaf values across mapped trees.

        Returns:
            Sum of the first two integer leaves or the first leaf.
        """
        first, second = leaves[0], leaves[1]
        if isinstance(first, int) and isinstance(second, int):
            return first + second
        return first

    doubled = tree_map(_double_leaves, tree)
    assert doubled == {"a": [2, 4], "b": (6, 8)}

    added = tree_map(_sum_leaves, tree, doubled)
    assert added == {"a": [3, 6], "b": (9, 12)}

    with pytest.raises(ValueError, match="PyTree structure mismatch"):
        tree_map(_sum_leaves, tree, {"a": [1, 2]})

    assert tree_all({"a": True, "b": [1, True], "c": ZeroArray([1, 1], shape=(2,))})
    assert not tree_all({"a": True, "b": [0, True]})
    assert not tree_all({"a": ZeroArray([1, 0], shape=(2,))})

    leaves_flat, def_flat = tree_flatten(5)
    assert leaves_flat == [5]
    assert tree_unflatten(def_flat, [9]) == 9

    @dataclass
    class CustomNode(MappableDataclassMixin):
        """Dataclass testing _is_leaf_value and custom PyTree traversal."""

        val: int = 1

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
    mixed_arr = ZeroArray.zeros((2,), device=[gpu_dev, cpu_dev])
    sharded_arr = ZeroArray.zeros((2,), device=cpu_dev, sharding=sharding)

    assert not _check_sharding("plain_string")
    assert not _check_sharding(cpu_arr)
    assert _check_sharding(sharded_arr)

    assert_tree_is_on_device({"a": gpu_arr}, platform="gpu")
    assert_tree_is_on_device({"a": gpu_arr}, platform=("gpu", "tpu"), device=gpu_dev)
    with pytest.raises(AssertionError, match="not placed on any device"):
        assert_tree_is_on_device({"a": 42}, platform="gpu")
    with pytest.raises(AssertionError, match="empty devices"):
        assert_tree_is_on_device({"a": _EmptyDevicesHolder()}, platform="gpu")
    with pytest.raises(AssertionError, match="expected all in"):
        assert_tree_is_on_device({"a": cpu_arr}, platform="gpu")
    with pytest.raises(AssertionError, match="expected all in"):
        assert_tree_is_on_device({"a": mixed_arr}, platform="gpu")
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
    with pytest.raises(AssertionError, match="different structures"):
        assert_trees_all_equal_structs(t1, [ZeroArray([1.0, 2.0], shape=(2,))])
    with pytest.raises(AssertionError, match="different structures"):
        assert_trees_all_equal_structs(t1, {"b": ZeroArray([1.0, 2.0], shape=(2,))})

    assert_trees_all_equal_comparator(lambda a, b: bool(a == b), lambda a, b: "err", t1)
    assert_trees_all_equal_comparator(lambda a, b: bool(a == b), lambda a, b: "err", t1, t1)

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

    # ULP tests across magnitudes (1e-20, 1.0, 1e10, negative, sign boundary across 0, float64 overflow, complex)
    assert_trees_all_close_ulp(t1, t2, maxulp=1)
    assert_trees_all_close_ulp({"a": 1e10}, {"a": _next_float32(1e10, 1)}, maxulp=1)
    assert_trees_all_close_ulp({"a": 1e-20}, {"a": _next_float32(1e-20, 1)}, maxulp=1)
    assert_trees_all_close_ulp({"a": -2.0}, {"a": _next_float32(-2.0, 1)}, maxulp=1)
    assert _float32_ulp_diff(-0.0, 0.0) == 0
    pos_min = _next_float32(0.0, 1)
    neg_min = -pos_min
    assert _float32_ulp_diff(neg_min, pos_min) == 2
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
        assert_trees_all_equal({"a": True}, {"a": 1}, strict=False)
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


def test_cross_ecosystem_pytree_integration(tmp_path: pathlib.Path) -> None:
    """Test zero-chex PyTree assertions and traversal across zero-flax, zero-optax, zero-grain, and zero-orbax."""
    # 1. zero-flax.nnx.State with Param and BatchStat variables & FrozenDict
    w1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b1 = jnp.array([0.5, -0.5])
    mean1 = jnp.array([0.0, 0.0])
    flax_state_1 = State(
        {
            "linear": State({"kernel": Param(w1), "bias": Param(b1)}),
            "bn": State({"mean": BatchStat(mean1)}),
        }
    )
    flax_state_2 = State(
        {
            "linear": State(
                {"kernel": Param(jnp.array([[1.0, 2.0], [3.0, 4.0]])), "bias": Param(jnp.array([0.5, -0.5]))}
            ),
            "bn": State({"mean": BatchStat(jnp.array([0.0, 0.0]))}),
        }
    )

    assert_trees_all_equal_structs(flax_state_1, flax_state_2)
    assert_trees_all_equal_shapes(flax_state_1, flax_state_2)
    assert_trees_all_equal_dtypes(flax_state_1, flax_state_2)
    assert_trees_all_close(flax_state_1, flax_state_2)
    assert_tree_has_only_ndarrays(flax_state_1)

    def _identity_leaf(*leaves: LeafValue) -> LeafValue:
        """Return first leaf unchanged.

        Args:
            *leaves: Input leaf values.

        Returns:
            First leaf value.
        """
        return leaves[0]

    mapped_state = tree_map(_identity_leaf, flax_state_1)
    assert isinstance(mapped_state, State)
    assert_trees_all_equal_structs(flax_state_1, mapped_state)

    fdict_1 = FrozenDict({"w": w1, "b": b1})
    fdict_2 = FrozenDict({"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, -0.5])})
    assert_trees_all_equal_structs(fdict_1, fdict_2)
    assert_trees_all_close(fdict_1, fdict_2)
    mapped_fdict = tree_map(_identity_leaf, fdict_1)
    assert isinstance(mapped_fdict, FrozenDict)

    assert tree_all(State({"p": Param(jnp.array([1.0, 2.0]))}))
    assert not tree_all(State({"p": Param(jnp.array([1.0, 0.0]))}))

    # 2. zero-optax optimizer states (adam.init(params), TraceState, ScaleByScheduleState)
    ct1 = CompilerTensor([[1.0, 2.0]], TensorConfig(shape=(1, 2), dtype="float32"))
    ct2 = CompilerTensor([[1.0, 2.0]], TensorConfig(shape=(1, 2), dtype="float32"))
    assert_trees_all_close({"t": ct1}, {"t": ct2})
    opt = adam(learning_rate=1e-3)
    opt_params = {"w": as_tensor_obj([[1.0, 2.0], [3.0, 4.0]]), "b": as_tensor_obj([0.5, -0.5])}
    assert_trees_all_close(opt_params, {"w": as_tensor_obj([[1.0, 2.0], [3.0, 4.0]]), "b": as_tensor_obj([0.5, -0.5])})
    opt_state_1 = opt.init(opt_params)
    opt_state_2 = opt.init(opt_params)
    assert_trees_all_equal_structs(opt_state_1, opt_state_2)
    assert_trees_all_close(opt_state_1, opt_state_2)
    mapped_opt_state = tree_map(_identity_leaf, opt_state_1)
    assert_trees_all_equal_structs(opt_state_1, mapped_opt_state)

    trace_1 = TraceState(trace={"w": as_tensor_obj([1.0, 2.0])})
    trace_2 = TraceState(trace={"w": as_tensor_obj([1.0, 2.0])})
    assert_trees_all_equal_structs(trace_1, trace_2)
    assert_trees_all_close(trace_1, trace_2)

    sched_1 = ScaleByScheduleState(count=as_tensor_obj(0))
    sched_2 = ScaleByScheduleState(count=as_tensor_obj(0))
    assert_trees_all_equal_structs(sched_1, sched_2)
    assert_trees_all_close(sched_1, sched_2)

    # 3. zero-grain DataLoader batched PyTrees & SharedMemoryArray
    records: list[ElementTree] = [
        {"x": [1.0, 2.0], "y": 0},
        {"x": [3.0, 4.0], "y": 1},
        {"x": [5.0, 6.0], "y": 0},
        {"x": [7.0, 8.0], "y": 1},
    ]
    loader = DataLoader(
        data_source=InMemoryDataSource(records),
        sampler=SequentialSampler(num_records=4),
        operations=[Batch(batch_size=2)],
    )
    batches = [b for b in loader if isinstance(b, dict)]
    assert len(batches) == 2
    b0 = {k: v for k, v in batches[0].items() if isinstance(v, list)}
    b1_batch = {k: v for k, v in batches[1].items() if isinstance(v, list)}
    assert_trees_all_equal_shapes(b0, b1_batch)

    shm_arr = SharedMemoryArray((3,), "float32")
    shm_meta = shm_arr.metadata
    try:
        assert_tree_has_only_ndarrays({"arr": shm_arr, "meta": shm_meta})
        assert_trees_all_equal_shapes({"arr": shm_arr}, {"arr": ZeroArray([1.0, 2.0, 3.0], shape=(3,))})
        assert_trees_all_equal_dtypes({"arr": shm_meta}, {"arr": ZeroArray([1.0, 2.0, 3.0], shape=(3,))})
    finally:
        shm_arr.unlink_on_del()
        shm_meta.close_and_unlink()

    # 4. zero-orbax.checkpoint.PyTreeCheckpointer save and restore
    ckpt_dir = tmp_path / "orbax_ckpt"
    checkpointer = PyTreeCheckpointer()
    original_tree = {"weights": [1.0, 2.0, 3.0], "step": 5}
    checkpointer.save(ckpt_dir, original_tree)
    raw_restored = PyTreeCheckpointHandler().restore(ckpt_dir)
    assert isinstance(raw_restored, dict)
    restored_tree = {
        str(k): ([float(x) for x in v if isinstance(x, (int, float))] if isinstance(v, list) else int(v))
        for k, v in raw_restored.items()
        if isinstance(v, (list, int, float))
    }
    assert_trees_all_close(original_tree, restored_tree)
