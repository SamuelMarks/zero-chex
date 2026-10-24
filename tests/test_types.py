"""Tests for zero_chex type definitions, protocols, Device, PmapSharding, ZeroArray, and PyTreeDef."""

from __future__ import annotations

import pytest
from ml_switcheroo_ir import DType

from zero_chex import (
    Array,
    ArrayBatched,
    ArrayDevice,
    ArrayDeviceTree,
    ArrayDType,
    ArrayNumpy,
    ArrayNumpyTree,
    ArraySharded,
    ArrayTree,
    ChexifyChecks,
    Device,
    Numeric,
    PRNGKey,
    PyTreeDef,
    Scalar,
    Shape,
)
from zero_chex.types import (
    ArrayProtocol,
    DeviceProtocol,
    PmapSharding,
    ShardingProtocol,
    ZeroArray,
    is_array,
    is_scalar,
)


def test_public_type_aliases_and_checks() -> None:
    """Verify exported Chex type aliases and ChexifyChecks constants with concrete runtime instances."""
    sample_arr: Array = ZeroArray([1.0, 2.0], shape=(2,))
    sample_batched: ArrayBatched = ZeroArray([[1.0, 2.0]], shape=(1, 2))
    sample_device_arr: ArrayDevice = ZeroArray([1.0], device=Device("gpu", 0))
    sample_device_tree: ArrayDeviceTree = {"w": sample_device_arr}
    sample_dtype: ArrayDType = DType.float32
    sample_np: ArrayNumpy = sample_arr
    sample_np_tree: ArrayNumpyTree = [sample_np]
    sample_sharded: ArraySharded = ZeroArray([1.0], sharding=PmapSharding())
    sample_tree: ArrayTree = {"a": sample_arr, "b": [sample_batched]}
    sample_num: Numeric = 3.14
    sample_key: PRNGKey = (0, 42)
    sample_scalar: Scalar = 7
    sample_shape: Shape = (2, 3)

    assert isinstance(sample_arr, ArrayProtocol)
    assert sample_batched.shape == (1, 2)
    assert sample_device_tree == {"w": sample_device_arr}
    assert sample_dtype == DType.float32
    assert sample_np_tree == [sample_arr]
    assert isinstance(sample_sharded, ArrayProtocol) and sample_sharded.sharding is not None
    assert isinstance(sample_tree, dict) and "a" in sample_tree
    assert sample_num == 3.14
    assert sample_key == (0, 42)
    assert sample_scalar == 7
    assert sample_shape == (2, 3)

    assert ChexifyChecks.user == frozenset({"user"})
    assert ChexifyChecks.nan == frozenset({"nan"})
    assert ChexifyChecks.index == frozenset({"index"})
    assert ChexifyChecks.div == frozenset({"div"})
    assert ChexifyChecks.float == frozenset({"nan", "div"})
    assert ChexifyChecks.automatic == frozenset({"nan", "index", "div"})
    assert ChexifyChecks.all == frozenset({"user", "nan", "index", "div"})


def test_device_and_sharding_classes() -> None:
    """Test Device and PmapSharding implementations and protocols."""
    d_cpu = Device("cpu", 0)
    d_cpu_same = Device("cpu", 0)
    d_gpu = Device("gpu", 1)

    assert isinstance(d_cpu, DeviceProtocol)
    assert d_cpu.platform == "cpu"
    assert d_cpu.id == 0
    assert d_cpu == d_cpu_same
    assert d_cpu != d_gpu
    assert d_cpu != "not_a_device"
    assert hash(d_cpu) == hash(d_cpu_same)
    assert repr(d_cpu) == "Device(platform='cpu', id=0)"

    sharding_default = PmapSharding()
    assert isinstance(sharding_default, ShardingProtocol)
    assert sharding_default.device_set == frozenset({d_cpu})
    assert not sharding_default.is_fully_replicated

    sharding_custom = PmapSharding(devices=[d_cpu, d_gpu], is_fully_replicated=True)
    assert sharding_custom.device_set == frozenset({d_cpu, d_gpu})
    assert sharding_custom.is_fully_replicated


def test_zero_array_methods_and_properties() -> None:
    """Test ZeroArray construction, nested sequences, shape validation, factories, and equality."""
    arr_scalar = ZeroArray(5.0)
    assert arr_scalar.shape == ()
    assert arr_scalar.ndim == 0
    assert arr_scalar.size == 1
    assert arr_scalar.dtype == DType.float32
    assert arr_scalar.sharding is None
    assert arr_scalar.devices() == (Device("cpu", 0),)
    assert arr_scalar.to_list() == [5.0]
    assert arr_scalar == 5.0
    assert arr_scalar != 6.0
    assert arr_scalar != "invalid"
    assert hash(arr_scalar) == hash(ZeroArray(5.0))
    assert is_scalar(arr_scalar)
    with pytest.raises(TypeError, match="unsized 0-D"):
        len(arr_scalar)

    arr_nested = ZeroArray([[1, 2, 3], [4, 5, 6]])
    assert arr_nested.shape == (2, 3)
    assert arr_nested.to_list() == [1, 2, 3, 4, 5, 6]

    with pytest.raises(ValueError, match="Ragged nested sequence"):
        ZeroArray([[1, 2], [3]])
    with pytest.raises(ValueError, match="cannot reshape array"):
        ZeroArray([1, 2, 3], shape=(2, 2))

    arr_empty_seq = ZeroArray([])
    assert arr_empty_seq.shape == (0,)
    assert arr_empty_seq.to_list() == []

    d_gpu = Device("gpu", 0)
    sharding = PmapSharding([d_gpu])
    arr_broadcast = ZeroArray(2.0, shape=(2, 3), device=d_gpu, sharding=sharding)
    assert arr_broadcast.shape == (2, 3)
    assert arr_broadcast.ndim == 2
    assert arr_broadcast.size == 6
    assert len(arr_broadcast) == 2
    assert list(arr_broadcast) == [2.0] * 6
    assert arr_broadcast.devices() == (d_gpu,)
    assert arr_broadcast.sharding is sharding

    arr_multi_dev = ZeroArray([1, 2], shape=(2,), dtype=DType.int32, device=[d_gpu])
    assert arr_multi_dev.devices() == (d_gpu,)
    assert "ZeroArray(" in repr(arr_multi_dev)

    arr_empty = ZeroArray([1.0], shape=(0,))
    assert arr_empty.to_list() == []
    assert arr_empty.size == 0

    z_float = ZeroArray.zeros((2, 2))
    z_int = ZeroArray.zeros(3, dtype=DType.int32)
    o_float = ZeroArray.ones((2, 2))
    o_int = ZeroArray.ones(3, dtype=DType.int32)
    assert z_float.to_list() == [0.0, 0.0, 0.0, 0.0]
    assert z_int.to_list() == [0, 0, 0]
    assert o_float.to_list() == [1.0, 1.0, 1.0, 1.0]
    assert o_int.to_list() == [1, 1, 1]
    assert z_int == ZeroArray([0, 0, 0], shape=(3,), dtype=DType.int32)
    assert isinstance(z_float, ArrayProtocol)
    assert is_array(z_float)
    assert not is_array(42)
    assert is_scalar(42)
    assert not is_scalar("not_a_scalar")
    assert not is_scalar(z_float)


def test_pytreedef_methods() -> None:
    """Test PyTreeDef structural properties, equality, hashing, and repr."""
    leaf = PyTreeDef()
    assert leaf.node_type == "leaf"
    assert leaf.keys == ()
    assert leaf.children == ()
    assert leaf.custom_cls is None
    assert leaf.num_leaves == 1
    assert repr(leaf) == "PyTreeDef(*)"

    node = PyTreeDef(node_type="dict", keys=("a", "b"), children=(leaf, leaf))
    node_same = PyTreeDef(node_type="dict", keys=("a", "b"), children=(leaf, leaf))
    assert node.num_leaves == 2
    assert node == node_same
    assert node != leaf
    assert node != "not_a_treedef"
    assert hash(node) == hash(node_same)
    assert "PyTreeDef(dict" in repr(node)
