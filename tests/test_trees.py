# ruff: noqa: F821, F403, F405, E402
"""Tests for tree assertions."""

from unittest.mock import Mock, patch


def test_check_sharding():
    from zero_chex._src.asserts.trees import _check_sharding

    # Test PmapSharding
    mock_leaf = Mock()
    mock_leaf.sharding = Mock()
    type(mock_leaf.sharding).__name__ = "PmapSharding"
    assert _check_sharding(mock_leaf) is True

    # Test other sharding with multiple devices
    mock_leaf2 = Mock()
    mock_leaf2.sharding = Mock()
    type(mock_leaf2.sharding).__name__ = "OtherSharding"
    mock_leaf2.sharding.device_set = [1, 2]
    assert _check_sharding(mock_leaf2) is True

    # Test Exception
    mock_leaf3 = Mock()
    del mock_leaf3.sharding
    mock_leaf3.devices.side_effect = Exception("err")
    assert _check_sharding(mock_leaf3) is False

    # Test len devices > 1
    mock_leaf4 = Mock()
    mock_leaf4.sharding = None
    mock_leaf4.devices.return_value = [1, 2]
    assert _check_sharding(mock_leaf4) is True


def test_assert_tree_is_on_device_mocked():
    mock_leaf = Mock()
    mock_leaf.__class__ = np.ndarray
    mock_leaf.devices.return_value = [Mock(platform="cpu")]
    mock_leaf.sharding = None
    with pytest.raises(AssertionError, match="expected '\\('gpu',\\)'"):
        assert_tree_is_on_device({"a": mock_leaf}, platform="gpu")

    mock_sharded = Mock()
    mock_sharded.__class__ = np.ndarray
    mock_sharded.sharding = Mock()
    type(mock_sharded.sharding).__name__ = "PmapSharding"
    del mock_sharded.addressable_shards
    with pytest.raises(AssertionError, match="ShardedDeviceArray which are disallowed"):
        assert_tree_is_on_device({"a": mock_sharded}, platform="cpu")

    mock_wrong_device = Mock()
    mock_wrong_device.__class__ = np.ndarray
    mock_dev = Mock(platform="cpu")
    mock_wrong_device.devices.return_value = {mock_dev}
    mock_wrong_device.sharding = None
    with pytest.raises(AssertionError, match="expected Mock"):
        assert_tree_is_on_device(
            {"a": mock_wrong_device},
            device=Mock(platform="cpu", __repr__=lambda self: "Mock"),
        )


def test_assert_tree_is_sharded_mocked():
    mock_sharded = Mock()
    mock_sharded.__class__ = np.ndarray
    mock_sharded.sharding = Mock()
    type(mock_sharded.sharding).__name__ = "PmapSharding"
    mock_dev1 = Mock()
    mock_dev2 = Mock()
    mock_sharded.addressable_shards = [Mock(device=mock_dev1)]

    with pytest.raises(AssertionError, match="is sharded across"):
        assert_tree_is_sharded({"a": mock_sharded}, devices=[mock_dev2])

    del mock_sharded.addressable_shards
    with pytest.raises(AssertionError, match="is not sharded"):
        assert_tree_is_sharded({"a": mock_sharded}, devices=[mock_dev2])


def test_assert_trees_all_close_mocked():
    with patch("numpy.testing.assert_allclose", side_effect=AssertionError("msg")):
        with pytest.raises(AssertionError):
            assert_trees_all_close(
                {"a": jnp.array(1)}, {"a": jnp.array(2)}, rtol=1e-6, atol=0.0
            )


def test_assert_trees_all_close_ulp_mocked():
    with patch("numpy.testing.assert_array_max_ulp", side_effect=AssertionError("msg")):
        with pytest.raises(AssertionError):
            assert_trees_all_close_ulp(
                {"a": jnp.array(1)}, {"a": jnp.array(2)}, maxulp=1
            )


def test_assert_trees_all_equal_mocked():
    with patch("numpy.testing.assert_array_equal", side_effect=AssertionError("msg")):
        with pytest.raises(AssertionError):
            assert_trees_all_equal(
                {"a": jnp.array(1)}, {"a": jnp.array(2)}, strict=False
            )


import zero_jax.numpy as jnp
import numpy as np
import pytest

from zero_chex import (
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
)


def test_assert_tree_all_finite():
    assert_tree_all_finite({"a": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)})
    with pytest.raises(AssertionError):
        assert_tree_all_finite({"a": jnp.array([1.0, jnp.inf])})


def test_assert_tree_has_only_ndarrays():
    assert_tree_has_only_ndarrays({"a": jnp.array(1), "b": np.array(2)})
    with pytest.raises(AssertionError):
        assert_tree_has_only_ndarrays({"a": jnp.array(1), "b": 2})


def test_assert_tree_is_on_device():
    # Only have CPU usually in test env, so we'll patch check_sharding
    # Actually, CPU is not allowed by default if we ask for GPU/TPU
    # Let's test failure
    with pytest.raises(AssertionError):
        mock_leaf2 = Mock()
        mock_leaf2.__class__ = np.ndarray
        mock_leaf2.devices = lambda: [Mock(platform="cpu")]
        assert_tree_is_on_device({"a": mock_leaf2})

    # Test passing
    mock_leaf3 = Mock()
    mock_leaf3.__class__ = np.ndarray
    mock_leaf3.devices = lambda: [Mock(platform="cpu")]
    assert_tree_is_on_device({"a": mock_leaf3}, platform="cpu")
    assert_tree_is_on_device({"a": mock_leaf3}, platform=["cpu"])

    # Test non-ndarray
    with pytest.raises(AssertionError):
        assert_tree_is_on_device({"a": 1})

    class DummyDevice:
        platform = "cpu"

        def __repr__(self):
            return "DummyDevice"

    device = DummyDevice()
    with pytest.raises(AssertionError):  # jnp.array is on real cpu device
        assert_tree_is_on_device({"a": mock_leaf3}, device=device)  # type: ignore


def test_assert_tree_is_on_host():
    mock_leaf4 = Mock()
    mock_leaf4.__class__ = np.ndarray
    mock_leaf4.devices = lambda: [Mock(platform="cpu")]
    assert_tree_is_on_host({"a": mock_leaf4})

    mock_leaf_fail = Mock()
    mock_leaf_fail.__class__ = np.ndarray
    mock_leaf_fail.devices = lambda: [Mock(platform="gpu")]
    assert_tree_is_on_host({"a": mock_leaf_fail})
    with pytest.raises(AssertionError):
        assert_tree_is_on_host({"a": jnp.array(1)}, allow_cpu_device=False)
        assert_tree_is_on_host({"a": 1})


def test_assert_tree_is_sharded():
    with pytest.raises(AssertionError):
        assert_tree_is_sharded({"a": jnp.array(1)}, devices=[])  # not sharded


def test_assert_tree_no_nones():
    assert_tree_no_nones({"a": 1, "b": {"c": 2}})
    with pytest.raises(AssertionError):
        assert_tree_no_nones({"a": 1, "b": None})


def test_assert_tree_shape_prefix():
    assert_tree_shape_prefix({"a": jnp.zeros((2, 3)), "b": jnp.zeros((2, 4))}, [2])
    assert_tree_shape_prefix({"a": jnp.zeros((2, 3))}, [])
    with pytest.raises(AssertionError):
        assert_tree_shape_prefix({"a": jnp.zeros((2, 3)), "b": jnp.zeros((3, 4))}, [2])
    with pytest.raises(AssertionError):
        assert_tree_shape_prefix({"a": jnp.zeros((2, 3))}, [2, 3, 4])


def test_assert_tree_shape_suffix():
    assert_tree_shape_suffix({"a": jnp.zeros((2, 3)), "b": jnp.zeros((4, 3))}, [3])
    assert_tree_shape_suffix({"a": jnp.zeros((2, 3))}, [])
    with pytest.raises(AssertionError):
        assert_tree_shape_suffix({"a": jnp.zeros((2, 3)), "b": jnp.zeros((4, 2))}, [3])
    with pytest.raises(AssertionError):
        assert_tree_shape_suffix({"a": jnp.zeros((2, 3))}, [2, 3, 4])


def test_assert_trees_all_equal_structs():
    assert_trees_all_equal_structs({"a": 1}, {"a": 2})
    with pytest.raises(ValueError):
        assert_trees_all_equal_structs({"a": 1})
    with pytest.raises(AssertionError):
        assert_trees_all_equal_structs({"a": 1}, {"b": 2})


def test_assert_trees_all_equal_comparator():
    assert_trees_all_equal_comparator(
        lambda x, y: x == y, lambda x, y: "err", {"a": 1}, {"a": 1}
    )
    with pytest.raises(AssertionError):
        assert_trees_all_equal_comparator(
            lambda x, y: x == y, lambda x, y: "err", {"a": 1}, {"a": 2}
        )

    with pytest.raises(ValueError):
        assert_trees_all_equal_comparator(
            lambda x, y: x == y, lambda x, y: "err", {"a": 1}
        )

    with pytest.raises(AssertionError, match="Trees \\(arrays\\) 0 and 1 differ: err"):
        assert_trees_all_equal_comparator(
            lambda x, y: x == y, lambda x, y: "err", jnp.array(1), jnp.array(2)
        )


def test_assert_trees_all_close():
    assert_trees_all_close(
        {"a": jnp.array([1.0])}, {"a": jnp.array([1.0])}, rtol=1e-6, atol=0.0
    )
    with pytest.raises(AssertionError):
        assert_trees_all_close(
            {"a": jnp.array([1.0])}, {"a": jnp.array([2.0])}, rtol=1e-6, atol=0.0
        )


def test_assert_trees_all_close_ulp():
    assert_trees_all_close_ulp(
        {"a": jnp.array([1.0])}, {"a": jnp.array([1.0])}, maxulp=1
    )
    with pytest.raises(AssertionError):
        assert_trees_all_close_ulp(
            {"a": jnp.array([1.0])}, {"a": jnp.array([2.0])}, maxulp=1
        )


def test_assert_trees_all_equal():
    assert_trees_all_equal({"a": jnp.array([1])}, {"a": jnp.array([1])}, strict=False)
    with pytest.raises(AssertionError):
        assert_trees_all_equal(
            {"a": jnp.array([1])}, {"a": jnp.array([2])}, strict=False
        )


def test_assert_trees_all_equal_dtypes():
    assert_trees_all_equal_dtypes({"a": jnp.array([1])}, {"a": jnp.array([2])})
    with pytest.raises(AssertionError):
        assert_trees_all_equal_dtypes(
            {"a": jnp.array([1], dtype=np.int32)},
            {"a": jnp.array([2], dtype=np.float32)},
        )
    with pytest.raises(AssertionError, match="is not a \\(j-\\)?np array"):
        assert_trees_all_equal_dtypes({"a": 1}, {"a": jnp.array([2])})  # no dtype
    with pytest.raises(AssertionError, match="is not a \\(j-\\)?np array"):
        assert_trees_all_equal_dtypes({"a": jnp.array([1])}, {"a": 2})  # arr_2 no dtype


def test_assert_trees_all_equal_shapes():
    assert_trees_all_equal_shapes({"a": jnp.zeros((2,))}, {"a": jnp.zeros((2,))})
    with pytest.raises(AssertionError):
        assert_trees_all_equal_shapes({"a": jnp.zeros((2,))}, {"a": jnp.zeros((3,))})


def test_assert_trees_all_equal_sizes():
    assert_trees_all_equal_sizes({"a": jnp.zeros((2,))}, {"a": jnp.zeros((2,))})
    with pytest.raises(AssertionError):
        assert_trees_all_equal_sizes({"a": jnp.zeros((2,))}, {"a": jnp.zeros((3,))})


def test_assert_trees_all_equal_shapes_and_dtypes():
    assert_trees_all_equal_shapes_and_dtypes(
        {"a": jnp.zeros((2,))}, {"a": jnp.zeros((2,))}
    )
    with pytest.raises(AssertionError):
        assert_trees_all_equal_shapes_and_dtypes(
            {"a": jnp.zeros((2,))}, {"a": jnp.zeros((3,))}
        )
