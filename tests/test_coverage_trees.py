import pytest
import ml_switcheroo_compiler.ops as np
import ml_switcheroo_compiler.core.dtype as dt

import zero_chex as chex


def test_assert_tree_all_finite_fail():
    with pytest.raises(AssertionError):
        chex.assert_tree_all_finite({"a": float("inf")})


def test_assert_tree_has_only_ndarrays_fail():
    with pytest.raises(AssertionError):
        chex.assert_tree_has_only_ndarrays({"a": "not an array"})


def test_assert_tree_no_nones_fail():
    with pytest.raises(AssertionError):
        chex.assert_tree_no_nones({"a": None})


def test_assert_tree_shape_prefix_fail():
    with pytest.raises(AssertionError):
        chex.assert_tree_shape_prefix(
            {"a": np.ones((2, 3)), "b": np.ones((3, 3))}, (2,)
        )


def test_assert_tree_shape_suffix_fail():
    with pytest.raises(AssertionError):
        chex.assert_tree_shape_suffix(
            {"a": np.ones((3, 2)), "b": np.ones((3, 3))}, (2,)
        )


def test_assert_trees_all_close_fail():
    with pytest.raises(AssertionError):
        chex.assert_trees_all_close(
            {"a": np.ones((1,))}, {"a": np.add(np.ones((1,)), np.ones((1,)))}
        )


def test_assert_trees_all_equal_fail():
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal(
            {"a": np.ones((1,))}, {"a": np.add(np.ones((1,)), np.ones((1,)))}
        )


def test_assert_trees_all_equal_comparator_fail():
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal_comparator(
            lambda x, y: x == y, lambda x, y: "err", {"a": 1}, {"a": 2}
        )


def test_assert_trees_all_equal_shapes_fail():
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal_shapes({"a": np.ones(2)}, {"a": np.ones(3)})


def test_assert_trees_all_equal_dtypes_fail():
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal_dtypes(
            {"a": np.ones(2, dtype=dt.DType.Int32)},
            {"a": np.ones(2, dtype=dt.DType.Float32)},
        )


def test_assert_tree_is_on_device_fail():
    class DummyDevice:
        def __init__(self, p):
            self.platform = p

    class DummyLeaf:
        def devices(self):
            return [DummyDevice("cpu")]

    with pytest.raises(AssertionError):
        chex.assert_tree_is_on_device({"a": DummyLeaf()}, "gpu")


def test_assert_tree_is_sharded_fail():
    class DummyLeafUnsharded:
        pass

    with pytest.raises(AssertionError):
        chex.assert_tree_is_sharded({"a": DummyLeafUnsharded()})


def test_assert_trees_all_equal_sizes_fail():
    with pytest.raises(AssertionError):
        chex.assert_trees_all_equal_sizes({"a": np.ones(2)}, {"a": np.ones(3)})


def test_assert_tree_is_on_host_fail():
    class DummyDevice:
        def __init__(self, p):
            self.platform = p

    class DummyLeaf:
        def devices(self):
            return [DummyDevice("gpu")]

    with pytest.raises(AssertionError):
        chex.assert_tree_is_on_host({"a": DummyLeaf()})


def test_assert_trees_all_equal_structs_fail():
    with pytest.raises(ValueError):
        chex.assert_trees_all_equal_structs({"a": 1}, [1])


def test_check_sharding_true():
    class DummySharding:
        pass

    class DummyLeafSharded:
        def __init__(self):
            self.sharding = DummySharding()
            self.sharding.__class__.__name__ = "PmapSharding"

    assert chex._src.asserts.trees._check_sharding(DummyLeafSharded())


def test_assert_tree_all_finite_pass():
    import zero_chex as chex

    chex.assert_tree_all_finite({"a": 1.0})


def test_assert_tree_all_finite_tensor_fail_mocked(mocker):
    import zero_chex as chex

    mocker.patch("ml_switcheroo_compiler.ops.all", return_value=False)
    import pytest

    with pytest.raises(AssertionError):
        chex.assert_tree_all_finite(
            {"a": __import__("ml_switcheroo_compiler").ops.ones((1,))}
        )


def test_assert_tree_all_finite_scalar_nan():
    from zero_chex import assert_tree_all_finite
    import pytest

    with pytest.raises(AssertionError):
        assert_tree_all_finite([float("nan")])


def test_assert_tree_all_finite_scalar_nan_mocked(mocker):
    from zero_chex import assert_tree_all_finite
    import pytest

    mocker.patch("ml_switcheroo_compiler.ops.isfinite", side_effect=TypeError)
    with pytest.raises(AssertionError):
        assert_tree_all_finite([float("nan")])
