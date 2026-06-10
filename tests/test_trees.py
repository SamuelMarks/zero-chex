import pytest
import ml_switcheroo.ops as jnp


def test_trees():
    from zero_chex import (
        assert_tree_is_on_device,
        assert_tree_is_sharded,
        assert_trees_all_close,
        assert_trees_all_close_ulp,
        assert_trees_all_equal,
        assert_tree_all_finite,
        assert_tree_has_only_ndarrays,
        assert_tree_is_on_host,
        assert_tree_no_nones,
        assert_tree_shape_prefix,
        assert_tree_shape_suffix,
        assert_trees_all_equal_structs,
        assert_trees_all_equal_comparator,
        assert_trees_all_equal_dtypes,
        assert_trees_all_equal_shapes,
        assert_trees_all_equal_sizes,
    )

    class DummyDev:
        platform = "cpu"

    class DummyLeaf:
        def devices(self):
            return [DummyDev()]

    class DummyPmapSharding:
        pass

    DummyPmapSharding.__name__ = "PmapSharding"

    class DummyLeafSharded:
        sharding = DummyPmapSharding()

    # Create dummy trees
    tree1 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
    tree2 = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0])}
    tree_host = {"a": DummyLeaf(), "b": DummyLeaf()}
    tree_sharded = {"a": DummyLeafSharded(), "b": DummyLeafSharded()}

    assert_tree_is_on_device(tree_host, "cpu")
    assert_tree_is_sharded(tree_sharded)
    assert_trees_all_close(tree1, tree2)
    assert_trees_all_close_ulp(tree1, tree2)
    assert_trees_all_equal(tree1, tree2)
    assert_tree_all_finite(tree1)
    assert_tree_has_only_ndarrays(tree1)
    assert_tree_is_on_host(tree_host)
    assert_tree_no_nones(tree1)
    assert_tree_shape_prefix(tree1, ())
    assert_tree_shape_suffix(tree1, ())
    assert_trees_all_equal_structs(tree1, tree2)
    assert_trees_all_equal_comparator(lambda x, y: True, lambda x, y: "", tree1, tree2)
    assert_trees_all_equal_dtypes(tree1, tree2)
    assert_trees_all_equal_shapes(tree1, tree2)
    assert_trees_all_equal_sizes(tree1, tree2)


def test_assert_tree_is_on_host_fail_cpu():
    from zero_chex import assert_tree_is_on_host

    class DummyDev:
        platform = "cpu"

    class DummyLeaf:
        def devices(self):
            return [DummyDev()]

    with pytest.raises(AssertionError):
        assert_tree_is_on_host([DummyLeaf()], allow_cpu_device=False)


def test_assert_trees_all_equal_dtypes_fail_no_dtype():
    from zero_chex import assert_trees_all_equal_dtypes

    with pytest.raises(AssertionError):
        assert_trees_all_equal_dtypes([1], [1])


def test_assert_trees_all_equal_shapes_and_dtypes():
    from zero_chex import assert_trees_all_equal_shapes_and_dtypes
    import ml_switcheroo.ops as jnp

    assert_trees_all_equal_shapes_and_dtypes([jnp.zeros((2,))], [jnp.zeros((2,))])
