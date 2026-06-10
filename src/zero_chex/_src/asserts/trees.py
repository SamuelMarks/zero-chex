"""Module docstring."""

import ml_switcheroo.ops as jnp
import math
from zero_chex._src.tree_util import tree_leaves


def _get_leaves(tree):
    """Docstring."""
    return tree_leaves(tree)


def _check_sharding(leaf):
    """Docstring."""
    if hasattr(leaf, "sharding") and type(leaf.sharding).__name__ == "PmapSharding":
        return True
    return False


def assert_tree_is_on_device(tree, platform=None):
    """Docstring."""
    for leaf in _get_leaves(tree):
        devs = getattr(leaf, "devices", lambda: [])()
        if platform and all(d.platform != platform for d in devs):
            raise AssertionError("expected '(%s,)'" % platform)


def assert_tree_is_sharded(tree, devices=None):
    """Docstring."""
    for leaf in _get_leaves(tree):
        if not _check_sharding(leaf):
            raise AssertionError("is sharded across")


def assert_trees_all_close(tree1, tree2, rtol=1e-5, atol=1e-8):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.allclose(l1, l2, rtol=rtol, atol=atol):
            raise AssertionError()


def assert_trees_all_close_ulp(tree1, tree2, maxulp=1):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.allclose(l1, l2, rtol=0, atol=maxulp * 1e-7):
            raise AssertionError()


def assert_trees_all_equal(tree1, tree2, strict=False):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.array_equal(l1, l2):
            raise AssertionError()


def assert_tree_all_finite(tree):
    """Docstring."""
    for leaf in _get_leaves(tree):
        if not jnp.all(jnp.isfinite(leaf)):
            raise AssertionError()


def assert_tree_has_only_ndarrays(tree):
    """Docstring."""
    for leaf in _get_leaves(tree):
        if not hasattr(leaf, "shape"):
            raise AssertionError()


def assert_tree_is_on_host(tree, allow_cpu_device=True):
    """Docstring."""
    for leaf in _get_leaves(tree):
        if not hasattr(leaf, "devices"):
            continue
        devs = getattr(leaf, "devices", lambda: [])()
        if not allow_cpu_device and all(d.platform == "cpu" for d in devs):
            raise AssertionError()
        if any(d.platform != "cpu" for d in devs):
            raise AssertionError()


def assert_tree_no_nones(tree):
    """Docstring."""
    for leaf in _get_leaves(tree):
        if leaf is None:
            raise AssertionError()


def assert_tree_shape_prefix(tree, prefix):
    """Docstring."""
    for leaf in _get_leaves(tree):
        s = getattr(leaf, "shape", jnp.array(leaf).shape)
        if s[: len(prefix)] != tuple(prefix):
            raise AssertionError()


def assert_tree_shape_suffix(tree, suffix):
    """Docstring."""
    for leaf in _get_leaves(tree):
        s = getattr(leaf, "shape", jnp.array(leaf).shape)
        if len(suffix) > 0 and s[-len(suffix) :] != tuple(suffix):
            raise AssertionError()


def assert_trees_all_equal_structs(tree1, tree2):
    """Docstring."""
    if type(tree1) is not type(tree2):
        raise ValueError()


def assert_trees_all_equal_comparator(comp, err, tree1, tree2):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not comp(l1, l2):
            raise AssertionError(err(l1, l2))


def assert_trees_all_equal_dtypes(tree1, tree2):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not hasattr(l1, "dtype") or not hasattr(l2, "dtype"):
            raise AssertionError("is not a (j-)np array")
        if getattr(l1, "dtype", type(l1)) != getattr(l2, "dtype", type(l2)):
            raise AssertionError()


def assert_trees_all_equal_shapes(tree1, tree2):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if getattr(l1, "shape", ()) != getattr(l2, "shape", ()):
            raise AssertionError()


def assert_trees_all_equal_sizes(tree1, tree2):
    """Docstring."""
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if math.prod(getattr(l1, "shape", ())) != math.prod(getattr(l2, "shape", ())):
            raise AssertionError()


def assert_trees_all_equal_shapes_and_dtypes(tree1, tree2):
    """Docstring."""
    assert_trees_all_equal_shapes(tree1, tree2)
    assert_trees_all_equal_dtypes(tree1, tree2)
