"""Tree assertions."""

import ml_switcheroo.ops as jnp
import math
from zero_chex._src.tree_util import tree_leaves


def _get_leaves(tree):
    """Retrieves all leaves from a PyTree.

    Args:
        tree: The PyTree to extract leaves from.

    Returns:
        A list of leaf values from the tree.
    """
    return tree_leaves(tree)


def _check_sharding(leaf):
    """Checks if a leaf is sharded.

    Args:
        leaf: The tree leaf to check.

    Returns:
        True if the leaf has a PmapSharding, False otherwise.
    """
    if hasattr(leaf, "sharding") and type(leaf.sharding).__name__ == "PmapSharding":
        return True
    return False


def assert_tree_is_on_device(tree, platform=None):
    """Asserts that all leaves in a tree are on the specified device platform.

    Args:
        tree: The PyTree to check.
        platform: The target device platform (e.g., 'cpu', 'gpu', 'tpu'). If None, device checking is skipped.

    Returns:
        None

    Raises:
        AssertionError: If any leaf in the tree is not on the expected platform.
    """
    for leaf in _get_leaves(tree):
        devs = getattr(leaf, "devices", lambda: [])()
        if platform and all(d.platform != platform for d in devs):
            raise AssertionError("expected '(%s,)'" % platform)


def assert_tree_is_sharded(tree, devices=None):
    """Asserts that all leaves in a tree are sharded.

    Args:
        tree: The PyTree to check.
        devices: (Optional) The specific devices the tree should be sharded across.

    Returns:
        None

    Raises:
        AssertionError: If any leaf in the tree is not sharded.
    """
    for leaf in _get_leaves(tree):
        if not _check_sharding(leaf):
            raise AssertionError("is sharded across")


def assert_trees_all_close(tree1, tree2, rtol=1e-5, atol=1e-8):
    """Asserts that two trees are approximately equal.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.

    Returns:
        None

    Raises:
        AssertionError: If the leaves of the trees are not approximately equal within the specified tolerances.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.allclose(l1, l2, rtol=rtol, atol=atol):
            raise AssertionError()


def assert_trees_all_close_ulp(tree1, tree2, maxulp=1):
    """Asserts that two trees are equal within a specified number of Units in the Last Place (ULP).

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.
        maxulp: The maximum allowed difference in ULP.

    Returns:
        None

    Raises:
        AssertionError: If the leaves differ by more than maxulp.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.allclose(l1, l2, rtol=0, atol=maxulp * 1e-7):
            raise AssertionError()


def assert_trees_all_equal(tree1, tree2, strict=False):
    """Asserts that all corresponding leaves in two trees are exactly equal.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.
        strict: If True, enforce strict equality of dtypes as well.

    Returns:
        None

    Raises:
        AssertionError: If any corresponding leaves are not exactly equal.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.array_equal(l1, l2):
            raise AssertionError()


def assert_tree_all_finite(tree):
    """Asserts that all leaves in a tree contain only finite values (no NaN or Inf).

    Args:
        tree: The PyTree to check.

    Returns:
        None

    Raises:
        AssertionError: If any leaf contains non-finite values.
    """
    for leaf in _get_leaves(tree):
        if not jnp.all(jnp.isfinite(leaf)):
            raise AssertionError()


def assert_tree_has_only_ndarrays(tree):
    """Asserts that all leaves in a tree are ndarray-like objects (having a 'shape' attribute).

    Args:
        tree: The PyTree to check.

    Returns:
        None

    Raises:
        AssertionError: If any leaf does not have a 'shape' attribute.
    """
    for leaf in _get_leaves(tree):
        if not hasattr(leaf, "shape"):
            raise AssertionError()


def assert_tree_is_on_host(tree, allow_cpu_device=True):
    """Asserts that all leaves in a tree are located on the host device (CPU).

    Args:
        tree: The PyTree to check.
        allow_cpu_device: If True, considers 'cpu' devices as host.

    Returns:
        None

    Raises:
        AssertionError: If any leaf is not on the host device.
    """
    for leaf in _get_leaves(tree):
        if not hasattr(leaf, "devices"):
            continue
        devs = getattr(leaf, "devices", lambda: [])()
        if not allow_cpu_device and all(d.platform == "cpu" for d in devs):
            raise AssertionError()
        if any(d.platform != "cpu" for d in devs):
            raise AssertionError()


def assert_tree_no_nones(tree):
    """Asserts that no leaves in the tree are None.

    Args:
        tree: The PyTree to check.

    Returns:
        None

    Raises:
        AssertionError: If any leaf is None.
    """
    for leaf in _get_leaves(tree):
        if leaf is None:
            raise AssertionError()


def assert_tree_shape_prefix(tree, prefix):
    """Asserts that the shape of all leaves in a tree starts with the specified prefix.

    Args:
        tree: The PyTree to check.
        prefix: The expected shape prefix as a tuple or list.

    Returns:
        None

    Raises:
        AssertionError: If any leaf's shape does not start with the prefix.
    """
    for leaf in _get_leaves(tree):
        s = getattr(leaf, "shape", jnp.array(leaf).shape)
        if s[: len(prefix)] != tuple(prefix):
            raise AssertionError()


def assert_tree_shape_suffix(tree, suffix):
    """Asserts that the shape of all leaves in a tree ends with the specified suffix.

    Args:
        tree: The PyTree to check.
        suffix: The expected shape suffix as a tuple or list.

    Returns:
        None

    Raises:
        AssertionError: If any leaf's shape does not end with the suffix.
    """
    for leaf in _get_leaves(tree):
        s = getattr(leaf, "shape", jnp.array(leaf).shape)
        if len(suffix) > 0 and s[-len(suffix) :] != tuple(suffix):
            raise AssertionError()


def assert_trees_all_equal_structs(tree1, tree2):
    """Asserts that two trees have the exact same structure type.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        None

    Raises:
        ValueError: If the types of the two trees are different.
    """
    if type(tree1) is not type(tree2):
        raise ValueError()


def assert_trees_all_equal_comparator(comp, err, tree1, tree2):
    """Asserts that all corresponding leaves in two trees satisfy a custom comparator.

    Args:
        comp: A callable that takes two leaves and returns a boolean.
        err: A callable that takes two leaves and returns an error message.
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        None

    Raises:
        AssertionError: If the comparator returns False for any pair of leaves.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not comp(l1, l2):
            raise AssertionError(err(l1, l2))


def assert_trees_all_equal_dtypes(tree1, tree2):
    """Asserts that corresponding leaves in two trees have identical data types.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        None

    Raises:
        AssertionError: If corresponding leaves do not have the same dtype.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not hasattr(l1, "dtype") or not hasattr(l2, "dtype"):
            raise AssertionError("is not a (j-)np array")
        if getattr(l1, "dtype", type(l1)) != getattr(l2, "dtype", type(l2)):
            raise AssertionError()


def assert_trees_all_equal_shapes(tree1, tree2):
    """Asserts that corresponding leaves in two trees have identical shapes.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        None

    Raises:
        AssertionError: If corresponding leaves do not have the same shape.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if getattr(l1, "shape", ()) != getattr(l2, "shape", ()):
            raise AssertionError()


def assert_trees_all_equal_sizes(tree1, tree2):
    """Asserts that corresponding leaves in two trees have identical total sizes (number of elements).

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        None

    Raises:
        AssertionError: If corresponding leaves do not have the same total size.
    """
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if math.prod(getattr(l1, "shape", ())) != math.prod(getattr(l2, "shape", ())):
            raise AssertionError()


def assert_trees_all_equal_shapes_and_dtypes(tree1, tree2):
    """Asserts that corresponding leaves in two trees have identical shapes and data types.

    Args:
        tree1: The first PyTree.
        tree2: The second PyTree.

    Returns:
        None

    Raises:
        AssertionError: If corresponding leaves differ in either shape or dtype.
    """
    assert_trees_all_equal_shapes(tree1, tree2)
    assert_trees_all_equal_dtypes(tree1, tree2)
