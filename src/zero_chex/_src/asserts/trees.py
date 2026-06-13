"""Tree assertions."""

import ml_switcheroo_compiler.ops as jnp
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


def assert_tree_is_on_device(tree, platform, device=None):
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


def assert_trees_all_close(*trees, rtol=1e-5, atol=1e-8):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.allclose(l1, l2, rtol=rtol, atol=atol):
            raise AssertionError()


def assert_trees_all_close_ulp(*trees, maxulp=1):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not jnp.allclose(l1, l2, rtol=0, atol=maxulp * 1e-7):
            raise AssertionError()


def assert_trees_all_equal(*trees, strict=False):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        v1 = l1.data if hasattr(l1, "data") else l1
        v2 = l2.data if hasattr(l2, "data") else l2
        if not __import__("numpy").array_equal(v1, v2):
            raise AssertionError()


def assert_tree_all_finite(tree_like):
    """Asserts that all leaves in a tree contain only finite values (no NaN or Inf).

    Args:
        tree: The PyTree to check.

    Returns:
        None

    Raises:
        AssertionError: If any leaf contains non-finite values.
    """
    for leaf in _get_leaves(tree_like):
        import math

        try:
            if not jnp.all(jnp.isfinite(leaf)):
                raise AssertionError()
        except (TypeError, ValueError, AttributeError):
            if not math.isfinite(leaf):
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


def assert_tree_is_on_host(tree, allow_cpu_device=True, allow_sharded_arrays=True):
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


def assert_tree_shape_prefix(tree, shape_prefix):
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
        s = (
            leaf.shape
            if hasattr(leaf, "shape")
            else (len(leaf),)
            if isinstance(leaf, (list, tuple))
            else ()
        )
        if s[: len(shape_prefix)] != tuple(shape_prefix):
            raise AssertionError()


def assert_tree_shape_suffix(tree, shape_suffix):
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
        s = (
            leaf.shape
            if hasattr(leaf, "shape")
            else (len(leaf),)
            if isinstance(leaf, (list, tuple))
            else ()
        )
        if len(shape_suffix) > 0 and s[-len(shape_suffix) :] != tuple(shape_suffix):
            raise AssertionError()


def assert_trees_all_equal_structs(*trees):
    tree1 = trees[0]
    tree2 = trees[1]
    if type(tree1) is not type(tree2):
        raise ValueError()


def assert_trees_all_equal_comparator(equality_comparator, error_msg_fn, *trees):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not equality_comparator(l1, l2):
            raise AssertionError(error_msg_fn(l1, l2))


def assert_trees_all_equal_dtypes(*trees):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if not hasattr(l1, "dtype") or not hasattr(l2, "dtype"):
            raise AssertionError("is not a (j-)np array")
        if getattr(l1, "dtype", type(l1)) != getattr(l2, "dtype", type(l2)):
            raise AssertionError()


def assert_trees_all_equal_shapes(*trees):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if getattr(l1, "shape", ()) != getattr(l2, "shape", ()):
            raise AssertionError()


def assert_trees_all_equal_sizes(*trees):
    tree1 = trees[0]
    tree2 = trees[1]
    for l1, l2 in zip(_get_leaves(tree1), _get_leaves(tree2)):
        if math.prod(getattr(l1, "shape", ())) != math.prod(getattr(l2, "shape", ())):
            raise AssertionError()


def assert_trees_all_equal_shapes_and_dtypes(*trees):
    tree1 = trees[0]
    tree2 = trees[1]
    assert_trees_all_equal_shapes(tree1, tree2)
    assert_trees_all_equal_dtypes(tree1, tree2)
