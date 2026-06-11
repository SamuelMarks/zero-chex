"""Tree utility functions and definitions."""

from zero_jax.tree_util import (
    tree_all,
    tree_leaves,
    tree_map,
    tree_structure,
)

__all__ = [
    "tree_all",
    "tree_leaves",
    "tree_map",
    "tree_structure",
    "tree_flatten_with_path",
]


def tree_flatten_with_path(tree):
    """Flattens a tree into a list of (path, leaf) pairs and a tree structure.

    Args:
        tree: The nested tree structure to flatten.

    Returns:
        A tuple containing:
            - A list of tuples, where each tuple is (None, leaf_value). The path is currently always None.
            - None, representing the tree structure (currently simplified).
    """
    leaves = tree_leaves(tree)
    return [(None, leaf) for leaf in leaves], None
