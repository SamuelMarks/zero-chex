"""Module docstring."""

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
    """Docstring."""
    leaves = tree_leaves(tree)
    return [(None, leaf) for leaf in leaves], None
