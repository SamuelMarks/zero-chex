"""Simple tree utilities for zero_chex, bridging to zero_jax.tree_util."""

from typing import Any, Callable, List, Tuple

import zero_jax.tree_util as jtu


def tree_leaves(tree: Any) -> List[Any]:
    """Docstring."""
    return jtu.tree_flatten(tree)[0]


def tree_map(f: Callable, tree: Any) -> Any:
    """Docstring."""
    leaves, treedef = jtu.tree_flatten(tree)
    new_leaves = [f(leaf) for leaf in leaves]
    return jtu.tree_unflatten(treedef, new_leaves)


def tree_all(tree: Any) -> bool:
    """Docstring."""
    return all(tree_leaves(tree))


def tree_flatten_with_path(
    tree: Any, current_path: str = ""
) -> Tuple[List[Tuple[str, Any]], Any]:
    """Docstring."""
    leaves = []
    if isinstance(tree, dict):
        for k in sorted(tree.keys()):
            p = f"{current_path}['{k}']" if current_path else f"['{k}']"
            leaves.extend(tree_flatten_with_path(tree[k], p)[0])
    elif isinstance(tree, (list, tuple)):
        for i, item in enumerate(tree):
            p = f"{current_path}[{i}]" if current_path else f"[{i}]"
            leaves.extend(tree_flatten_with_path(item, p)[0])
    else:
        p = current_path if current_path else "()"
        leaves.append((p, tree))
    return leaves, None


def register_dataclass(*a: Any, **k: Any) -> None:
    """Docstring."""
    pass


def tree_structure(tree: Any, is_leaf: Callable = lambda x: False) -> Any:
    """Docstring."""
    # We still need a custom structural representation for tests since zero_jax
    # doesn't export exactly what chex tests expect yet for nested PyTreeDefs.
    if is_leaf(tree):
        return "*"  # pragma: no cover
    if isinstance(tree, dict):
        return {"dict": {k: tree_structure(v, is_leaf) for k, v in tree.items()}}
    elif isinstance(tree, list):
        return ["list", [tree_structure(v, is_leaf) for v in tree]]
    elif isinstance(tree, tuple):
        return ("tuple", [tree_structure(v, is_leaf) for v in tree])
    else:
        return "*"
