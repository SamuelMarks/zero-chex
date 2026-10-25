"""Pure-Python PyTree traversal, flattening, and custom node registration utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Protocol, TypeVar

from ml_switcheroo_ir import LogicalNode, SymInt, SymNode, TensorSpec
from typing_extensions import TypeGuard

from zero_chex.types import (
    ArrayProtocol,
    DeviceProtocol,
    DevicesMethodProtocol,
    DictKey,
    DTypeAttrProtocol,
    LeafValue,
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    OptStateLeafProtocol,
    PyTree,
    PyTreeDef,
    ShapeOnlyProtocol,
    ShardingAttrProtocol,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    StateMappingProtocol,
    VariableLeafProtocol,
    _extract_flat_scalars,
)

FlattenFn = Callable[[MappableDataclassProtocol], tuple[tuple[PyTree, ...], tuple[str, ...]]]
UnflattenFn = Callable[[tuple[str, ...], tuple[PyTree, ...]], PyTree]


class TreeMapCallable(Protocol):
    """Protocol for callables passed to tree_map."""

    def __call__(self, *leaves: LeafValue) -> LeafValue:
        """Transform one or more corresponding PyTree leaf values.

        Args:
            *leaves: Corresponding leaf values across input PyTrees.

        Returns:
            Transformed leaf value.
        """
        ...


TCls = TypeVar("TCls")
_CUSTOM_PYTREE_NODES: dict[type[MappableDataclassProtocol], tuple[FlattenFn, UnflattenFn]] = {}


def register_custom_pytree_node(
    cls: type[MappableDataclassProtocol],
    flatten_fn: FlattenFn,
    unflatten_fn: UnflattenFn,
) -> None:
    """Register a custom dataclass type for PyTree traversal.

    Args:
        cls: The class type to register as an internal PyTree container node.
        flatten_fn: Callable extracting child values and field names from an instance.
        unflatten_fn: Callable reconstructing an instance from field names and child values.
    """
    _CUSTOM_PYTREE_NODES[cls] = (flatten_fn, unflatten_fn)


def _lookup_custom_pytree_node(
    cls: type[MappableDataclassProtocol],
) -> tuple[FlattenFn, UnflattenFn] | None:
    """Look up registered custom PyTree flatten and unflatten functions by class reference.

    Args:
        cls: The class type to look up.

    Returns:
        Tuple of (flatten_fn, unflatten_fn) if registered, or None.
    """
    return _CUSTOM_PYTREE_NODES.get(cls)


def _is_mappable_dataclass_cls(
    cls: type[MappableDataclassProtocol] | type[NamedTuplePyTreeProtocol] | type[StateMappingProtocol],
) -> TypeGuard[type[MappableDataclassProtocol]]:
    """Check whether a custom container class is a MappableDataclassProtocol subclass.

    Args:
        cls: Custom container class type.

    Returns:
        True if cls is a MappableDataclassProtocol type.
    """
    return issubclass(cls, MappableDataclassProtocol)


def _is_namedtuple_cls(
    cls: type[MappableDataclassProtocol] | type[NamedTuplePyTreeProtocol] | type[StateMappingProtocol],
) -> TypeGuard[type[NamedTuplePyTreeProtocol]]:
    """Check whether a custom container class is a NamedTuplePyTreeProtocol subclass.

    Args:
        cls: Custom container class type.

    Returns:
        True if cls is a tuple/NamedTuplePyTreeProtocol type.
    """
    return issubclass(cls, tuple)


def _is_state_mapping_cls(
    cls: type[MappableDataclassProtocol] | type[NamedTuplePyTreeProtocol] | type[StateMappingProtocol],
) -> TypeGuard[type[StateMappingProtocol]]:
    """Check whether a custom container class is a StateMappingProtocol subclass.

    Args:
        cls: Custom container class type.

    Returns:
        True if cls is a StateMappingProtocol type.
    """
    return issubclass(cls, StateMappingProtocol)


def _is_leaf_value(node: PyTree) -> TypeGuard[LeafValue]:
    """Determine whether a node is a terminal PyTree leaf.

    Args:
        node: The PyTree element to inspect.

    Returns:
        True if node is a leaf value rather than a traversable container.
    """
    if isinstance(node, MappableDataclassProtocol):
        return False
    if isinstance(
        node,
        (
            ArrayProtocol,
            VariableLeafProtocol,
            SharedMemoryArray,
            SharedMemoryArrayMetadata,
            TensorSpec,
            LogicalNode,
        ),
    ):
        return True
    if isinstance(node, (list, tuple, Mapping, NamedTuplePyTreeProtocol, StateMappingProtocol)):
        return False
    return isinstance(
        node,
        (
            int,
            float,
            complex,
            bool,
            str,
            type(None),
            SymInt,
            SymNode,
            ShapeOnlyProtocol,
            DeviceProtocol,
            DevicesMethodProtocol,
            ShardingAttrProtocol,
            DTypeAttrProtocol,
            OptStateLeafProtocol,
        ),
    )


def _normalize_dict_key(k: DictKey) -> DictKey:
    """Validate and return a dictionary key for PyTreeDef storage.

    Args:
        k: Dictionary key from a mapping node.

    Returns:
        The preserved dictionary key.
    """
    return k


def tree_flatten_with_path(
    tree: PyTree,
) -> tuple[list[tuple[tuple[DictKey, ...], LeafValue]], PyTreeDef]:
    """Flatten a PyTree into a list of (path, leaf) pairs and a PyTreeDef structure.

    Args:
        tree: The nested PyTree structure to flatten.

    Returns:
        A tuple containing:
            - A list of (key_path_tuple, leaf_value) pairs.
            - The PyTreeDef structural descriptor.
    """
    leaves_with_paths: list[tuple[tuple[DictKey, ...], LeafValue]] = []

    def _visit(node: PyTree, path: tuple[DictKey, ...]) -> PyTreeDef:
        """Recursively visit a PyTree node and collect path-annotated leaves.

        Args:
            node: Current subtree or leaf node.
            path: Accumulated key/index path from the root.

        Returns:
            PyTreeDef describing the visited subtree.
        """
        if isinstance(node, MappableDataclassProtocol):
            node_cls = type(node)
            custom_entry = _lookup_custom_pytree_node(node_cls)
            if custom_entry is not None:
                flatten_fn, _ = custom_entry
                children_vals, field_names = flatten_fn(node)
            else:
                field_names = node.__chex_mappable_fields__()
                children_vals = node.__chex_field_values__()
            child_defs = tuple(
                _visit(child_val, path + (field_name,)) for field_name, child_val in zip(field_names, children_vals)
            )
            return PyTreeDef(
                node_type="custom",
                keys=field_names,
                children=child_defs,
                custom_cls=node_cls,
            )

        if not _is_leaf_value(node):
            if isinstance(node, NamedTuplePyTreeProtocol):
                nt_cls = type(node)
                nt_fields = tuple(node._fields)
                nt_values = tuple(node)
                child_defs = tuple(
                    _visit(child_val, path + (field_name,)) for field_name, child_val in zip(nt_fields, nt_values)
                )
                return PyTreeDef(
                    node_type="namedtuple",
                    keys=nt_fields,
                    children=child_defs,
                    custom_cls=nt_cls,
                )
            if isinstance(node, StateMappingProtocol) and type(node).__name__ != "dict":
                map_cls = type(node)
                ordered_keys: tuple[DictKey, ...] = tuple(_normalize_dict_key(k) for k in node.keys())
                child_defs = tuple(_visit(node[k], path + (k,)) for k in ordered_keys)
                return PyTreeDef(
                    node_type="state_mapping",
                    keys=ordered_keys,
                    children=child_defs,
                    custom_cls=map_cls,
                )
            if isinstance(node, Mapping):
                sorted_keys: tuple[DictKey, ...] = tuple(
                    _normalize_dict_key(k) for k in sorted(node.keys(), key=lambda item: (str(type(item)), str(item)))
                )
                child_defs = tuple(_visit(node[k], path + (k,)) for k in sorted_keys)
                return PyTreeDef(node_type="dict", keys=sorted_keys, children=child_defs)
            if isinstance(node, list):
                child_defs = tuple(_visit(item, path + (idx,)) for idx, item in enumerate(node))
                return PyTreeDef(
                    node_type="list",
                    keys=tuple(range(len(node))),
                    children=child_defs,
                )
            if isinstance(node, tuple):
                child_defs = tuple(_visit(item, path + (idx,)) for idx, item in enumerate(node))
                return PyTreeDef(
                    node_type="tuple",
                    keys=tuple(range(len(node))),
                    children=child_defs,
                )

        if _is_leaf_value(node):
            leaves_with_paths.append((path, node))
        return PyTreeDef(node_type="leaf")

    treedef = _visit(tree, ())
    return leaves_with_paths, treedef


def tree_flatten(tree: PyTree) -> tuple[list[LeafValue], PyTreeDef]:
    """Flatten a PyTree into a list of leaves and a PyTreeDef structure.

    Args:
        tree: The nested PyTree structure to flatten.

    Returns:
        A tuple of (leaves_list, PyTreeDef).
    """
    path_leaves, treedef = tree_flatten_with_path(tree)
    return [leaf for _, leaf in path_leaves], treedef


def tree_leaves(tree: PyTree) -> list[LeafValue]:
    """Extract all leaves from a PyTree in deterministic traversal order.

    Args:
        tree: The PyTree structure to extract leaves from.

    Returns:
        A list of leaf values.
    """
    leaves, _ = tree_flatten(tree)
    return leaves


def tree_structure(tree: PyTree) -> PyTreeDef:
    """Compute the structural PyTreeDef descriptor for a PyTree.

    Args:
        tree: The PyTree structure to inspect.

    Returns:
        The PyTreeDef representing the container hierarchy of tree.
    """
    _, treedef = tree_flatten(tree)
    return treedef


def tree_unflatten(treedef: PyTreeDef, leaves: Iterable[LeafValue]) -> PyTree:
    """Reconstruct a PyTree from a PyTreeDef and an iterable of leaves.

    Args:
        treedef: Structural PyTreeDef descriptor.
        leaves: Sequence or iterable of leaf values.

    Returns:
        Reconstructed PyTree matching treedef.

    Raises:
        ValueError: If leaf count does not match treedef.num_leaves or node_type is unknown.
    """
    leaf_list = list(leaves)
    if len(leaf_list) != treedef.num_leaves:
        raise ValueError(
            f"Too {'few' if len(leaf_list) < treedef.num_leaves else 'many'} leaves for PyTreeDef: "
            f"expected {treedef.num_leaves}, got {len(leaf_list)}."
        )
    leaf_iter: Iterator[LeafValue] = iter(leaf_list)

    def _build(current_def: PyTreeDef) -> PyTree:
        """Recursively reconstruct a subtree from a PyTreeDef node.

        Args:
            current_def: Current PyTreeDef node.

        Returns:
            Reconstructed subtree or leaf value.

        Raises:
            ValueError: If current_def.node_type is unrecognized.
        """
        if current_def.node_type == "leaf":
            return next(leaf_iter)
        built_children = tuple(_build(child) for child in current_def.children)
        if current_def.node_type == "list":
            return list(built_children)
        if current_def.node_type == "tuple":
            return built_children
        if current_def.node_type == "dict":
            return dict(zip(current_def.keys, built_children))
        if current_def.node_type in ("namedtuple", "state_mapping", "custom") and current_def.custom_cls is not None:
            custom_cls = current_def.custom_cls
            if _is_namedtuple_cls(custom_cls):
                return custom_cls(*built_children)
            if _is_mappable_dataclass_cls(custom_cls):
                custom_entry = _lookup_custom_pytree_node(custom_cls)
                if custom_entry is not None:
                    _, unflatten_fn = custom_entry
                    str_keys = tuple(str(k) for k in current_def.keys)
                    return unflatten_fn(str_keys, built_children)
                kwargs = {str(k): child_val for k, child_val in zip(current_def.keys, built_children)}
                return custom_cls(**kwargs)
            assert _is_state_mapping_cls(custom_cls)
            return custom_cls({str(k): child_val for k, child_val in zip(current_def.keys, built_children)})
        raise ValueError(f"Unknown PyTreeDef node_type: {current_def.node_type!r}")

    return _build(treedef)


def tree_map(
    fn: TreeMapCallable,
    tree: PyTree,
    *rest: PyTree,
) -> PyTree:
    """Map a function over all corresponding leaves of one or more PyTrees.

    Args:
        fn: Callable applied to each leaf (or tuple of leaves across trees).
        tree: Primary PyTree determining structure.
        *rest: Additional PyTrees with identical structure.

    Returns:
        A new PyTree with the same structure as tree and transformed leaves.

    Raises:
        ValueError: If any secondary PyTree in rest has a different structure than tree.
    """
    leaves0, treedef = tree_flatten(tree)
    if not rest:
        mapped_leaves = [fn(leaf) for leaf in leaves0]
    else:
        rest_leaves: list[list[LeafValue]] = []
        for other in rest:
            other_l, other_def = tree_flatten(other)
            if other_def != treedef:
                raise ValueError(f"PyTree structure mismatch in tree_map: expected {treedef!r}, got {other_def!r}.")
            rest_leaves.append(other_l)
        mapped_leaves = [fn(*leaf_group) for leaf_group in zip(leaves0, *rest_leaves)]
    return tree_unflatten(treedef, mapped_leaves)


def tree_all(tree: PyTree) -> bool:
    """Return True if all leaves in tree evaluate to truthy booleans.

    Args:
        tree: The PyTree whose leaves are tested for truthiness.

    Returns:
        True if every leaf is truthy, False otherwise.
    """
    for leaf in tree_leaves(tree):
        inner = leaf.value if isinstance(leaf, VariableLeafProtocol) else leaf
        if isinstance(inner, ArrayProtocol):
            if not all(bool(x) for x in _extract_flat_scalars(inner)):
                return False
        elif not bool(inner):
            return False
    return True


__all__ = [
    "FlattenFn",
    "TreeMapCallable",
    "UnflattenFn",
    "register_custom_pytree_node",
    "tree_all",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_leaves",
    "tree_map",
    "tree_structure",
    "tree_unflatten",
]
