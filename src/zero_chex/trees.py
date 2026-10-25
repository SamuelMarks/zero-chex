"""PyTree structural and leaf value assertions for zero_chex."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence

from ml_switcheroo_ir import TensorSpec

from zero_chex._src.tree_util import (
    TreeMapCallable,
    tree_flatten_with_path,
    tree_leaves,
    tree_map,
    tree_structure,
)
from zero_chex.assertions import _dtypes_match, _get_shape, _get_size, _get_type
from zero_chex.mocks import are_asserts_enabled
from zero_chex.types import (
    ArrayProtocol,
    DictKey,
    DTypeAttrProtocol,
    LeafValue,
    PyTree,
    Scalar,
    ShapeOnlyProtocol,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    VariableLeafProtocol,
    _extract_flat_scalars,
    _has_valid_shape_tuple,
)


def _format_path(path: tuple[DictKey, ...]) -> str:
    """Format a PyTree key path tuple into a readable string representation.

    Args:
        path: Tuple of string, integer, or tuple keys from root to leaf.

    Returns:
        Formatted path string (e.g., "root['a'][0]").
    """
    if not path:
        return "root"
    return "root" + "".join(f"[{p!r}]" for p in path)


def _get_leaves(tree: PyTree) -> list[LeafValue]:
    """Retrieve all leaves from a PyTree in deterministic traversal order.

    Args:
        tree: The PyTree to extract leaves from.

    Returns:
        List of leaf values.
    """
    return tree_leaves(tree)


def _scalar_is_nan(val: Scalar) -> bool:
    """Check whether a scalar (real or complex) has any NaN component.

    Args:
        val: Scalar value to inspect.

    Returns:
        True if real or imaginary part is NaN.
    """
    if isinstance(val, complex):
        return math.isnan(val.real) or math.isnan(val.imag)
    return math.isnan(float(val))


def _scalar_is_inf(val: Scalar) -> bool:
    """Check whether a scalar (real or complex) has any infinite component.

    Args:
        val: Scalar value to inspect.

    Returns:
        True if real or imaginary part is infinite.
    """
    if isinstance(val, complex):
        return math.isinf(val.real) or math.isinf(val.imag)
    return math.isinf(float(val))


def _leaf_close(
    l1: LeafValue,
    l2: LeafValue,
    rtol: float = 1e-06,
    atol: float = 0.0,
    equal_nan: bool = False,
) -> bool:
    """Compare two leaf values for elementwise numerical closeness.

    Args:
        l1: First leaf value.
        l2: Second leaf value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether two NaN values at the same position are treated as equal.

    Returns:
        True if all corresponding elements are close within tolerance.
    """
    if _get_shape(l1) != _get_shape(l2):
        return False
    s1 = _extract_flat_scalars(l1)
    s2 = _extract_flat_scalars(l2)
    if not s1 and not s2:
        return bool(l1 == l2)
    for a, b in zip(s1, s2):
        if _scalar_is_nan(a) or _scalar_is_nan(b):
            if equal_nan and _scalar_is_nan(a) and _scalar_is_nan(b):
                continue
            return False
        if _scalar_is_inf(a) or _scalar_is_inf(b):
            if a == b:
                continue
            return False
        ca = complex(a) if isinstance(a, complex) else complex(float(a), 0.0)
        cb = complex(b) if isinstance(b, complex) else complex(float(b), 0.0)
        if abs(ca - cb) > atol + rtol * abs(cb):
            return False
    return True


def _float32_ulp_diff(a: float, b: float) -> int:
    """Compute the IEEE-754 single-precision (float32) ULP distance between two real floats.

    Args:
        a: First floating-point number.
        b: Second floating-point number.

    Returns:
        Non-negative integer ULP distance in 32-bit IEEE-754 representation.
    """
    if a == b:
        return 0
    try:
        bits_a = int(struct.unpack("!I", struct.pack("!f", a))[0])
        bits_b = int(struct.unpack("!I", struct.pack("!f", b))[0])
    except OverflowError:
        bits_a64 = int(struct.unpack("!Q", struct.pack("!d", a))[0])
        bits_b64 = int(struct.unpack("!Q", struct.pack("!d", b))[0])
        ord_a64 = -(bits_a64 & 0x7FFFFFFFFFFFFFFF) if (bits_a64 & 0x8000000000000000) else bits_a64
        ord_b64 = -(bits_b64 & 0x7FFFFFFFFFFFFFFF) if (bits_b64 & 0x8000000000000000) else bits_b64
        return abs(ord_a64 - ord_b64)

    ord_a = -(bits_a & 0x7FFFFFFF) if (bits_a & 0x80000000) else bits_a
    ord_b = -(bits_b & 0x7FFFFFFF) if (bits_b & 0x80000000) else bits_b
    return abs(ord_a - ord_b)


def _leaf_close_ulp(l1: LeafValue, l2: LeafValue, maxulp: int = 1) -> bool:
    """Compare two leaf values within maxulp IEEE-754 Units in the Last Place.

    Args:
        l1: First leaf value.
        l2: Second leaf value.
        maxulp: Maximum allowed ULP difference.

    Returns:
        True if all corresponding elements are within maxulp ULPs.
    """
    if _get_shape(l1) != _get_shape(l2):
        return False
    s1 = _extract_flat_scalars(l1)
    s2 = _extract_flat_scalars(l2)
    if not s1 and not s2:
        return bool(l1 == l2)
    for a, b in zip(s1, s2):
        if _scalar_is_nan(a) or _scalar_is_nan(b):
            return False
        if _scalar_is_inf(a) or _scalar_is_inf(b):
            if a == b:
                continue
            return False
        if isinstance(a, complex) or isinstance(b, complex):
            ca = complex(a)
            cb = complex(b)
            if _float32_ulp_diff(ca.real, cb.real) > maxulp or _float32_ulp_diff(ca.imag, cb.imag) > maxulp:
                return False
        else:
            if _float32_ulp_diff(float(a), float(b)) > maxulp:
                return False
    return True


def _leaf_equal(l1: LeafValue, l2: LeafValue, strict: bool = False) -> bool:
    """Check whether two PyTree leaves are equal in shape, value, and optionally dtype.

    Args:
        l1: First leaf value.
        l2: Second leaf value.
        strict: If True, also requires matching dtypes.

    Returns:
        True if leaves are equal.
    """
    if _get_shape(l1) != _get_shape(l2):
        return False
    if strict:
        dt1 = _get_type(l1)
        dt2 = _get_type(l2)
        if not _dtypes_match(dt1, dt2):
            return False
    s1 = _extract_flat_scalars(l1)
    s2 = _extract_flat_scalars(l2)
    if s1 or s2:
        for a, b in zip(s1, s2):
            if isinstance(a, bool) != isinstance(b, bool):
                return False
            if a != b:
                return False
        return True
    return bool(l1 == l2)


def assert_tree_all_finite(tree_like: PyTree) -> None:
    """Assert that all numeric and array leaves in tree_like contain only finite values.

    Args:
        tree_like: The PyTree to inspect for NaN or Inf values.

    Raises:
        AssertionError: If any leaf contains a non-finite value.
    """
    if not are_asserts_enabled():
        return
    path_leaves, _ = tree_flatten_with_path(tree_like)
    for path, leaf in path_leaves:
        scalars = _extract_flat_scalars(leaf)
        for val in scalars:
            if isinstance(val, complex):
                if not (math.isfinite(val.real) and math.isfinite(val.imag)):
                    raise AssertionError(f"Tree contains non-finite value {val!r} at {_format_path(path)}.")
            else:
                if not math.isfinite(float(val)):
                    raise AssertionError(f"Tree contains non-finite value {val!r} at {_format_path(path)}.")


def assert_tree_has_only_ndarrays(tree: PyTree) -> None:
    """Assert that every leaf in tree is an n-dimensional array object.

    Args:
        tree: The PyTree whose leaves are checked.

    Raises:
        AssertionError: If any leaf does not expose a valid shape attribute.
    """
    if not are_asserts_enabled():
        return
    path_leaves, _ = tree_flatten_with_path(tree)
    for path, leaf in path_leaves:
        inner = leaf.value if isinstance(leaf, VariableLeafProtocol) else leaf
        is_ndarray = isinstance(
            inner,
            (
                ArrayProtocol,
                SharedMemoryArray,
                SharedMemoryArrayMetadata,
                TensorSpec,
            ),
        ) or (isinstance(inner, ShapeOnlyProtocol) and _has_valid_shape_tuple(inner))
        if not is_ndarray:
            raise AssertionError(
                f"Tree leaf {leaf!r} at {_format_path(path)} of type {type(leaf).__name__} is not an ndarray."
            )


def assert_tree_no_nones(tree: PyTree) -> None:
    """Assert that no leaf in tree is None.

    Args:
        tree: The PyTree to inspect.

    Raises:
        AssertionError: If any leaf in tree is None.
    """
    if not are_asserts_enabled():
        return
    if tree is None:
        raise AssertionError("Tree is None.")
    path_leaves, _ = tree_flatten_with_path(tree)
    for path, leaf in path_leaves:
        if leaf is None:
            raise AssertionError(f"None detected in tree at {_format_path(path)}.")


def assert_tree_shape_prefix(tree: PyTree, shape_prefix: Sequence[int]) -> None:
    """Assert that all leaves in tree have shapes starting with shape_prefix.

    Args:
        tree: The PyTree whose leaf shapes are checked.
        shape_prefix: Sequence of leading dimension sizes expected on every leaf.

    Raises:
        AssertionError: If any leaf's shape does not begin with shape_prefix.
    """
    if not are_asserts_enabled():
        return
    prefix = tuple(shape_prefix)
    path_leaves, _ = tree_flatten_with_path(tree)
    for path, leaf in path_leaves:
        s = _get_shape(leaf)
        if len(s) < len(prefix) or s[: len(prefix)] != prefix:
            raise AssertionError(
                f"Leaf shape {s} at {_format_path(path)} does not start with expected prefix {prefix}."
            )


def assert_tree_shape_suffix(tree: PyTree, shape_suffix: Sequence[int]) -> None:
    """Assert that all leaves in tree have shapes ending with shape_suffix.

    Args:
        tree: The PyTree whose leaf shapes are checked.
        shape_suffix: Sequence of trailing dimension sizes expected on every leaf.

    Raises:
        AssertionError: If any leaf's shape does not end with shape_suffix.
    """
    if not are_asserts_enabled():
        return
    suffix = tuple(shape_suffix)
    path_leaves, _ = tree_flatten_with_path(tree)
    for path, leaf in path_leaves:
        s = _get_shape(leaf)
        if not suffix:
            continue
        if len(s) < len(suffix) or s[-len(suffix) :] != suffix:
            raise AssertionError(f"Leaf shape {s} at {_format_path(path)} does not end with expected suffix {suffix}.")


def assert_trees_all_equal_structs(*trees: PyTree) -> None:
    """Assert that all provided PyTrees share the exact same container structure.

    Args:
        *trees: Two or more PyTrees to compare structurally.

    Raises:
        AssertionError: If the PyTreeDef structures of any two trees differ.
    """
    if not are_asserts_enabled():
        return
    if len(trees) < 2:
        return
    first_tree = trees[0]
    def0 = tree_structure(first_tree)
    for t in trees[1:]:
        if tree_structure(t) != def0:
            raise AssertionError(f"Trees have different structures: {def0!r} vs {tree_structure(t)!r}.")


def assert_trees_all_equal_comparator(
    equality_comparator: Callable[[LeafValue, LeafValue], bool],
    error_msg_fn: Callable[[LeafValue, LeafValue], str],
    *trees: PyTree,
) -> None:
    """Assert that all PyTrees have identical structure and satisfy equality_comparator on leaves.

    Args:
        equality_comparator: Binary predicate comparing corresponding leaves.
        error_msg_fn: Function producing an error string for a failing leaf pair.
        *trees: Two or more PyTrees to compare.

    Raises:
        AssertionError: If structures differ or equality_comparator returns False for any leaf pair.
    """
    if not are_asserts_enabled():
        return
    if len(trees) < 2:
        return
    assert_trees_all_equal_structs(*trees)
    path_leaves0, _ = tree_flatten_with_path(trees[0])
    for other_tree in trees[1:]:
        other_leaves = _get_leaves(other_tree)
        for (path, l1), l2 in zip(path_leaves0, other_leaves):
            if not equality_comparator(l1, l2):
                raise AssertionError(f"Error at {_format_path(path)}: {error_msg_fn(l1, l2)}")


def assert_trees_all_close(
    *trees: PyTree,
    rtol: float = 1e-06,
    atol: float = 0.0,
    equal_nan: bool = False,
) -> None:
    """Assert that all PyTrees have identical structure and numerically close leaf values.

    Args:
        *trees: Two or more PyTrees to compare.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: Whether NaNs in corresponding positions are considered equal.

    Raises:
        AssertionError: If structures differ or any leaves are not within tolerance.
    """
    if not are_asserts_enabled():
        return
    assert_trees_all_equal_comparator(
        lambda l1, l2: _leaf_close(l1, l2, rtol=rtol, atol=atol, equal_nan=equal_nan),
        lambda l1, l2: f"Leaves not close: {l1!r} vs {l2!r} (rtol={rtol}, atol={atol})",
        *trees,
    )


def assert_trees_all_close_ulp(
    *trees: PyTree,
    maxulp: int = 1,
) -> None:
    """Assert that all PyTrees have identical structure and leaves within maxulp ULPs.

    Args:
        *trees: Two or more PyTrees to compare.
        maxulp: Maximum number of ULPs allowed between corresponding leaf elements.

    Raises:
        AssertionError: If structures differ or leaves differ by more than maxulp ULPs.
    """
    if not are_asserts_enabled():
        return
    assert_trees_all_equal_comparator(
        lambda l1, l2: _leaf_close_ulp(l1, l2, maxulp=maxulp),
        lambda l1, l2: f"Leaves differ by more than {maxulp} ULPs: {l1!r} vs {l2!r}",
        *trees,
    )


def assert_trees_all_equal(
    *trees: PyTree,
    strict: bool = False,
) -> None:
    """Assert that all PyTrees have identical structure and equal leaf values.

    Args:
        *trees: Two or more PyTrees to compare.
        strict: Whether to also enforce strict dtype equality on leaves.

    Raises:
        AssertionError: If structures or leaf values differ.
    """
    if not are_asserts_enabled():
        return
    assert_trees_all_equal_comparator(
        lambda l1, l2: _leaf_equal(l1, l2, strict=strict),
        lambda l1, l2: f"Leaves are not equal: {l1!r} != {l2!r}",
        *trees,
    )


def assert_trees_all_equal_dtypes(*trees: PyTree) -> None:
    """Assert that all PyTrees have identical structure and matching array dtypes on leaves.

    Args:
        *trees: Two or more PyTrees to compare.

    Raises:
        AssertionError: If any leaf lacks a dtype attribute or leaf dtypes differ.
    """
    if not are_asserts_enabled():
        return
    if len(trees) < 2:
        return
    assert_trees_all_equal_structs(*trees)
    path_leaves0, _ = tree_flatten_with_path(trees[0])
    for other_tree in trees[1:]:
        other_leaves = _get_leaves(other_tree)
        for (path, l1), l2 in zip(path_leaves0, other_leaves):
            inner1 = l1.value if isinstance(l1, VariableLeafProtocol) else l1
            inner2 = l2.value if isinstance(l2, VariableLeafProtocol) else l2
            if not isinstance(inner1, (ArrayProtocol, DTypeAttrProtocol, TensorSpec)) or not isinstance(
                inner2, (ArrayProtocol, DTypeAttrProtocol, TensorSpec)
            ):
                raise AssertionError(
                    f"Leaf {l1!r} or {l2!r} at {_format_path(path)} is not a (j-)np array exposing dtype."
                )
            if not _dtypes_match(inner1.dtype, inner2.dtype):
                raise AssertionError(
                    f"Leaf dtypes do not match at {_format_path(path)}: {inner1.dtype!r} != {inner2.dtype!r}."
                )


def assert_trees_all_equal_shapes(*trees: PyTree) -> None:
    """Assert that all PyTrees have identical structure and matching shapes on leaves.

    Args:
        *trees: Two or more PyTrees to compare.

    Raises:
        AssertionError: If any corresponding leaves have different shapes.
    """
    if not are_asserts_enabled():
        return
    assert_trees_all_equal_comparator(
        lambda l1, l2: _get_shape(l1) == _get_shape(l2),
        lambda l1, l2: f"Leaf shapes do not match: {_get_shape(l1)} != {_get_shape(l2)}",
        *trees,
    )


def assert_trees_all_equal_sizes(*trees: PyTree) -> None:
    """Assert that all PyTrees have identical structure and matching element counts on leaves.

    Args:
        *trees: Two or more PyTrees to compare.

    Raises:
        AssertionError: If any corresponding leaves have different total element counts.
    """
    if not are_asserts_enabled():
        return
    assert_trees_all_equal_comparator(
        lambda l1, l2: _get_size(l1) == _get_size(l2),
        lambda l1, l2: f"Leaf sizes do not match: {_get_size(l1)} != {_get_size(l2)}",
        *trees,
    )


def assert_trees_all_equal_shapes_and_dtypes(*trees: PyTree) -> None:
    """Assert that all PyTrees have identical structure, leaf shapes, and leaf dtypes.

    Args:
        *trees: Two or more PyTrees to compare.

    Raises:
        AssertionError: If structures, leaf shapes, or leaf dtypes differ.
    """
    if not are_asserts_enabled():
        return
    assert_trees_all_equal_shapes(*trees)
    assert_trees_all_equal_dtypes(*trees)


__all__ = [
    "TreeMapCallable",
    "_extract_flat_scalars",
    "assert_tree_all_finite",
    "assert_tree_has_only_ndarrays",
    "assert_tree_no_nones",
    "assert_tree_shape_prefix",
    "assert_tree_shape_suffix",
    "assert_trees_all_close",
    "assert_trees_all_close_ulp",
    "assert_trees_all_equal",
    "assert_trees_all_equal_comparator",
    "assert_trees_all_equal_dtypes",
    "assert_trees_all_equal_shapes",
    "assert_trees_all_equal_shapes_and_dtypes",
    "assert_trees_all_equal_sizes",
    "assert_trees_all_equal_structs",
    "tree_map",
]
