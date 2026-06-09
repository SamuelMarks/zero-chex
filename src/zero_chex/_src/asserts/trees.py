# ruff: noqa: F821, F403, F405, E402, E731, F841, E721

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy
    import jax
    from typing import *

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        user = None


"""Tree assertions."""

import collections.abc
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

import zero_jax as jax
from zero_chex._src import tree_util
import numpy as np

ArrayTree = Union[
    jax.Array,
    np.ndarray,
    np.bool_,
    np.number,
    Iterable["ArrayTree"],
    Mapping[Any, "ArrayTree"],
]


def assert_tree_all_finite(
    tree_like: Union[
        jax.Array,
        np.ndarray,
        np.bool_,
        np.number,
        Iterable["ArrayTree"],
        Mapping[Any, "ArrayTree"],
    ],
) -> None:
    """Checks that all leaves in a tree are finite."""
    all_finite = tree_util.tree_all(
        tree_util.tree_map(lambda x: np.all(jax.numpy.isfinite(x)), tree_like)
    )
    if not all_finite:
        is_finite = lambda x: "Finite" if np.all(jax.numpy.isfinite(x)) else "Nonfinite"
        error_msg = tree_util.tree_map(is_finite, tree_like)
        raise AssertionError(f"Tree contains non-finite value: {error_msg}.")


def assert_tree_has_only_ndarrays(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that all `tree`'s leaves are n-dimensional arrays (tensors)."""
    errors = []

    def _assert_fn(path: "str", leaf: "Any") -> "NoneType":
        if leaf is not None:
            if not isinstance(leaf, (np.ndarray, jax.Array)):
                errors.append(
                    f"Tree leaf '{path}' is not an ndarray (type={type(leaf)})."
                )

    for path, leaf in tree_util.tree_flatten_with_path(tree)[0]:
        _assert_fn(str(path), leaf)
    if errors:
        raise AssertionError("\n".join(errors))


def _check_sharding(leaf: jax.Array) -> bool:
    try:
        sharding = getattr(leaf, "sharding", None)
        if sharding is not None:
            if type(sharding).__name__ == "PmapSharding":
                return True
            return len(sharding.device_set) > 1
        return len(getattr(leaf, "devices", lambda: [])()) > 1
    except Exception:
        return False


def assert_tree_is_on_device(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
    platform: "Sequence[str] | str" = ("gpu", "tpu"),
    device: "jaxlib.xla_extension.Device | None" = None,
) -> "NoneType":
    """Checks that all leaves are ndarrays residing in device memory (in HBM)."""
    assert_tree_has_only_ndarrays(tree)

    if device is not None:
        platform = (device.platform,)
    elif not isinstance(platform, collections.abc.Sequence) or isinstance(
        platform, str
    ):
        platform = (platform,)

    errors = []

    def _assert_fn(path: "str", leaf: "Any") -> "NoneType":
        if leaf is not None:
            if isinstance(leaf, jax.Array):  # pragma: no cover
                if _check_sharding(leaf) and not hasattr(leaf, "addressable_shards"):
                    errors.append(
                        f"Tree leaf '{path}' is a ShardedDeviceArray which are disallowed. "
                        f" (type={type(leaf)})."
                    )
                else:
                    leaf_device = list(getattr(leaf, "devices", lambda: [])())[0]
                    if leaf_device.platform not in platform:
                        errors.append(
                            f"Tree leaf '{path}' resides on '{leaf_device.platform}', "
                            f"expected '{platform}'."
                        )

                    if device is not None and getattr(
                        leaf, "devices", lambda: []
                    )() != {device}:
                        errors.append(
                            f"Tree leaf '{path}' resides on {getattr(leaf, 'devices', lambda: [])()}, expected {device}."
                        )
            else:  # pragma: no cover
                errors.append(
                    f"Tree leaf '{path}' has unexpected type: {type(leaf)}."
                )  # pragma: no cover

    for path, leaf in tree_util.tree_flatten_with_path(tree)[0]:
        _assert_fn(str(path), leaf)
    if errors:
        raise AssertionError("\n".join(errors))


def assert_tree_is_on_host(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
    allow_cpu_device: "bool" = True,
    allow_sharded_arrays: "bool" = False,
) -> "NoneType":
    """Checks that all leaves are ndarrays residing in the host memory (on CPU)."""
    assert_tree_has_only_ndarrays(tree)
    errors = []

    def _assert_fn(path: "str", leaf: "Any") -> "NoneType":
        if leaf is not None:
            if not isinstance(leaf, np.ndarray):
                if isinstance(leaf, jax.Array):  # pragma: no cover
                    if _check_sharding(leaf) and hasattr(leaf, "addressable_shards"):
                        if not allow_sharded_arrays:
                            errors.append(
                                f"Tree leaf '{path}' is sharded and resides on {getattr(leaf, 'devices', lambda: [])()} "
                                "(sharded arrays are disallowed)."
                            )
                        elif allow_cpu_device:
                            if any(
                                d.platform != "cpu"
                                for d in getattr(leaf, "devices", lambda: [])()
                            ):
                                errors.append(
                                    f"Tree leaf '{path}' is sharded and resides on {getattr(leaf, 'devices', lambda: [])()}."
                                )
                        else:
                            errors.append(
                                f"Tree leaf '{path}' is sharded and resides on {getattr(leaf, 'devices', lambda: [])()} "
                                "(CPU devices are disallowed)."
                            )
                    elif allow_cpu_device:
                        leaf_device = list(getattr(leaf, "devices", lambda: [])())[0]
                        if leaf_device.platform != "cpu":
                            errors.append(
                                f"Tree leaf '{path}' resides on {leaf_device}."
                            )
                    else:
                        errors.append(
                            f"Tree leaf '{path}' resides on {getattr(leaf, 'devices', lambda: [])()} "
                            "(CPU devices are disallowed)."
                        )
                else:
                    errors.append(
                        f"Tree leaf '{path}' has unexpected type: {type(leaf)}."
                    )  # pragma: no cover

    for path, leaf in tree_util.tree_flatten_with_path(tree)[0]:
        _assert_fn(str(path), leaf)
    if errors:
        raise AssertionError("\n".join(errors))  # pragma: no cover


def assert_tree_is_sharded(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
    devices: "Sequence[jaxlib.xla_extension.Device]",
) -> "NoneType":
    """Checks that all leaves are ndarrays sharded across the specified devices."""
    assert_tree_has_only_ndarrays(tree)

    errors = []
    devices_tuple = tuple(devices)

    def _assert_fn(path: "str", leaf: "Any") -> "NoneType":
        if leaf is not None:
            if isinstance(leaf, jax.Array):  # pragma: no cover
                if hasattr(leaf, "addressable_shards"):
                    shards = tuple(shard.device for shard in leaf.addressable_shards)
                    if shards != devices_tuple:
                        errors.append(
                            f"Tree leaf '{path}' is sharded across {shards} devices, "
                            f"expected {devices_tuple}."
                        )
                else:
                    errors.append(
                        f"Tree leaf '{path}' is not sharded (devices={getattr(leaf, 'devices', lambda: [])()})."
                    )
            else:
                errors.append(
                    f"Tree leaf '{path}' is not a jax.Array (type={type(leaf)})."
                )  # pragma: no cover

    for path, leaf in tree_util.tree_flatten_with_path(tree)[0]:
        _assert_fn(str(path), leaf)
    if errors:
        raise AssertionError("\n".join(errors))


def assert_tree_no_nones(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that a tree does not contain `None`."""
    has_nones = False

    def _is_leaf(value: Any) -> bool:
        if value is None:
            nonlocal has_nones
            has_nones = True
        return False

    treedef = tree_util.tree_structure(tree, is_leaf=_is_leaf)
    if has_nones:
        raise AssertionError(f"Tree contains `None`(s): {treedef}.")


def assert_tree_shape_prefix(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
    shape_prefix: "Sequence[int]",
) -> "NoneType":
    """Checks that all ``tree`` leaves' shapes have the same prefix."""
    shape_prefix_tuple = tuple(shape_prefix)

    if not shape_prefix_tuple:
        return

    errors = []

    def _assert_fn(path: "str", leaf: "Any") -> "NoneType":
        if len(shape_prefix_tuple) > len(leaf.shape):
            errors.append(
                f"Tree leaf '{path}' has a shape of length {leaf.ndim} (shape={leaf.shape}) "
                f"which is smaller than the expected prefix of length {len(shape_prefix_tuple)} "
                f"(prefix={shape_prefix_tuple})."
            )
            return

        suffix = leaf.shape[: len(shape_prefix_tuple)]
        if suffix != shape_prefix_tuple:
            errors.append(
                f"Tree leaf '{path}' has a shape prefix different from expected: "
                f"{suffix} != {shape_prefix_tuple}."
            )

    for path, leaf in tree_util.tree_flatten_with_path(tree)[0]:
        _assert_fn(str(path), leaf)
    if errors:
        raise AssertionError("\n".join(errors))


def assert_tree_shape_suffix(
    tree: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
    shape_suffix: "Sequence[int]",
) -> "NoneType":
    """Checks that all ``tree`` leaves' shapes have the same suffix."""
    shape_suffix_tuple = tuple(shape_suffix)

    if not shape_suffix_tuple:
        return

    errors = []

    def _assert_fn(path: "str", leaf: "Any") -> "NoneType":
        if len(shape_suffix_tuple) > len(leaf.shape):
            errors.append(
                f"Tree leaf '{path}' has a shape of length {len(leaf.shape)} "
                f"(shape={leaf.shape}) which is smaller than the expected suffix "
                f"of length {len(shape_suffix_tuple)} (suffix={shape_suffix_tuple})."
            )
            return

        suffix = leaf.shape[-len(shape_suffix_tuple) :]
        if suffix != shape_suffix_tuple:
            errors.append(
                f"Tree leaf '{path}' has a shape suffix different from expected: "
                f"{suffix} != {shape_suffix_tuple}."
            )

    for path, leaf in tree_util.tree_flatten_with_path(tree)[0]:
        _assert_fn(str(path), leaf)
    if errors:
        raise AssertionError("\n".join(errors))


def assert_trees_all_equal_structs(
    *trees: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that trees have the same structure."""
    if len(trees) < 2:
        raise ValueError(
            "assert_trees_all_equal_structs on a single tree does not make sense. "
            "Maybe you wrote `assert_trees_all_equal_structs([a, b])` instead of "
            "`assert_trees_all_equal_structs(a, b)` ?"
        )

    first_treedef = tree_util.tree_structure(trees[0])
    other_treedefs = (tree_util.tree_structure(t) for t in trees[1:])
    for i, treedef in enumerate(other_treedefs, start=1):
        if first_treedef != treedef:
            raise AssertionError(
                f"Error in tree structs equality check: trees 0 and {i} do not match,\n"
                f" tree 0: {first_treedef}\n"
                f" tree {i}: {treedef}"
            )


def assert_trees_all_equal_comparator(
    equality_comparator: "Callable[[Any, Any], bool]",
    error_msg_fn: "Callable[[Any, Any], str]",
    *trees: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that all trees are equal as per the custom comparator for leaves."""
    if len(trees) < 2:
        raise ValueError(
            "Assertions over only one tree does not make sense. Maybe you wrote "
            "`assert_trees_xxx([a, b])` instead of `assert_trees_xxx(a, b)`, or "
            "forgot the `error_msg_fn` arg to `assert_trees_all_equal_comparator`?"
        )
    assert_trees_all_equal_structs(*trees)

    paths = [str(path) for path, _ in tree_util.tree_flatten_with_path(trees[0])[0]]
    trees_leaves = [tree_util.tree_leaves(tree) for tree in trees]
    for leaf_i, path in enumerate(paths):
        first_leaf = trees_leaves[0][leaf_i]
        for tree_j in range(1, len(trees)):
            other_leaf = trees_leaves[tree_j][leaf_i]
            if not equality_comparator(first_leaf, other_leaf):
                msg = error_msg_fn(first_leaf, other_leaf)
                if path and path != "()":
                    raise AssertionError(
                        f"Trees 0 and {tree_j} differ in leaves '{path}': {msg}."
                    )
                else:
                    raise AssertionError(
                        f"Trees (arrays) 0 and {tree_j} differ: {msg}."
                    )


def assert_trees_all_close(
    *trees: Union[
        jax.Array,
        np.ndarray,
        np.bool_,
        np.number,
        Iterable["ArrayTree"],
        Mapping[Any, "ArrayTree"],
    ],
    rtol: float,
    atol: float = 0.0,
) -> None:
    """Checks that all trees have leaves with approximately equal values."""

    def assert_fn(arr_1: Any, arr_2: Any) -> None:
        if not jax.numpy.allclose(
            np.asarray(arr_1),
            np.asarray(arr_2),
            rtol=rtol,
            atol=atol,
        ):
            raise AssertionError(
                "Error in value equality check: Values not approximately equal"
            )

    def cmp_fn(arr_1: Any, arr_2: Any) -> bool:
        try:
            assert_fn(arr_1, arr_2)
        except AssertionError:
            return False
        return True

    def err_msg_fn(arr_1: Any, arr_2: Any) -> str:
        try:
            assert_fn(arr_1, arr_2)
        except AssertionError as e:
            return f"{str(e)} \nOriginal dtypes: {np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}"
        return ""  # pragma: no cover

    assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def assert_trees_all_close_ulp(
    *trees: Union[
        jax.Array,
        np.ndarray,
        np.bool_,
        np.number,
        Iterable["ArrayTree"],
        Mapping[Any, "ArrayTree"],
    ],
    maxulp: int,
) -> None:
    """Checks that tree leaves differ by at most `maxulp` Units in the Last Place."""

    def assert_fn(arr_1: Any, arr_2: Any) -> None:
        np.testing.assert_array_max_ulp(
            np.asarray(arr_1), np.asarray(arr_2), maxulp=maxulp
        )

    def cmp_fn(arr_1: Any, arr_2: Any) -> bool:
        try:
            assert_fn(arr_1, arr_2)
        except AssertionError:
            return False
        return True

    def err_msg_fn(arr_1: Any, arr_2: Any) -> str:
        try:
            assert_fn(arr_1, arr_2)
        except AssertionError as e:
            return f"{str(e)} \nOriginal dtypes: {np.asarray(arr_1).dtype}, {np.asarray(arr_2).dtype}"
        return ""  # pragma: no cover

    assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def assert_trees_all_equal(
    *trees: Union[
        jax.Array,
        np.ndarray,
        np.bool_,
        np.number,
        Iterable["ArrayTree"],
        Mapping[Any, "ArrayTree"],
    ],
    strict: bool,
) -> None:
    """Checks that all trees have leaves with *exactly* equal values."""

    def assert_fn(arr_1: Any, arr_2: Any) -> None:
        if not jax.numpy.array_equal(
            np.asarray(arr_1),
            np.asarray(arr_2),
        ):
            raise AssertionError(
                "Error in value equality check: Values not exactly equal"
            )

    def cmp_fn(arr_1: Any, arr_2: Any) -> bool:
        try:
            assert_fn(arr_1, arr_2)
        except AssertionError:
            return False
        return True

    def err_msg_fn(arr_1: Any, arr_2: Any) -> str:
        try:
            assert_fn(arr_1, arr_2)
        except AssertionError as e:
            dtype_1 = (
                arr_1.dtype if hasattr(arr_1, "dtype") else np.asarray(arr_1).dtype
            )
            dtype_2 = (
                arr_2.dtype if hasattr(arr_2, "dtype") else np.asarray(arr_2).dtype
            )
            return f"{str(e)} \nOriginal dtypes: {dtype_1}, {dtype_2}"
        return ""  # pragma: no cover

    assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def assert_trees_all_equal_dtypes(
    *trees: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that trees' leaves have the same dtype."""

    def cmp_fn(arr_1: Any, arr_2: Any) -> bool:
        return (
            hasattr(arr_1, "dtype")
            and hasattr(arr_2, "dtype")
            and arr_1.dtype == arr_2.dtype
        )

    def err_msg_fn(arr_1: Any, arr_2: Any) -> str:
        if not hasattr(arr_1, "dtype"):
            return f"{type(arr_1)} is not a (j-)np array (has no `dtype` property)"
        if not hasattr(arr_2, "dtype"):
            return f"{type(arr_2)} is not a (j-)np array (has no `dtype` property)"
        return f"types: {arr_1.dtype} != {arr_2.dtype}"

    assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def assert_trees_all_equal_shapes(
    *trees: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that trees have the same structure and leaves' shapes."""
    cmp_fn = lambda arr_1, arr_2: arr_1.shape == arr_2.shape
    err_msg_fn = lambda arr_1, arr_2: f"shapes: {arr_1.shape} != {arr_2.shape}"
    assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)


def assert_trees_all_equal_shapes_and_dtypes(
    *trees: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that trees' leaves have the same shape and dtype."""
    assert_trees_all_equal_shapes(*trees)
    assert_trees_all_equal_dtypes(*trees)


def assert_trees_all_equal_sizes(
    *trees: "jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]] | Mapping[Any, jax.Array | numpy.ndarray | numpy.bool | numpy.number | Iterable[ForwardRef('ArrayTree')] | Mapping[Any, ForwardRef('ArrayTree')]]",
) -> "NoneType":
    """Checks that trees have the same structure and leaves' sizes."""
    cmp_fn = lambda arr_1, arr_2: arr_1.size == arr_2.size
    err_msg_fn = lambda arr_1, arr_2: f"sizes: {arr_1.size} != {arr_2.size}"
    assert_trees_all_equal_comparator(cmp_fn, err_msg_fn, *trees)
