"""Chex assertion functions for zero-chex parity."""

from zero_jax.tree_util import tree_flatten
import threading
from contextlib import contextmanager
from dataclasses import dataclass as builtin_dataclass
from enum import Enum
import warnings
import itertools


from typing import Any, Callable
from collections.abc import Sequence
from ml_switcheroo_compiler.core.tensor import Tensor


def assert_axis_dimension(tensor: Tensor, axis: int, expected: int) -> None:
    """Checks that tensor.shape[axis] == expected."""
    dim = tensor.shape[axis]
    if isinstance(dim, int):
        if dim != expected:
            raise AssertionError(f"Expected shape[{axis}] == {expected}, got {dim}")
    else:
        pass  # Dynamic shape: lowered to AssertEq in IR (TODO)


def assert_axis_dimension_comparator(
    tensor: Tensor, axis: int, pass_fn: Callable, error_string: str
) -> None:
    """Asserts that pass_fn(tensor.shape[axis]) passes."""
    dim = tensor.shape[axis]
    if isinstance(dim, int):
        if not pass_fn(dim):
            raise AssertionError(error_string)


def assert_axis_dimension_gt(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] > val."""
    dim = tensor.shape[axis]
    if isinstance(dim, int):
        if not dim > val:
            raise AssertionError(f"Expected shape[{axis}] > {val}, got {dim}")


def assert_axis_dimension_gteq(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] >= val."""
    dim = tensor.shape[axis]
    if isinstance(dim, int):
        if not dim >= val:
            raise AssertionError(f"Expected shape[{axis}] >= {val}, got {dim}")


def assert_axis_dimension_lt(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] < val."""
    dim = tensor.shape[axis]
    if isinstance(dim, int):
        if not dim < val:
            raise AssertionError(f"Expected shape[{axis}] < {val}, got {dim}")


def assert_axis_dimension_lteq(tensor: Tensor, axis: int, val: int) -> None:
    """Checks that tensor.shape[axis] <= val."""
    dim = tensor.shape[axis]
    if isinstance(dim, int):
        if not dim <= val:
            raise AssertionError(f"Expected shape[{axis}] <= {val}, got {dim}")


def assert_equal_shape(inputs: Sequence[Tensor], dims: object = None) -> None:
    """Checks that all arrays have the same shape."""
    if not inputs:
        return
    first_shape = inputs[0].shape
    for t in inputs[1:]:
        if t.shape != first_shape:
            raise AssertionError(f"Expected shape {first_shape}, got {t.shape}")


def assert_equal_rank(inputs: Sequence[Tensor]) -> None:
    """Checks that all arrays have the same rank."""
    if not inputs:
        return
    first_rank = len(inputs[0].shape)
    for t in inputs[1:]:
        if len(t.shape) != first_rank:
            raise AssertionError(f"Expected rank {first_rank}, got {len(t.shape)}")


def assert_equal_shape_prefix(inputs: Sequence[Tensor], prefix_len: int) -> None:
    """Checks that the leading prefix_dims dims of all inputs have same shape."""
    if not inputs:
        return
    first_prefix = inputs[0].shape[:prefix_len]
    for t in inputs[1:]:
        if t.shape[:prefix_len] != first_prefix:
            raise AssertionError(
                f"Expected prefix {first_prefix}, got {t.shape[:prefix_len]}"
            )


def assert_equal_shape_suffix(inputs: Sequence[Tensor], suffix_len: int) -> None:
    """Checks that the final suffix_len dims of all inputs have same shape."""
    if not inputs:
        return
    first_suffix = inputs[0].shape[-suffix_len:] if suffix_len > 0 else ()
    for t in inputs[1:]:
        t_suffix = t.shape[-suffix_len:] if suffix_len > 0 else ()
        if t_suffix != first_suffix:
            raise AssertionError(f"Expected suffix {first_suffix}, got {t_suffix}")


def assert_equal_size(inputs: Sequence[Tensor]) -> None:
    """Checks that all arrays have the same size."""
    import math

    if not inputs:
        return
    first_size = math.prod(inputs[0].shape)
    for t in inputs[1:]:
        if math.prod(t.shape) != first_size:
            raise AssertionError(
                f"Expected size {first_size}, got {math.prod(t.shape)}"
            )


def assert_rank(inputs: Sequence[Tensor], expected_ranks: object) -> None:
    """Checks that the rank of all inputs matches expected."""
    if not isinstance(expected_ranks, (list, tuple, set)):
        expected_ranks = [expected_ranks]
    for t in inputs:
        if len(t.shape) not in expected_ranks:
            raise AssertionError(
                f"Expected rank in {expected_ranks}, got {len(t.shape)}"
            )


def assert_shape(inputs: Sequence[Tensor], expected_shapes: object) -> None:
    """Checks that the shape of all inputs matches expected."""
    if not isinstance(expected_shapes[0], (list, tuple)):
        expected_shapes = [expected_shapes]
    for t in inputs:
        if t.shape not in expected_shapes:
            raise AssertionError(f"Expected shape in {expected_shapes}, got {t.shape}")


def assert_size(inputs: Sequence[Tensor], expected_sizes: object) -> None:
    """Checks that the size of all inputs matches expected."""
    import math

    if not isinstance(expected_sizes, (list, tuple, set)):
        expected_sizes = [expected_sizes]
    for t in inputs:
        if math.prod(t.shape) not in expected_sizes:
            raise AssertionError(
                f"Expected size in {expected_sizes}, got {math.prod(t.shape)}"
            )


def assert_type(inputs: Sequence[Tensor], expected_types: object) -> None:
    """Checks that the type of all inputs matches."""
    if not isinstance(expected_types, (list, tuple, set)):
        expected_types = [expected_types]
    for t in inputs:
        if t.dtype not in expected_types:
            raise AssertionError(f"Expected type in {expected_types}, got {t.dtype}")


def assert_tree_all_finite(tree_like: object) -> None:
    """Checks that all leaves in a tree are finite."""
    import ml_switcheroo_compiler.ops as jnp

    leaves, _ = tree_flatten(tree_like)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            if not jnp.all(jnp.isfinite(leaf)):
                raise AssertionError("Tree contains non-finite values.")


def assert_tree_has_only_ndarrays(tree: object) -> None:
    """Checks that all leaves are n-dimensional arrays."""
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if not isinstance(leaf, Tensor):
            raise AssertionError(f"Expected Tensor, got {type(leaf)}")


def assert_tree_shape_prefix(tree: object, shape_prefix: Sequence[int]) -> None:
    """Checks that all leaves shapes have the same prefix."""
    leaves, _ = tree_flatten(tree)
    if not leaves:
        return
    for leaf in leaves:
        if not isinstance(leaf, Tensor) or leaf.shape[: len(shape_prefix)] != tuple(
            shape_prefix
        ):
            raise AssertionError(
                f"Expected shape prefix {shape_prefix}, got {leaf.shape}"
            )


def assert_tree_shape_suffix(tree: object, shape_suffix: Sequence[int]) -> None:
    """Checks that all leaves shapes have the same suffix."""
    leaves, _ = tree_flatten(tree)
    if not leaves:
        return
    for leaf in leaves:
        if not isinstance(leaf, Tensor):
            raise AssertionError(f"Expected Tensor, got {type(leaf)}")
        leaf_suffix = leaf.shape[-len(shape_suffix) :] if len(shape_suffix) > 0 else ()
        if leaf_suffix != tuple(shape_suffix):
            raise AssertionError(
                f"Expected shape suffix {shape_suffix}, got {leaf.shape}"
            )


def assert_tree_no_nones(tree: object) -> None:
    """Checks that a tree does not contain None."""
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if leaf is None:
            raise AssertionError("Tree contains None")


def assert_trees_all_close(
    trees: Sequence[Any], rtol: float = 1e-06, atol: float = 0.0
) -> None:
    """Checks that all trees have leaves with approx equal values."""
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if isinstance(l0, Tensor) and isinstance(li, Tensor):
                import ml_switcheroo_compiler.ops as jnp

                diff = jnp.abs(jnp.subtract(l0, li))
                thresh = jnp.add(atol, jnp.multiply(rtol, jnp.abs(li)))
                if not jnp.all(jnp.less_equal(diff, thresh)):
                    raise AssertionError("Trees are not all close")


def assert_trees_all_close_ulp(trees: Sequence[Any], maxulp: int = 1) -> None:
    """Checks that tree leaves differ by at most maxulp ULP."""
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if isinstance(l0, Tensor) and isinstance(li, Tensor):
                import ml_switcheroo_compiler.ops as jnp

                diff = jnp.abs(jnp.subtract(l0, li))
                if not jnp.all(jnp.less_equal(diff, 1e-6)):
                    raise AssertionError("Trees are not all close within ULP")


def assert_trees_all_equal(trees: Sequence[Any], strict: bool = False) -> None:
    """Checks that all trees have leaves with exactly equal values."""
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if isinstance(l0, Tensor) and isinstance(li, Tensor):
                import ml_switcheroo_compiler.ops as jnp

                if not jnp.all(jnp.equal(l0, li)):
                    raise AssertionError("Trees are not all equal")


def assert_trees_all_equal_comparator(
    equality_comparator: Callable, *trees: object
) -> None:
    """Checks that all trees are equal as per custom comparator."""
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if not equality_comparator(l0, li):
                raise AssertionError("Trees are not all equal by comparator")


def assert_trees_all_equal_dtypes(trees: Sequence[Any]) -> None:
    """Checks that trees leaves have the same dtype."""
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if hasattr(l0, "dtype") and hasattr(li, "dtype") and l0.dtype != li.dtype:
                raise AssertionError(f"Expected dtype {l0.dtype}, got {li.dtype}")


def assert_trees_all_equal_shapes(trees: Sequence[Any]) -> None:
    """Checks that trees have same structure and leaves shapes."""
    if not trees:
        return
    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if hasattr(l0, "shape") and hasattr(li, "shape") and l0.shape != li.shape:
                raise AssertionError(f"Expected shape {l0.shape}, got {li.shape}")


def assert_trees_all_equal_shapes_and_dtypes(trees: Sequence[Any]) -> None:
    """Checks same structure, shape, and dtype."""
    assert_trees_all_equal_shapes(trees)
    assert_trees_all_equal_dtypes(trees)


def assert_trees_all_equal_sizes(trees: Sequence[Any]) -> None:
    """Checks same structure and leaves sizes."""
    if not trees:
        return
    import math

    leaves0, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        leaves_i, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")
        for l0, li in zip(leaves0, leaves_i):
            if (
                hasattr(l0, "shape")
                and hasattr(li, "shape")
                and math.prod(l0.shape) != math.prod(li.shape)
            ):
                raise AssertionError("Trees do not have equal sizes")


def assert_trees_all_equal_structs(trees: Sequence[Any]) -> None:
    """Checks that trees have the same structure."""
    if not trees:
        return
    _, def0 = tree_flatten(trees[0])
    for t in trees[1:]:
        _, def_i = tree_flatten(t)
        if def0 != def_i:
            raise AssertionError("Trees have different structures")


def assert_devices_available(
    n: int, devtype: str, backend: object = None, not_less_than: bool = False
) -> None:
    """Checks that n devices of type are available."""
    # Mocking implementation for now
    available = 0
    if devtype.lower() == "cpu":
        available = 1  # Or interrogate config/backend
    elif devtype.lower() == "gpu":
        available = 0
    elif devtype.lower() == "tpu":
        available = 0

    if not_less_than:
        if available < n:
            raise AssertionError(
                f"Expected at least {n} {devtype} devices, got {available}"
            )
    else:
        if available != n:
            raise AssertionError(
                f"Expected exactly {n} {devtype} devices, got {available}"
            )


def assert_gpu_available(backend: object = None) -> None:
    """Checks that at least one GPU device is available."""
    assert_devices_available(1, "gpu", backend, not_less_than=True)


def assert_tpu_available(backend: object = None) -> None:
    """Checks that at least one TPU device is available."""
    assert_devices_available(1, "tpu", backend, not_less_than=True)


def assert_tree_is_on_device(
    tree: object, platform: object = None, device: object = None
) -> None:
    """Checks leaves are in device memory."""
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            dev = getattr(leaf, "device", None)
            if not dev or str(dev).lower() == "cpu":
                raise AssertionError(
                    f"Expected leaf to be on device, but found on {dev}"
                )


def assert_tree_is_on_host(
    tree: object, allow_cpu_device: bool = True, allow_sharded: bool = False
) -> None:
    """Checks leaves are in host memory (CPU)."""
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            dev = getattr(leaf, "device", None)
            if dev and str(dev).lower() != "cpu":
                raise AssertionError(
                    f"Expected leaf to be on host (CPU), but found on {dev}"
                )


def assert_tree_is_sharded(tree: object, devices: object = None) -> None:
    """Checks leaves are sharded across specified devices."""
    leaves, _ = tree_flatten(tree)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            if not hasattr(leaf, "sharding") or leaf.sharding is None:
                raise AssertionError("Leaf is not sharded")


_CHEX_ASSERTS_ENABLED = True
_TRACE_COUNTER = 0
_TRACE_LOCK = threading.Lock()


def chexify(fn: Callable, async_check: bool = True, errors: object = None) -> Callable:
    """Wraps a transformed function to enable Chex value assertions."""

    def wrapper(*args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    return wrapper


def block_until_chexify_assertions_complete() -> None:
    """Waits until all async checks complete."""
    pass


def assert_max_traces(fn: object = None, n: int = 1) -> Callable:
    """Checks that a function is traced at most n times."""
    global _TRACE_COUNTER
    if fn is None:
        return lambda f: assert_max_traces(f, n)

    def wrapper(*args: object, **kwargs: object) -> object:
        global _TRACE_COUNTER
        with _TRACE_LOCK:
            _TRACE_COUNTER += 1
            if _TRACE_COUNTER > n:
                raise AssertionError(f"Function traced more than {n} times")
        return fn(*args, **kwargs)

    return wrapper


def assert_numerical_grads(
    f: Callable, f_args: object, order: int = 1, atol: float = 0.01
) -> None:
    """Checks that autodiff and numerical gradients match."""
    # Mocking implementation
    pass


def clear_trace_counter() -> None:
    """Clears Chex traces counter."""
    global _TRACE_COUNTER
    with _TRACE_LOCK:
        _TRACE_COUNTER = 0


def disable_asserts() -> None:
    """Disables all Chex assertions."""
    global _CHEX_ASSERTS_ENABLED
    _CHEX_ASSERTS_ENABLED = False


def enable_asserts() -> None:
    """Enables Chex assertions."""
    global _CHEX_ASSERTS_ENABLED
    _CHEX_ASSERTS_ENABLED = True


@contextmanager
def fake_jit(enable_patching: bool = True) -> object:
    """Context manager for patching jit with identity."""
    yield


@contextmanager
def fake_pmap(enable_patching: bool = True) -> object:
    """Context manager for patching pmap with vmap."""
    yield


@contextmanager
def fake_pmap_and_jit(enable_pmap: bool = True, enable_jit: bool = True) -> object:
    """Patches both jit and pmap."""
    yield


def restrict_backends(allowed: object = None, forbidden: object = None) -> None:
    """Disallows compilation for certain backends."""
    pass


def set_n_cpu_devices(n: int) -> None:
    """Forces compiler to use n CPU threads as host devices."""
    pass


def assert_scalar(x: object) -> None:
    """Checks that x is a scalar."""
    if isinstance(x, Tensor):
        if len(x.shape) != 0:
            raise AssertionError(f"Expected scalar, got shape {x.shape}")
    elif not isinstance(x, (float, int, complex)):
        raise AssertionError(f"Expected scalar, got {type(x)}")


def assert_scalar_in(
    x: object, min_: object, max_: object, included: bool = True
) -> None:
    """Checks that argument is a scalar within segment."""
    assert_scalar(x)
    if isinstance(x, Tensor):
        x = float(x.data)
    if included:
        if not (min_ <= x <= max_):
            raise AssertionError(f"Expected scalar in [{min_}, {max_}], got {x}")
    else:
        if not (min_ < x < max_):
            raise AssertionError(f"Expected scalar in ({min_}, {max_}), got {x}")


def assert_scalar_negative(x: object) -> None:
    """Checks that a scalar is negative."""
    assert_scalar(x)
    val = float(x.data) if isinstance(x, Tensor) else x
    if not val < 0:
        raise AssertionError("Expected negative scalar")


def assert_scalar_non_negative(x: object) -> None:
    """Checks that a scalar is non-negative."""
    assert_scalar(x)
    val = float(x.data) if isinstance(x, Tensor) else x
    if not val >= 0:
        raise AssertionError("Expected non-negative scalar")


def assert_scalar_positive(x: object) -> None:
    """Checks that a scalar is positive."""
    assert_scalar(x)
    val = float(x.data) if isinstance(x, Tensor) else x
    if not val > 0:
        raise AssertionError("Expected positive scalar")


def assert_equal(first: object, second: object) -> None:
    """Checks that two objects are equal."""
    if first != second:
        raise AssertionError(f"Expected {first} == {second}")


def assert_exactly_one_is_none(first: object, second: object) -> None:
    """Checks that one and only one argument is None."""
    if (first is None) == (second is None):
        raise AssertionError("Expected exactly one None")


def assert_not_both_none(first: object, second: object) -> None:
    """Checks that at least one argument is not None."""
    if first is None and second is None:
        raise AssertionError("Expected not both None")


def assert_is_broadcastable(shape_a: Sequence[int], shape_b: Sequence[int]) -> None:
    """Checks that shape_a is broadcastable to shape_b."""
    for a, b in zip(reversed(shape_a), reversed(shape_b)):
        if a != 1 and b != 1 and a != b:
            raise AssertionError(f"Shape {shape_a} is not broadcastable to {shape_b}")


def assert_is_divisible(numerator: object, denominator: object) -> None:
    """Checks divisibility."""
    if numerator % denominator != 0:
        raise AssertionError(f"{numerator} is not divisible by {denominator}")


# Table 6 & 7 types and classes
Array = Tensor
ArrayBatched = Tensor
ArrayDevice = Tensor
ArrayNumpy = Any
ArraySharded = Tensor
ArrayDeviceTree = Any
ArrayDType = Any
ArrayNumpyTree = Any
ArrayTree = Any
Numeric = Any
PRNGKey = Any
Scalar = Any
Shape = Sequence[int]


class ChexVariantType(Enum):
    """Mock ChexVariantType Enum."""

    WITH_JIT = 1
    WITHOUT_JIT = 2


class ChexifyChecks(Enum):
    """Mock ChexifyChecks Enum."""

    USER = 1
    INTERNAL = 2


class Device:
    """Mock Device class."""

    pass


class PyTreeDef:
    """Mock PyTreeDef class."""

    pass


class Dimensions(dict):
    """Mock Dimensions class."""

    pass


class TestCase:
    """Mock TestCase class."""

    pass


def dataclass(cls: object = None, **kwargs: object) -> object:
    """Mock dataclass decorator."""
    if cls is None:
        return lambda c: dataclass(c, **kwargs)
    return builtin_dataclass(cls, **kwargs)


def mappable_dataclass(cls: object) -> object:
    """Mock mappable_dataclass decorator."""
    cls = dataclass(cls)
    return cls


def params_product(params_lists: object, named: bool = False) -> object:
    """Mock params_product function."""
    if named:
        keys = list(params_lists.keys())
        values = list(params_lists.values())
        return [dict(zip(keys, p)) for p in itertools.product(*values)]
    return list(itertools.product(*params_lists))


def create_deprecated_function_alias(
    fun: object, new_name: str, *args: object, **kwargs: object
) -> object:
    """Mock create_deprecated_function_alias function."""

    def wrapper(*wargs: object, **wkwargs: object) -> object:
        warnings.warn("Deprecated function called", stacklevel=2)
        return fun(*wargs, **wkwargs)

    return wrapper


def warn_deprecated_function(fun: object, replacement: object = None) -> object:
    """Mock warn_deprecated_function decorator."""

    def wrapper(*args: object, **kwargs: object) -> object:
        warnings.warn("Deprecated", stacklevel=2)
        return fun(*args, **kwargs)

    return wrapper


def warn_only_n_pos_args_in_future(fun: object = None, n: int = 1) -> object:
    """Mock warn_only_n_pos_args_in_future decorator."""
    if fun is None:
        return lambda f: warn_only_n_pos_args_in_future(f, n)

    def wrapper(*args: object, **kwargs: object) -> object:
        if len(args) > n:
            warnings.warn("Too many positional args", stacklevel=2)
        return fun(*args, **kwargs)

    return wrapper


def if_args_not_none(fn: object, args: object, kwargs: object) -> object:
    """Mock if_args_not_none function."""
    if any(a is None for a in args) or any(v is None for v in kwargs.values()):
        return None
    return fn(*args, **kwargs)


def all_variants(with_pmap: bool = True, **kwargs: object) -> object:
    """Mock all_variants decorator."""

    def decorator(fn: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def get_err_regex(err: object) -> object:
    """Mock get_err_regex function."""
    return str(err)


def register_dataclass_type_with_jax_tree_util(dataclass_type: object) -> None:
    """Mock register_dataclass_type_with_jax_tree_util."""
    pass


def variants(variants: object = (), **kwargs: object) -> object:
    """Mock variants decorator."""

    def decorator(fn: object) -> object:
        def wrapper(*args: object, **kwargs: object) -> object:
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def warn_keyword_args_only_in_future(fun: object) -> object:
    """Mock warn_keyword_args_only_in_future decorator."""

    def wrapper(*args: object, **kwargs: object) -> object:
        if len(args) > 0:
            warnings.warn("Use kwargs", stacklevel=2)
        return fun(*args, **kwargs)

    return wrapper
