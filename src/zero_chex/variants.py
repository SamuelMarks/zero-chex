"""Test case variants, ChexVariantType enum, and variant decorators for zero_chex."""

from __future__ import annotations

import functools
import unittest
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Generic, Protocol, TypeVar, Union, overload, runtime_checkable

import zero_jax
from typing_extensions import ParamSpec, TypeGuard

from zero_chex._src.asserts_internal import (
    RUNTIME_STATE as _STATE,
)
from zero_chex._src.asserts_internal import (
    check_backend_permitted,
)
from zero_chex._src.tree_util import tree_map
from zero_chex.mocks import _abstract_arg_signature, _fake_pmap_impl, _is_pytree_result
from zero_chex.types import Device, JaxNdarray, LeafValue, PyTree, ZeroArray

P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
TInstance = TypeVar("TInstance")
VariantKwargValue = Union[bool, int, str, Sequence[int], Sequence[str], Device, None]


@runtime_checkable
class VariantAttributeHolderProtocol(Protocol):
    """Protocol for custom test case classes holding a mutable `variant` attribute."""

    variant: VariantDecorator | None


class ChexVariantType(Enum):
    """Enumeration of test execution variants supported by Chex."""

    WITH_JIT = "_with_jit"
    WITHOUT_JIT = "_without_jit"
    WITH_DEVICE = "_with_device"
    WITHOUT_DEVICE = "_without_device"
    WITH_PMAP = "_with_pmap"
    WITHOUT_PMAP = "_without_pmap"

    def __str__(self) -> str:
        """Return the string value of the variant type.

        Returns:
            The variant suffix string.
        """
        return self.value


def _place_leaf_on_device(leaf: LeafValue, target_dev: Device) -> LeafValue:
    """Place a single PyTree leaf onto target_dev if it is a ZeroArray or JaxNdarray.

    Args:
        leaf: Leaf value to transfer.
        target_dev: Target Device instance.

    Returns:
        Device-placed leaf value.
    """
    if isinstance(leaf, ZeroArray):
        return ZeroArray(
            leaf.to_list(),
            shape=leaf.shape,
            dtype=leaf.dtype,
            device=target_dev,
            sharding=leaf.sharding,
        )
    if isinstance(leaf, JaxNdarray):
        return zero_jax.device_put(leaf, target_dev.platform)
    return leaf


def _is_int_or_int_seq(val: VariantKwargValue) -> TypeGuard[int | Sequence[int]]:
    """Check whether val is an integer or a sequence of integers.

    Args:
        val: Candidate VariantKwargValue.

    Returns:
        True if val is an int (not bool) or Sequence[int].
    """
    if isinstance(val, int) and not isinstance(val, bool):
        return True
    return (
        isinstance(val, Sequence)
        and not isinstance(val, str)
        and all(isinstance(x, int) and not isinstance(x, bool) for x in val)
    )


def _is_str_or_str_seq(val: VariantKwargValue) -> TypeGuard[str | Sequence[str]]:
    """Check whether val is a string or a sequence of strings.

    Args:
        val: Candidate VariantKwargValue.

    Returns:
        True if val is a str or Sequence[str].
    """
    if isinstance(val, str):
        return True
    return isinstance(val, Sequence) and all(isinstance(x, str) for x in val)


class VariantDecorator:
    """Callable variant wrapper attached to TestCase.variant during test execution."""

    def __init__(self, variant_type: ChexVariantType = ChexVariantType.WITHOUT_JIT) -> None:
        """Initialize a VariantDecorator for a given ChexVariantType.

        Args:
            variant_type: The active ChexVariantType for the test run.
        """
        self.type: ChexVariantType = variant_type
        self.call_count: int = 0

    def _wrap_target(
        self,
        target_fn: Callable[P, R],
        **variant_kwargs: VariantKwargValue,
    ) -> Callable[P, R]:
        """Wrap a target callable according to self.type and variant_kwargs.

        Args:
            target_fn: The function to transform under the active variant.
            **variant_kwargs: Optional variant transformation options (e.g., static_argnums, backend, device).

        Returns:
            Wrapped function executing under the variant's semantics.
        """
        vtype = self.type
        trace_key = _STATE.allocate_fn_key("variant_target")
        raw_backend = variant_kwargs.get("backend")
        raw_device = variant_kwargs.get("device")
        if isinstance(raw_backend, str):
            target_platform = raw_backend
        elif isinstance(raw_device, Device):
            target_platform = raw_device.platform
        elif isinstance(raw_device, str):
            target_platform = raw_device.split(":")[0]
        else:
            target_platform = "cpu"

        raw_static_nums = variant_kwargs.get("static_argnums")
        static_argnums: int | Sequence[int] | None = raw_static_nums if _is_int_or_int_seq(raw_static_nums) else None
        raw_static_names = variant_kwargs.get("static_argnames")
        static_argnames: str | Sequence[str] | None = raw_static_names if _is_str_or_str_seq(raw_static_names) else None
        raw_axis_name = variant_kwargs.get("axis_name")
        axis_name: str | None = raw_axis_name if isinstance(raw_axis_name, str) else None

        @functools.wraps(target_fn)
        def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            """Execute target_fn with variant-specific compilation, device, or pmap tracking.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Result of target_fn.

            Raises:
                AssertionError: If the active variant requires a backend forbidden by restrict_backends.
            """
            self.call_count += 1
            if vtype in (ChexVariantType.WITH_DEVICE, ChexVariantType.WITH_PMAP, ChexVariantType.WITH_JIT):
                check_backend_permitted(target_platform, "cpu")
            if vtype == ChexVariantType.WITH_JIT and not _STATE.jit_disabled:
                pytree_args = [a for a in args if _is_pytree_result(a)]
                pytree_kwargs = {k: v for k, v in kwargs.items() if _is_pytree_result(v)}
                sig = _abstract_arg_signature(
                    pytree_args,
                    pytree_kwargs,
                    static_argnums=static_argnums,
                    static_argnames=static_argnames,
                )
                with _STATE.lock:
                    seen = _STATE.trace_seen_signatures.setdefault(trace_key, set())
                    if sig not in seen:
                        seen.add(sig)
                        _STATE.trace_counts[trace_key] = _STATE.trace_counts.get(trace_key, 0) + 1
            elif vtype == ChexVariantType.WITH_PMAP:
                if _STATE.pmap_disabled:
                    faked = _fake_pmap_impl(target_fn, axis_name=axis_name)
                    return faked(*args, **kwargs)
                pytree_args = [a for a in args if _is_pytree_result(a)]
                pytree_kwargs = {k: v for k, v in kwargs.items() if _is_pytree_result(v)}
                sig = _abstract_arg_signature(pytree_args, pytree_kwargs)
                with _STATE.lock:
                    seen = _STATE.trace_seen_signatures.setdefault(trace_key, set())
                    if sig not in seen:
                        seen.add(sig)
                        _STATE.trace_counts[trace_key] = _STATE.trace_counts.get(trace_key, 0) + 1
            elif vtype == ChexVariantType.WITH_DEVICE:
                target_dev = raw_device if isinstance(raw_device, Device) else Device(target_platform, 0)

                def _place_fn(*leaves: LeafValue) -> LeafValue:
                    """Place the first leaf onto target_dev.

                    Args:
                        *leaves: Single-element leaf tuple.

                    Returns:
                        Placed leaf value.
                    """
                    return _place_leaf_on_device(leaves[0], target_dev)

                pytree_args = [a for a in args if _is_pytree_result(a)]
                pytree_kwargs = {k: v for k, v in kwargs.items() if _is_pytree_result(v)}
                placed_args = tree_map(_place_fn, pytree_args)
                placed_kwargs = tree_map(_place_fn, pytree_kwargs)
                assert isinstance(placed_args, list) and isinstance(placed_kwargs, dict)
                return functools.partial(target_fn, *placed_args, **placed_kwargs)()
            return target_fn(*args, **kwargs)

        return _wrapped

    @overload
    def __call__(
        self,
        fn: Callable[P, R],
        **kwargs: VariantKwargValue,
    ) -> Callable[P, R]: ...

    @overload
    def __call__(
        self,
        fn: None = None,
        **kwargs: VariantKwargValue,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def __call__(
        self,
        fn: Callable[P, R] | None = None,
        **kwargs: VariantKwargValue,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorate a function under the active test variant.

        Args:
            fn: The function to wrap, or None when parameterized with keyword arguments.
            **kwargs: Optional compilation arguments (e.g., static_argnums, backend, device).

        Returns:
            The wrapped function or a decorator accepting the function.
        """
        if fn is None:

            def _dec(target_fn: Callable[P, R]) -> Callable[P, R]:
                """Transform target_fn for execution under the active variant.

                Args:
                    target_fn: The function to wrap.

                Returns:
                    The transformed callable.
                """
                return self._wrap_target(target_fn, **kwargs)

            return _dec
        return self._wrap_target(fn, **kwargs)


class TestCase(unittest.TestCase):
    """Base test case class providing support for @variants and @all_variants decorators."""

    VARIANTS_MAIN_TEST_FN_NAME: str = "__variants_main_test_fn"
    _active_variant: VariantDecorator | None = None

    @overload
    def variant(
        self,
        fn: Callable[P, R],
        **kwargs: VariantKwargValue,
    ) -> Callable[P, R]: ...

    @overload
    def variant(
        self,
        fn: None = None,
        **kwargs: VariantKwargValue,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def variant(
        self,
        fn: Callable[P, R] | None = None,
        **kwargs: VariantKwargValue,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Transform a function according to the active test variant.

        Args:
            fn: The function to transform, or None when called with keyword arguments.
            **kwargs: Optional variant compilation arguments.

        Returns:
            The transformed function or decorator.

        Raises:
            RuntimeError: If self.variant is called outside a @variants decorated test method.
        """
        if self._active_variant is None:
            raise RuntimeError("self.variant is not defined; decorate the test method with @chex.variants.")
        return self._active_variant(fn, **kwargs)


class BoundMethodProtocol(Protocol[R_co]):
    """Protocol for a bound test method accepting zero or more PyTree arguments."""

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> R_co:
        """Invoke the bound test method.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the test method.
        """
        ...


class BoundVariantsTestMethod(Generic[TInstance, R]):
    """Bound callable representing a VariantsTestMethodWrapper bound to a test instance."""

    def __init__(
        self,
        bound_func: BoundMethodProtocol[R],
        wrapper: VariantsTestMethodWrapper[P, R],
        instance: TInstance,
    ) -> None:
        """Initialize a BoundVariantsTestMethod.

        Args:
            bound_func: The underlying method already bound to instance via types.MethodType.
            wrapper: The originating VariantsTestMethodWrapper.
            instance: The bound test class instance.
        """
        self._bound_func = bound_func
        self._wrapper = wrapper
        self._instance = instance

    def __call__(self, *args: PyTree, **kwargs: PyTree) -> R:
        """Invoke the bound test method across all configured variants.

        Args:
            *args: Positional arguments forwarded to the bound test method.
            **kwargs: Keyword arguments forwarded to the bound test method.

        Returns:
            The return value from the final executed variant.
        """
        self._wrapper.executed_variants.clear()
        vtypes = self._wrapper.variant_types
        call_target = functools.partial(self._bound_func, *args, **kwargs)
        result: R = self._wrapper._run_bound_variant(
            vtypes[0],
            self._instance,
            call_target,
        )
        for vtype in vtypes[1:]:
            result = self._wrapper._run_bound_variant(
                vtype,
                self._instance,
                call_target,
            )
        return result


class VariantsTestMethodWrapper(Generic[P, R]):
    """Callable test method wrapper that runs all enabled variants and binds self._active_variant."""

    def __init__(
        self,
        func: Callable[P, R],
        variant_types: Sequence[ChexVariantType] = (ChexVariantType.WITHOUT_JIT,),
    ) -> None:
        """Initialize a VariantsTestMethodWrapper.

        Args:
            func: The test method to wrap.
            variant_types: Ordered sequence of ChexVariantType variants to execute.
        """
        self._func = func
        self._variant_types: tuple[ChexVariantType, ...] = (
            tuple(variant_types) if variant_types else (ChexVariantType.WITHOUT_JIT,)
        )
        self.executed_variants: list[ChexVariantType] = []
        functools.update_wrapper(self, func)

    @property
    def variant_types(self) -> tuple[ChexVariantType, ...]:
        """Return the tuple of enabled ChexVariantType members for this test wrapper.

        Returns:
            Tuple of ChexVariantType members.
        """
        return self._variant_types

    def _run_bound_variant(
        self,
        vtype: ChexVariantType,
        instance: TInstance,
        bound_call: Callable[[], R],
    ) -> R:
        """Execute a bound test method under a single ChexVariantType.

        Args:
            vtype: The variant type to activate.
            instance: Bound test class instance.
            bound_call: Zero-argument callable bound to instance and call arguments.

        Returns:
            The return value of bound_call().
        """
        decorator = VariantDecorator(vtype)
        self.executed_variants.append(vtype)
        if isinstance(instance, TestCase):
            prev = instance._active_variant
            instance._active_variant = decorator
            try:
                with instance.subTest(variant=vtype.value):
                    return bound_call()
            finally:
                instance._active_variant = prev
        if isinstance(instance, VariantAttributeHolderProtocol):
            prev_holder = instance.variant
            instance.variant = decorator
            try:
                return bound_call()
            finally:
                instance.variant = prev_holder
        return bound_call()

    def _run_single_variant(
        self,
        vtype: ChexVariantType,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute the wrapped test function under a single ChexVariantType.

        Args:
            vtype: The variant type to activate.
            *args: Positional arguments forwarded to self._func.
            **kwargs: Keyword arguments forwarded to self._func.

        Returns:
            The return value of self._func(*args, **kwargs).
        """
        decorator = VariantDecorator(vtype)
        self.executed_variants.append(vtype)
        if args and isinstance(args[0], TestCase):
            test_instance = args[0]
            prev = test_instance._active_variant
            test_instance._active_variant = decorator
            try:
                with test_instance.subTest(variant=vtype.value):
                    return self._func(*args, **kwargs)
            finally:
                test_instance._active_variant = prev
        if args and isinstance(args[0], VariantAttributeHolderProtocol):
            holder = args[0]
            prev_holder = holder.variant
            holder.variant = decorator
            try:
                return self._func(*args, **kwargs)
            finally:
                holder.variant = prev_holder
        return self._func(*args, **kwargs)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Invoke the wrapped test method across all configured variants.

        Args:
            *args: Positional arguments (first argument is typically the TestCase self).
            **kwargs: Keyword arguments forwarded to the test method.

        Returns:
            The return value of the final variant execution.
        """
        self.executed_variants.clear()
        result: R = self._run_single_variant(self._variant_types[0], *args, **kwargs)
        for vtype in self._variant_types[1:]:
            result = self._run_single_variant(vtype, *args, **kwargs)
        return result

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[TInstance] | None = None,
    ) -> VariantsTestMethodWrapper[P, R]: ...

    @overload
    def __get__(
        self,
        instance: TInstance,
        owner: type[TInstance] | None = None,
    ) -> BoundVariantsTestMethod[TInstance, R]: ...

    def __get__(
        self,
        instance: TInstance | None,
        owner: type[TInstance] | None = None,
    ) -> VariantsTestMethodWrapper[P, R] | BoundVariantsTestMethod[TInstance, R]:
        """Support descriptor binding when accessed as a method on a TestCase instance.

        Args:
            instance: The TestCase instance, or None if accessed on the class.
            owner: The owner class.

        Returns:
            A BoundVariantsTestMethod callable or self.
        """
        if instance is None:
            return self
        bound_func: BoundMethodProtocol[R] = self._func.__get__(instance, owner)
        return BoundVariantsTestMethod(bound_func, self, instance)


def _select_variant_types(
    with_jit: bool,
    without_jit: bool,
    with_device: bool,
    without_device: bool,
    with_pmap: bool,
    without_pmap: bool = False,
) -> tuple[ChexVariantType, ...]:
    """Collect all enabled ChexVariantType members in canonical order.

    Args:
        with_jit: Whether WITH_JIT is enabled.
        without_jit: Whether WITHOUT_JIT is enabled.
        with_device: Whether WITH_DEVICE is enabled.
        without_device: Whether WITHOUT_DEVICE is enabled.
        with_pmap: Whether WITH_PMAP is enabled.
        without_pmap: Whether WITHOUT_PMAP is enabled.

    Returns:
        Tuple of enabled ChexVariantType enum members.
    """
    selected: list[ChexVariantType] = []
    if with_jit:
        selected.append(ChexVariantType.WITH_JIT)
    if without_jit:
        selected.append(ChexVariantType.WITHOUT_JIT)
    if with_device:
        selected.append(ChexVariantType.WITH_DEVICE)
    if without_device:
        selected.append(ChexVariantType.WITHOUT_DEVICE)
    if with_pmap:
        selected.append(ChexVariantType.WITH_PMAP)
    if without_pmap:
        selected.append(ChexVariantType.WITHOUT_PMAP)
    if not selected:
        selected.append(ChexVariantType.WITHOUT_PMAP)
    return tuple(selected)


@overload
def variants(
    test_method: Callable[P, R],
    with_jit: bool | int | str | None = False,
    without_jit: bool | int | str | None = False,
    with_device: bool | int | str | None = False,
    without_device: bool | int | str | None = False,
    with_pmap: bool | int | str | None = False,
) -> VariantsTestMethodWrapper[P, R]: ...


@overload
def variants(
    test_method: str = "__no__default__",
    with_jit: bool | int | str | None = False,
    without_jit: bool | int | str | None = False,
    with_device: bool | int | str | None = False,
    without_device: bool | int | str | None = False,
    with_pmap: bool | int | str | None = False,
) -> Callable[[Callable[P, R]], VariantsTestMethodWrapper[P, R]]: ...


def variants(
    test_method: Callable[P, R] | str = "__no__default__",
    with_jit: bool | int | str | None = False,
    without_jit: bool | int | str | None = False,
    with_device: bool | int | str | None = False,
    without_device: bool | int | str | None = False,
    with_pmap: bool | int | str | None = False,
) -> VariantsTestMethodWrapper[P, R] | Callable[[Callable[P, R]], VariantsTestMethodWrapper[P, R]]:
    """Decorate a test method to run under specified compilation/device variants.

    Args:
        test_method: The test method to wrap, or sentinel when called with keyword arguments.
        with_jit: Execute test variant with JIT compilation enabled.
        without_jit: Execute test variant without JIT compilation.
        with_device: Execute test variant with inputs placed on device.
        without_device: Execute test variant with inputs in host memory.
        with_pmap: Execute test variant with PMAP parallelization.

    Returns:
        Wrapped test method or a decorator returning a wrapped test method.
    """
    check_variant_arguments(
        with_jit=with_jit,
        without_jit=without_jit,
        with_device=with_device,
        without_device=without_device,
        with_pmap=with_pmap,
    )
    selected = _select_variant_types(
        with_jit=bool(with_jit),
        without_jit=bool(without_jit),
        with_device=bool(with_device),
        without_device=bool(without_device),
        with_pmap=bool(with_pmap),
    )

    def _decorator(func: Callable[P, R]) -> VariantsTestMethodWrapper[P, R]:
        """Wrap func in a VariantsTestMethodWrapper.

        Args:
            func: Test method to wrap.

        Returns:
            VariantsTestMethodWrapper instance.
        """
        return VariantsTestMethodWrapper(func, variant_types=selected)

    if callable(test_method):
        return _decorator(test_method)
    return _decorator


@overload
def all_variants(
    test_method: Callable[P, R],
    with_jit: bool = True,
    without_jit: bool = True,
    with_device: bool = True,
    without_device: bool = True,
    with_pmap: bool = True,
) -> VariantsTestMethodWrapper[P, R]: ...


@overload
def all_variants(
    test_method: str = "__no__default__",
    with_jit: bool = True,
    without_jit: bool = True,
    with_device: bool = True,
    without_device: bool = True,
    with_pmap: bool = True,
) -> Callable[[Callable[P, R]], VariantsTestMethodWrapper[P, R]]: ...


def all_variants(
    test_method: Callable[P, R] | str = "__no__default__",
    with_jit: bool = True,
    without_jit: bool = True,
    with_device: bool = True,
    without_device: bool = True,
    with_pmap: bool = True,
) -> VariantsTestMethodWrapper[P, R] | Callable[[Callable[P, R]], VariantsTestMethodWrapper[P, R]]:
    """Decorate a test method to run across all standard Chex variants.

    Args:
        test_method: The test method to wrap, or sentinel when called with keyword arguments.
        with_jit: Execute test variant with JIT compilation enabled.
        without_jit: Execute test variant without JIT compilation.
        with_device: Execute test variant with inputs placed on device.
        without_device: Execute test variant with inputs in host memory.
        with_pmap: Execute test variant with PMAP parallelization.

    Returns:
        Wrapped test method or a decorator returning a wrapped test method.
    """
    if callable(test_method):
        return variants(
            test_method,
            with_jit=with_jit,
            without_jit=without_jit,
            with_device=with_device,
            without_device=without_device,
            with_pmap=with_pmap,
        )
    return variants(
        with_jit=with_jit,
        without_jit=without_jit,
        with_device=with_device,
        without_device=without_device,
        with_pmap=with_pmap,
    )


def count_num_calls(fn: Callable[P, R]) -> tuple[Callable[P, R], Callable[[], int]]:
    """Wrap a function and return a pair of (wrapped_fn, get_num_calls_fn).

    Args:
        fn: Target callable whose invocations should be counted.

    Returns:
        A 2-tuple of (wrapped_fn, count_getter) where count_getter() returns total calls.
    """
    calls: list[int] = [0]

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Increment call counter and invoke fn.

        Args:
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            Result of fn(*args, **kwargs).
        """
        calls[0] += 1
        return fn(*args, **kwargs)

    def _get_count() -> int:
        """Return the number of times _wrapped has been called.

        Returns:
            Integer call count.
        """
        return calls[0]

    return _wrapped, _get_count


_VALID_VARIANT_NAMES: frozenset[str] = frozenset(
    {
        "with_jit",
        "without_jit",
        "with_device",
        "without_device",
        "with_pmap",
        "without_pmap",
    }
)


def check_variant_arguments(**kwargs: PyTree) -> None:
    """Validate that keyword arguments passed to a variant decorator are known boolean flags.

    Args:
        **kwargs: Variant keyword arguments to validate.

    Raises:
        ValueError: If an unknown variant argument name is provided.
        TypeError: If any variant argument value is not a boolean.
    """
    for k, v in kwargs.items():
        if k not in _VALID_VARIANT_NAMES:
            raise ValueError(f"Unknown variant argument: {k!r}.")
        if not isinstance(v, bool):
            raise TypeError(f"Variant argument {k!r} must be a bool, got {type(v).__name__}.")


__all__ = [
    "BoundVariantsTestMethod",
    "ChexVariantType",
    "TestCase",
    "VariantAttributeHolderProtocol",
    "VariantDecorator",
    "VariantsTestMethodWrapper",
    "all_variants",
    "check_variant_arguments",
    "count_num_calls",
    "variants",
]
