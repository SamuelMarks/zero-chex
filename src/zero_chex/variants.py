"""Test case variants, ChexVariantType enum, and variant decorators for zero_chex."""

from __future__ import annotations

import functools
import unittest
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Generic, Protocol, TypeVar, overload, runtime_checkable

from typing_extensions import ParamSpec

from zero_chex.mocks import _STATE

P = ParamSpec("P")
R = TypeVar("R")
TInstance = TypeVar("TInstance")


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


class VariantDecorator:
    """Callable variant wrapper attached to TestCase.variant during test execution."""

    def __init__(self, variant_type: ChexVariantType = ChexVariantType.WITHOUT_JIT) -> None:
        """Initialize a VariantDecorator for a given ChexVariantType.

        Args:
            variant_type: The active ChexVariantType for the test run.
        """
        self.type: ChexVariantType = variant_type
        self.call_count: int = 0

    def _wrap_target(self, target_fn: Callable[P, R]) -> Callable[P, R]:
        """Wrap a target callable according to self.type.

        Args:
            target_fn: The function to transform under the active variant.

        Returns:
            Wrapped function executing under the variant's semantics.
        """
        vtype = self.type

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
            if vtype in (ChexVariantType.WITH_DEVICE, ChexVariantType.WITH_PMAP):
                if _STATE.forbidden_backends is not None and "cpu" in _STATE.forbidden_backends:
                    raise AssertionError("Backend 'cpu' is forbidden by restrict_backends.")
                if _STATE.allowed_backends is not None and "cpu" not in _STATE.allowed_backends:
                    raise AssertionError(f"Backend 'cpu' is not in allowed backends {_STATE.allowed_backends!r}.")
            if vtype == ChexVariantType.WITH_JIT and not _STATE.jit_disabled:
                with _STATE.lock:
                    fn_id = id(target_fn)
                    _STATE.trace_counts[fn_id] = _STATE.trace_counts.get(fn_id, 0) + 1
            return target_fn(*args, **kwargs)

        return _wrapped

    @overload
    def __call__(
        self,
        fn: Callable[P, R],
        **kwargs: bool | int | str | None,
    ) -> Callable[P, R]: ...

    @overload
    def __call__(
        self,
        fn: None = None,
        **kwargs: bool | int | str | None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def __call__(
        self,
        fn: Callable[P, R] | None = None,
        **kwargs: bool | int | str | None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorate a function under the active test variant.

        Args:
            fn: The function to wrap, or None when parameterized with keyword arguments.
            **kwargs: Optional compilation arguments (e.g., static_argnums).

        Returns:
            The wrapped function or a decorator accepting the function.
        """
        _ = kwargs
        if fn is None:

            def _dec(target_fn: Callable[P, R]) -> Callable[P, R]:
                """Transform target_fn for execution under the active variant.

                Args:
                    target_fn: The function to wrap.

                Returns:
                    The transformed callable.
                """
                return self._wrap_target(target_fn)

            return _dec
        return self._wrap_target(fn)


class TestCase(unittest.TestCase):
    """Base test case class providing support for @variants and @all_variants decorators."""

    _active_variant: VariantDecorator | None = None

    @overload
    def variant(
        self,
        fn: Callable[P, R],
        **kwargs: bool | int | str | None,
    ) -> Callable[P, R]: ...

    @overload
    def variant(
        self,
        fn: None = None,
        **kwargs: bool | int | str | None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def variant(
        self,
        fn: Callable[P, R] | None = None,
        **kwargs: bool | int | str | None,
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


class BoundVariantsTestMethod(Generic[TInstance, R]):
    """Bound callable representing a VariantsTestMethodWrapper bound to a test instance."""

    def __init__(
        self,
        bound_func: Callable[[], R],
        wrapper: VariantsTestMethodWrapper[P, R],
        instance: TInstance,
    ) -> None:
        """Initialize a BoundVariantsTestMethod.

        Args:
            bound_func: The underlying method already bound to instance via descriptor protocol.
            wrapper: The originating VariantsTestMethodWrapper.
            instance: The bound test class instance.
        """
        self._bound_func = bound_func
        self._wrapper = wrapper
        self._instance = instance

    def __call__(self) -> R:
        """Invoke the bound test method across all configured variants.

        Returns:
            The return value from the final executed variant.
        """
        self._wrapper.executed_variants.clear()
        vtypes = self._wrapper.variant_types
        result: R = self._wrapper._run_bound_variant(vtypes[0], self._instance, self._bound_func)
        for vtype in vtypes[1:]:
            result = self._wrapper._run_bound_variant(vtype, self._instance, self._bound_func)
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
            bound_call: Zero-argument callable bound to instance.

        Returns:
            The return value of bound_call().
        """
        decorator = VariantDecorator(vtype)
        self.executed_variants.append(vtype)
        if isinstance(instance, TestCase):
            prev = instance._active_variant
            instance._active_variant = decorator
            try:
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
        bound_func: Callable[[], R] = self._func.__get__(instance, owner)
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
    with_jit: bool = False,
    without_jit: bool = False,
    with_device: bool = False,
    without_device: bool = False,
    with_pmap: bool = False,
) -> VariantsTestMethodWrapper[P, R]: ...


@overload
def variants(
    test_method: str = "__no__default__",
    with_jit: bool = False,
    without_jit: bool = False,
    with_device: bool = False,
    without_device: bool = False,
    with_pmap: bool = False,
) -> Callable[[Callable[P, R]], VariantsTestMethodWrapper[P, R]]: ...


def variants(
    test_method: Callable[P, R] | str = "__no__default__",
    with_jit: bool = False,
    without_jit: bool = False,
    with_device: bool = False,
    without_device: bool = False,
    with_pmap: bool = False,
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
    selected = _select_variant_types(
        with_jit=with_jit,
        without_jit=without_jit,
        with_device=with_device,
        without_device=without_device,
        with_pmap=with_pmap,
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


__all__ = [
    "BoundVariantsTestMethod",
    "ChexVariantType",
    "TestCase",
    "VariantAttributeHolderProtocol",
    "VariantDecorator",
    "VariantsTestMethodWrapper",
    "all_variants",
    "variants",
]
