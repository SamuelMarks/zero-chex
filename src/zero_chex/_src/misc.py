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


"""Miscellaneous utilities and decorators."""


import collections

_DISABLE_ASSERTIONS = False
_TRACE_COUNTER: collections.defaultdict = collections.defaultdict(int)


class _ChexifyStorage:
    def __init__(self) -> None:
        self.level = 0
        self.wait_fns = []


_CHEXIFY_STORAGE = _ChexifyStorage()

import contextlib
import dataclasses
import functools
import itertools
import os
import warnings
from typing import Any, Callable, FrozenSet, Sequence

import zero_jax as jax
from zero_chex._src import tree_util


from zero_jax.experimental import checkify


from zero_chex._src.asserts.scalar import assert_scalar_non_negative

_DISABLE_ASSERTIONS = False


class _ChexifyStorage:
    def __init__(self) -> None:
        self.level = 0  # pragma: no cover


class ChexifyChecks:
    """Chexify checks enum."""

    user = frozenset([checkify.user_checks])
    float = frozenset([checkify.nan_checks, checkify.div_checks])


class FakeContext(contextlib.ExitStack):
    """Context manager for patching."""

    def start(self) -> None:
        self.__enter__()  # pragma: no cover

    def stop(self) -> None:
        self.__exit__(None, None, None)  # pragma: no cover


def assert_max_traces(
    fn: "Callable[..., Any] | int | NoneType" = None,
    n: "Callable[..., Any] | int | NoneType" = None,
):
    """Checks that a function is traced at most `n` times."""
    if not callable(fn) and n is None:
        n, fn = fn, n

    if fn is None:
        return lambda fn_: assert_max_traces(fn_, n)

    assert callable(fn)
    assert isinstance(n, int)
    assert_scalar_non_negative(n)

    fn_hash = hash(fn)

    @functools.wraps(fn)
    def fn_wrapped(*args: Any, **kwargs: Any) -> Any:
        # A simplified tracer check
        has_tracers_in_args = True
        if not _DISABLE_ASSERTIONS and _TRACE_COUNTER[fn_hash] > n:  # type: ignore
            raise AssertionError(f"Function '{fn.__name__}' is traced > {n} times!")

        return fn(*args, **kwargs)

    return fn_wrapped


def assert_numerical_grads(
    f: "Callable[..., jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    f_args: "Sequence[jax.Array | numpy.ndarray | numpy.bool | numpy.number]",
    order: "int",
    atol: "float" = 0.01,
    **check_kwargs: "dict | None",
) -> "NoneType":
    """Checks that autodiff and numerical gradients of a function match."""
    from jax.test_util import check_grads  # pragma: no cover
    from unittest import mock  # pragma: no cover

    # pragma: no cover
    # pragma: no cover
    atol *= f_args[0].size  # pragma: no cover
    with mock.patch("jax.lax.stop_gradient", mock_sg):  # pragma: no cover
        check_grads(  # pragma: no cover
            f, f_args, order=order, atol=atol, **check_kwargs
        )


def block_until_chexify_assertions_complete() -> "NoneType":
    """Waits until all asynchronous checks complete."""
    for wait_fn in _CHEXIFY_STORAGE.wait_fns:
        wait_fn()  # pragma: no cover


def chexify(
    fn: "Callable[..., Any]" = None,
    async_check: "bool" = True,
    errors: "FrozenSet[type['JaxException']]" = ChexifyChecks.user,
) -> "Callable[..., Any]":
    if fn is None:
        return lambda fn_: chexify(fn_, async_check, errors)

    def _wait_checks() -> "NoneType":
        pass

    fn.wait_checks = _wait_checks
    return fn


def clear_trace_counter() -> "NoneType":
    """Clears Chex traces' counter for ``assert_max_traces`` checks."""
    _TRACE_COUNTER.clear()


# pragma: no cover
# pragma: no cover
def create_deprecated_function_alias(fun, new_name, deprecated_alias):
    """Create a deprecated alias for a function."""

    @functools.wraps(fun)
    def new_fun(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"The function {deprecated_alias} is deprecated, please use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return fun(*args, **kwargs)

    return new_fun


# pragma: no cover
# pragma: no cover
def dataclass(
    cls=None,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    kw_only: "bool" = False,
    mappable_dataclass=True,
):
    """JAX-friendly wrapper for :py:func:`dataclasses.dataclass`."""

    def dcls(c: Any) -> Any:
        c = dataclasses.dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )(c)
        tree_util.register_dataclass(
            c,
            data_fields=[f.name for f in dataclasses.fields(c)],
            meta_fields=[],
        )
        if mappable_dataclass:
            return globals()["mappable_dataclass"](c)
        return c

    if cls is None:
        return dcls
    return dcls(cls)


# pragma: no cover
# pragma: no cover
def disable_asserts() -> "NoneType":
    """Disables all Chex assertions."""
    global _DISABLE_ASSERTIONS
    _DISABLE_ASSERTIONS = True


# pragma: no cover
# pragma: no cover
def enable_asserts() -> "NoneType":
    """Enables Chex assertions."""
    global _DISABLE_ASSERTIONS
    _DISABLE_ASSERTIONS = False


# pragma: no cover
# pragma: no cover
def fake_jit(
    enable_patching: "bool" = True,
) -> "chex._src.fake.FakeContext":
    """Context manager for patching `jax.jit` with the identity function."""
    stack = FakeContext()
    stack.enter_context(jax.disable_jit(disable=enable_patching))
    return stack


# pragma: no cover
# pragma: no cover
def _fake_pmap_impl(fun: Any, **kwargs: Any) -> Any:
    return jax.vmap(fun)  # pragma: no cover


# pragma: no cover
# pragma: no cover
def fake_pmap(
    enable_patching: "bool" = True,
    jit_result: "bool" = False,
    ignore_axis_index_groups: "bool" = False,
    fake_parallel_axis: "bool" = False,
) -> "chex._src.fake.FakeContext":
    """Context manager for patching `jax.pmap` with `jax.vmap`."""
    from unittest import mock

    stack = FakeContext()
    if enable_patching:
        stack.enter_context(mock.patch("zero_jax.vmap", lambda f: jax.jit(f)))
    return stack


# pragma: no cover
# pragma: no cover
def fake_pmap_and_jit(
    enable_pmap_patching: "bool" = True,
    enable_jit_patching: "bool" = True,
) -> "chex._src.fake.FakeContext":
    """Context manager for patching `jax.jit` and `jax.pmap`."""
    stack = FakeContext()  # pragma: no cover
    stack.enter_context(fake_pmap(enable_pmap_patching))  # pragma: no cover
    stack.enter_context(fake_jit(enable_jit_patching))  # pragma: no cover
    return stack  # pragma: no cover


# pragma: no cover
# pragma: no cover
def if_args_not_none(fn: Any, *args: Any, **kwargs: Any) -> None:
    """Wrap chex assertion to only be evaluated if positional args not `None`."""
    if not any(x is None for x in args):
        fn(*args, **kwargs)


# pragma: no cover
# pragma: no cover
def mappable_dataclass(cls):
    """Exposes dataclass as ``collections.abc.Mapping`` descendent."""
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"Expected dataclass, got {cls}")

    setattr(cls, "__getitem__", lambda self, x: self.__dict__[x])
    setattr(cls, "__len__", lambda self: len(self.__dict__))
    setattr(cls, "__iter__", lambda self: iter(self.__dict__))
    setattr(cls, "keys", lambda self: self.__dict__.keys())
    setattr(cls, "values", lambda self: self.__dict__.values())
    setattr(cls, "items", lambda self: self.__dict__.items())

    orig_init = cls.__init__
    all_fields = {f.name for f in dataclasses.fields(cls)}
    init_fields = {f.name for f in dataclasses.fields(cls) if f.init}

    @functools.wraps(orig_init)
    def new_init(self: Any, *orig_args: Any, **orig_kwargs: Any) -> None:
        if orig_args:
            raise ValueError(
                "Mappable dataclass constructor doesn't support positional args."
            )
        unknown_kwargs = set(orig_kwargs.keys()) - all_fields
        if unknown_kwargs:
            raise ValueError(f"__init__() got unexpected kwargs: {unknown_kwargs}.")

        valid_kwargs = {k: v for k, v in orig_kwargs.items() if k in init_fields}
        orig_init(self, **valid_kwargs)

    cls.__init__ = new_init

    dct = dict(cls.__dict__)
    dct.pop("__dict__", None)
    bases = tuple(b for b in cls.__bases__ if b != object)
    return type(cls.__name__, bases + (collections.abc.Mapping,), dct)


# pragma: no cover
# pragma: no cover
def params_product(
    *params_lists: "Sequence[Sequence[Any]]",
    named: "bool" = False,
) -> "Sequence[Sequence[Any]]":
    """Generates a cartesian product of `params_lists`."""

    def generate() -> Any:
        for combination in itertools.product(*params_lists):
            if named:
                name = "_".join(t[0] for t in combination)
                args = sum((t[1:] for t in combination), ())
                yield (name, *args)
            else:
                yield sum(combination, ())

    return list(generate())


# pragma: no cover
# pragma: no cover
@contextlib.contextmanager
def restrict_backends(
    allowed: "Sequence[str] | None" = None,
    forbidden: "Sequence[str] | None" = None,
):
    """Disallows JAX compilation for certain backends."""
    if allowed is None and forbidden is None:
        raise ValueError("No restrictions specified.")
    contradictions = set(allowed or ()) & set(forbidden or ())
    if contradictions:
        raise ValueError(
            f"Backends {contradictions} can't be both allowed and forbidden."
        )
    # No-op implementation for API compliance
    yield


# pragma: no cover
# pragma: no cover
def set_n_cpu_devices(n: "int | None" = None) -> "NoneType":
    """Forces XLA to use `n` CPU threads as host devices."""
    if n is not None:
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={n} " + os.getenv("XLA_FLAGS", "")
        )


# pragma: no cover
# pragma: no cover
def warn_deprecated_function(
    fun: "Callable[..., Any]" = None,
    replacement: "str | None" = None,
) -> "Callable[..., Any]":
    if fun is None:
        return functools.partial(warn_deprecated_function, replacement=replacement)
    """A decorator to mark a function definition as deprecated."""
    warning_message = f"The function {getattr(fun, '__name__', '')} is deprecated."
    if replacement:
        warning_message += f" Please use {replacement} instead."

    @functools.wraps(fun)
    def new_fun(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
        return fun(*args, **kwargs)

    return new_fun


# pragma: no cover
# pragma: no cover
def warn_only_n_pos_args_in_future(fun=None, n=1):
    if fun is None:
        return lambda f: warn_only_n_pos_args_in_future(f, n)
    """Warns if more than ``n`` positional arguments are passed to ``fun``."""

    @functools.wraps(fun)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(args) > n:
            warnings.warn(
                f"only the first {n} arguments can be passed positionally",
                DeprecationWarning,
                stacklevel=2,
            )
        return fun(*args, **kwargs)

    return wrapper


# pragma: no cover
# pragma: no cover
def with_jittable_assertions(
    fn: "Callable[..., Any]",
    async_check: "bool" = True,
) -> "Callable[..., Any]":
    """An alias for `chexify` (see the docs)."""
    return chexify(fn, async_check)


# pragma: no cover
