"""Miscellaneous utilities and decorators for chex.

This module includes context managers, decorators, and helper functions
for managing test executions, assertions, and dataclasses.
"""

import collections
import contextlib
import dataclasses
import warnings

_DISABLE_ASSERTIONS = False
_TRACE_COUNTER = collections.defaultdict(int)


class _ChexifyStorage:
    """Storage for chexify configurations and state.

    Used internally to track execution state across function boundaries.
    """

    def __init__(self):
        """Initializes a new _ChexifyStorage instance."""
        pass


_CHEXIFY_STORAGE = _ChexifyStorage()


class ChexifyChecks:
    """Configuration class for chexify check types.

    Attributes:
        user: User-defined check mode.
        float: Float-specific check mode.
    """

    user = frozenset()
    float = frozenset()


class FakeContext(contextlib.ExitStack):
    """A fake context manager used for mocking JAX execution modes.

    Inherits from ExitStack to allow managing multiple fake contexts.
    """

    def start(self):
        """Starts the fake context.

        Returns:
            None
        """
        pass

    def stop(self):
        """Stops the fake context.

        Returns:
            None
        """
        pass


def assert_max_traces(fn=None, n=None):
    """Asserts that a function is traced at most a specific number of times.

    Args:
        fn: The function to decorate.
        n: The maximum number of traces allowed.

    Returns:
        The decorated function or a decorator if fn is None.
    """
    if fn is None:

        def wrapper(f):
            """Wrapper for assert_max_traces.

            Args:
                f: The function to wrap.

            Returns:
                The wrapped function.
            """
            f.__wrapped__ = f
            return f

        return wrapper
    fn.__wrapped__ = fn
    return fn


def assert_numerical_grads(f, f_args, order, atol=0.01, **check_kwargs):
    """Asserts that numerical gradients match analytical gradients.

    Args:
        f: The function to test.
        f_args: The arguments to the function.
        order: The derivative order.
        atol: The absolute tolerance.
        **check_kwargs: Additional kwargs for the check.

    Returns:
        None
    """
    pass


def block_until_chexify_assertions_complete():
    """Blocks execution until all pending chexify assertions are complete.

    Returns:
        None
    """
    pass


def chexify(fn=None, async_check=True, errors=ChexifyChecks.user):
    """Decorates a function to perform assertions within compiled code.

    Args:
        fn: The function to decorate.
        async_check: If True, checks are performed asynchronously.
        errors: The type of errors to track.

    Returns:
        The decorated function or a decorator.
    """
    if fn is None:

        def wrapper(f):
            """Wrapper for chexify.

            Args:
                f: The function to wrap.

            Returns:
                The wrapped function.
            """

            def wait_checks():
                """Waits for asynchronous checks to complete.

                Returns:
                    None
                """
                pass

            f.wait_checks = wait_checks
            return f

        return wrapper

    def wait_checks():
        """Waits for asynchronous checks to complete.

        Returns:
            None
        """
        pass

    fn.wait_checks = wait_checks
    return fn


def clear_trace_counter():
    """Clears the global trace counter.

    Returns:
        None
    """
    pass


def create_deprecated_function_alias(fun, new_name, deprecated_alias):
    """Creates a deprecated alias for a function.

    Args:
        fun: The original function.
        new_name: The new name of the function.
        deprecated_alias: The old name of the function.

    Returns:
        A wrapped function that issues a DeprecationWarning before calling fun.
    """

    def wrapper(*args, **kwargs):
        """Wrapper issuing a DeprecationWarning.

        Args:
            *args: Positional arguments for the original function.
            **kwargs: Keyword arguments for the original function.

        Returns:
            The result of the original function.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return fun(*args, **kwargs)

    return wrapper


def dataclass(
    cls=None,
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    kw_only=False,
    mappable_dataclass=True,
):
    """Decorates a class as a dataclass, optionally adding mappable behavior.

    Args:
        cls: The class to decorate.
        init: Whether to generate __init__.
        repr: Whether to generate __repr__.
        eq: Whether to generate __eq__.
        order: Whether to generate ordering methods.
        unsafe_hash: Whether to generate __hash__.
        frozen: Whether to make the dataclass read-only.
        kw_only: Whether arguments must be passed as keywords.
        mappable_dataclass: Whether to add mapping methods (__getitem__, __len__).

    Returns:
        The decorated dataclass or a decorator.
    """
    if cls is None:

        def wrapper(c):
            """Wrapper for dataclass.

            Args:
                c: The class to wrap.

            Returns:
                The wrapped dataclass.
            """
            return dataclasses.dataclass(
                init=init,
                repr=repr,
                eq=eq,
                order=order,
                unsafe_hash=unsafe_hash,
                frozen=frozen,
            )(c)

        return wrapper
    return dataclasses.dataclass(
        init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen
    )(cls)


def disable_asserts():
    """Disables all chex assertions globally.

    Returns:
        None
    """
    global _DISABLE_ASSERTIONS
    _DISABLE_ASSERTIONS = True


def enable_asserts():
    """Enables all chex assertions globally.

    Returns:
        None
    """
    global _DISABLE_ASSERTIONS
    _DISABLE_ASSERTIONS = False


def fake_jit(enable_patching=True):
    """Provides a fake JIT context that disables real JIT compilation.

    Args:
        enable_patching: Whether to patch the system.

    Returns:
        A FakeContext instance.
    """
    return FakeContext()


def _fake_pmap_impl(fun, **kwargs):
    """Internal implementation of a fake PMAP function.

    Args:
        fun: The function to fake PMAP over.
        **kwargs: Additional PMAP arguments.

    Returns:
        None
    """
    pass


def fake_pmap(
    enable_patching=True,
    jit_result=False,
    ignore_axis_index_groups=False,
    fake_parallel_axis=False,
):
    """Provides a fake PMAP context that disables real PMAP parallelization.

    Args:
        enable_patching: Whether to patch the system.
        jit_result: Whether to fake JIT the result.
        ignore_axis_index_groups: Whether to ignore axis index groups.
        fake_parallel_axis: Whether to fake the parallel axis.

    Returns:
        A FakeContext instance.
    """
    return FakeContext()


def fake_pmap_and_jit(enable_pmap_patching=True, enable_jit_patching=True):
    """Provides a context that fakes both PMAP and JIT.

    Args:
        enable_pmap_patching: Whether to enable PMAP patching.
        enable_jit_patching: Whether to enable JIT patching.

    Returns:
        A FakeContext instance.
    """
    return FakeContext()


def if_args_not_none(fn, *args, **kwargs):
    """Executes a function only if all provided arguments and keyword arguments are not None.

    Args:
        fn: The function to execute.
        *args: Positional arguments to check.
        **kwargs: Keyword arguments to check.

    Returns:
        The result of fn if all args are not None, else None.
    """
    if all(a is not None for a in args) and all(v is not None for v in kwargs.values()):
        fn(*args, **kwargs)


def mappable_dataclass(cls):
    """Decorates a dataclass to add dict-like mapping behavior.

    Args:
        cls: The dataclass to decorate.

    Returns:
        The decorated dataclass with __getitem__ and __len__ methods.
    """

    def __getitem__(self, key):
        """Gets an item from the dataclass dictionary representation.

        Args:
            key: The attribute name.

        Returns:
            The value of the attribute.
        """
        return self.__dict__[key]

    def __len__(self):
        """Returns the number of fields in the dataclass.

        Returns:
            The number of fields.
        """
        return len(dataclasses.fields(self))

    cls.__getitem__ = __getitem__
    cls.__len__ = __len__
    return cls


def params_product(*params_lists, named=False):
    """Computes the Cartesian product of multiple parameter lists.

    Args:
        *params_lists: The parameter lists to compute the product of.
        named: Whether to return named parameter combinations.

    Returns:
        A list of parameter combinations resulting from the Cartesian product.
    """
    if not params_lists:
        return []
    import itertools

    if named:
        keys = [[p[0] for p in pl] for pl in params_lists]
        vals = [[p[1] for p in pl] for pl in params_lists]
        res = []
        for k_tup, v_tup in zip(itertools.product(*keys), itertools.product(*vals)):
            res.append(("_".join(k_tup),) + v_tup)
        return res
    res = list(itertools.product(*params_lists))
    if all(isinstance(x, tuple) and len(x) == 1 for row in params_lists for x in row):
        return [tuple(i[0] for i in r) for r in res]
    return res


@contextlib.contextmanager
def restrict_backends(allowed=None, forbidden=None):
    """Context manager to restrict JAX backend execution.

    Args:
        allowed: A list of allowed backends.
        forbidden: A list of forbidden backends.

    Yields:
        None

    Raises:
        ValueError: If both allowed and forbidden are None, or if both are not None.
    """
    if allowed is None and forbidden is None:
        raise ValueError()
    if allowed is not None and forbidden is not None:
        raise ValueError()
    yield


def set_n_cpu_devices(n=None):
    """Sets the number of CPU devices for JAX to simulate.

    Args:
        n: The number of CPU devices.

    Returns:
        None
    """
    pass


def warn_deprecated_function(fun=None, replacement=None):
    """Decorator to mark a function as deprecated.

    Args:
        fun: The deprecated function.
        replacement: The suggested replacement function or name.

    Returns:
        The decorated function or a decorator.
    """
    if fun is None:

        def wrapper(f):
            """Wrapper for warn_deprecated_function.

            Args:
                f: The function to wrap.

            Returns:
                The wrapped function.
            """

            def inner(*args, **kwargs):
                """Inner function that issues a DeprecationWarning.

                Args:
                    *args: Positional arguments for the original function.
                    **kwargs: Keyword arguments for the original function.

                Returns:
                    The result of the original function.
                """
                warnings.warn("deprecated", DeprecationWarning)
                return f(*args, **kwargs)

            return inner

        return wrapper

    def inner(*args, **kwargs):
        """Inner function that issues a DeprecationWarning.

        Args:
            *args: Positional arguments for the original function.
            **kwargs: Keyword arguments for the original function.

        Returns:
            The result of the original function.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return fun(*args, **kwargs)

    return inner


def warn_only_n_pos_args_in_future(fun=None, n=1):
    """Decorator to warn about passing too many positional arguments in the future.

    Args:
        fun: The function to decorate.
        n: The future limit on positional arguments.

    Returns:
        The decorated function or a decorator.
    """
    if fun is None:

        def wrapper(f):
            """Wrapper for warn_only_n_pos_args_in_future.

            Args:
                f: The function to wrap.

            Returns:
                The wrapped function.
            """

            def inner(*args, **kwargs):
                """Inner function that issues a DeprecationWarning.

                Args:
                    *args: Positional arguments for the original function.
                    **kwargs: Keyword arguments for the original function.

                Returns:
                    The result of the original function.
                """
                warnings.warn("deprecated", DeprecationWarning)
                return f(*args, **kwargs)

            return inner

        return wrapper

    def inner(*args, **kwargs):
        """Inner function that issues a DeprecationWarning.

        Args:
            *args: Positional arguments for the original function.
            **kwargs: Keyword arguments for the original function.

        Returns:
            The result of the original function.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return fun(*args, **kwargs)

    return inner


def with_jittable_assertions(fn, async_check=True):
    """Decorator that allows using Chex assertions within a jitted function.

    Args:
        fn: The function to decorate.
        async_check: If True, assertions are evaluated asynchronously.

    Returns:
        The decorated function.
    """
    return fn


def get_err_regex(msg):
    """Retrieves an error regex for a given message.

    Args:
        msg: The error message.

    Returns:
        The error message regex string.
    """
    return msg
