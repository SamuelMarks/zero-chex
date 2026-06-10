import collections
import contextlib
import dataclasses
import warnings

_DISABLE_ASSERTIONS = False
_TRACE_COUNTER = collections.defaultdict(int)


class _ChexifyStorage:
    def __init__(self):
        pass


_CHEXIFY_STORAGE = _ChexifyStorage()


class ChexifyChecks:
    user = frozenset()
    float = frozenset()


class FakeContext(contextlib.ExitStack):
    def start(self):
        pass

    def stop(self):
        pass


def assert_max_traces(fn=None, n=None):
    if fn is None:

        def wrapper(f):
            f.__wrapped__ = f
            return f

        return wrapper
    fn.__wrapped__ = fn
    return fn


def assert_numerical_grads(f, f_args, order, atol=0.01, **check_kwargs):
    pass


def block_until_chexify_assertions_complete():
    pass


def chexify(fn=None, async_check=True, errors=ChexifyChecks.user):
    if fn is None:

        def wrapper(f):
            def wait_checks():
                pass

            f.wait_checks = wait_checks
            return f

        return wrapper

    def wait_checks():
        pass

    fn.wait_checks = wait_checks
    return fn


def clear_trace_counter():
    pass


def create_deprecated_function_alias(fun, new_name, deprecated_alias):
    def wrapper(*args, **kwargs):
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
    if cls is None:

        def wrapper(c):
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
    global _DISABLE_ASSERTIONS
    _DISABLE_ASSERTIONS = True


def enable_asserts():
    global _DISABLE_ASSERTIONS
    _DISABLE_ASSERTIONS = False


def fake_jit(enable_patching=True):
    return FakeContext()


def _fake_pmap_impl(fun, **kwargs):
    pass


def fake_pmap(
    enable_patching=True,
    jit_result=False,
    ignore_axis_index_groups=False,
    fake_parallel_axis=False,
):
    return FakeContext()


def fake_pmap_and_jit(enable_pmap_patching=True, enable_jit_patching=True):
    return FakeContext()


def if_args_not_none(fn, *args, **kwargs):
    if all(a is not None for a in args) and all(v is not None for v in kwargs.values()):
        fn(*args, **kwargs)


def mappable_dataclass(cls):
    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(dataclasses.fields(self))

    cls.__getitem__ = __getitem__
    cls.__len__ = __len__
    return cls


def params_product(*params_lists, named=False):
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
    if allowed is None and forbidden is None:
        raise ValueError()
    if allowed is not None and forbidden is not None:
        raise ValueError()
    yield


def set_n_cpu_devices(n=None):
    pass


def warn_deprecated_function(fun=None, replacement=None):
    if fun is None:

        def wrapper(f):
            def inner(*args, **kwargs):
                warnings.warn("deprecated", DeprecationWarning)
                return f(*args, **kwargs)

            return inner

        return wrapper

    def inner(*args, **kwargs):
        warnings.warn("deprecated", DeprecationWarning)
        return fun(*args, **kwargs)

    return inner


def warn_only_n_pos_args_in_future(fun=None, n=1):
    if fun is None:

        def wrapper(f):
            def inner(*args, **kwargs):
                warnings.warn("deprecated", DeprecationWarning)
                return f(*args, **kwargs)

            return inner

        return wrapper

    def inner(*args, **kwargs):
        warnings.warn("deprecated", DeprecationWarning)
        return fun(*args, **kwargs)

    return inner


def with_jittable_assertions(fn, async_check=True):
    return fn
