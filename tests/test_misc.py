"""Tests for misc."""

from typing import Any
import pytest
import dataclasses
import zero_jax as jax
import zero_jax.numpy as jnp

from zero_chex import (
    assert_max_traces,
    block_until_chexify_assertions_complete,
    chexify,
    clear_trace_counter,
    create_deprecated_function_alias,
    dataclass,
    disable_asserts,
    enable_asserts,
    fake_jit,
    fake_pmap,
    fake_pmap_and_jit,
    if_args_not_none,
    mappable_dataclass,
    params_product,
    restrict_backends,
    set_n_cpu_devices,
    warn_deprecated_function,
    warn_only_n_pos_args_in_future,
    with_jittable_assertions,
)


def test_assert_max_traces():
    clear_trace_counter()

    @assert_max_traces(n=1)
    def my_fn(x: Any) -> Any:
        return x

    my_fn(jnp.array(1))
    from zero_chex._src.misc import _TRACE_COUNTER

    _TRACE_COUNTER[hash(my_fn.__wrapped__)] = 2
    with pytest.raises(AssertionError):
        my_fn(jnp.array(1))

    from zero_chex._src.misc import _TRACE_COUNTER

    _TRACE_COUNTER[hash(my_fn.__wrapped__)] = 2
    with pytest.raises(AssertionError):
        my_fn(jnp.array(2))

    @assert_max_traces(1)
    def my_fn2():
        pass  # pragma: no cover


def test_assert_numerical_grads():
    def f(x):
        return x**2  # pragma: no cover

    # m()


def test_chexify():
    @chexify(async_check=False)
    def my_fn(x):
        return x

    assert my_fn(1) == 1

    @chexify(async_check=True)
    def my_fn2(x):
        return x

    assert my_fn2(1) == 1
    my_fn2.wait_checks()

    block_until_chexify_assertions_complete()

    def my_fn3(x):
        return x

    with_jittable_assertions(my_fn3)(1)


def test_create_deprecated_function_alias():
    def f():
        return 1

    g = create_deprecated_function_alias(f, "g", "f")
    with pytest.warns(DeprecationWarning):
        assert g() == 1


def test_dataclass():
    @dataclass
    class A:
        x: int

    assert A(x=1).x == 1

    @dataclass(mappable_dataclass=False)
    class B:
        x: int

    assert B(x=1).x == 1


def test_disable_enable_asserts():
    disable_asserts()
    enable_asserts()


def test_fake_jit_pmap():
    with fake_jit():
        pass
    with fake_pmap():
        assert jax.pmap(lambda x: x)(jnp.array([1])) == jnp.array([1])
    with fake_pmap_and_jit():
        pass

    stack = fake_jit()
    stack.start()
    stack.stop()


def test_if_args_not_none():
    called = []

    def f(x):
        called.append(x)

    if_args_not_none(f, 1)
    assert called == [1]
    if_args_not_none(f, None)
    assert called == [1]


def test_tree_util():
    from zero_chex._src.tree_util import (
        tree_map,
        tree_leaves,
        tree_all,
        tree_flatten_with_path,
        tree_structure,
    )

    t = {"a": [1, (2, 3)], "b": 4}
    assert tree_leaves(t) == [1, 2, 3, 4]
    assert tree_map(lambda x: x * 2, t) == {"a": [2, (4, 6)], "b": 8}
    assert tree_all(t)
    assert len(tree_flatten_with_path(t)[0]) == 4
    assert tree_structure(t) == {
        "dict": {"a": ["list", ["*", ("tuple", ["*", "*"])]], "b": "*"}
    }


def test_mappable_dataclass():
    @mappable_dataclass
    @dataclasses.dataclass
    class A:
        x: int

    a = A(x=1)
    assert a["x"] == 1
    assert len(a) == 1
    assert list(a) == ["x"]
    assert list(a.keys()) == ["x"]
    assert list(a.values()) == [1]
    assert list(a.items()) == [("x", 1)]

    with pytest.raises(ValueError):
        A(1)
    with pytest.raises(ValueError):
        A(y=1)

    with pytest.raises(ValueError, match="Expected dataclass"):
        mappable_dataclass(1)


def test_params_product():
    res = params_product([("a", 1)], [("b", 2)], named=True)
    assert res == [("a_b", 1, 2)]
    res = params_product([(1,)], [(2,)], named=False)
    assert res == [(1, 2)]


def test_restrict_backends():
    with restrict_backends(allowed=["cpu"]):
        pass
    with pytest.raises(ValueError):
        with restrict_backends():
            pass  # pragma: no cover
    with pytest.raises(ValueError):
        with restrict_backends(allowed=["cpu"], forbidden=["cpu"]):
            pass  # pragma: no cover


def test_set_n_cpu_devices():
    set_n_cpu_devices(1)


def test_warn_deprecated_function():
    @warn_deprecated_function
    def f():
        return 1

    with pytest.warns(DeprecationWarning):
        f()

    @warn_deprecated_function(replacement="g")
    def h():
        return 1

    with pytest.warns(DeprecationWarning):
        h()


def test_warn_only_n_pos_args_in_future():
    @warn_only_n_pos_args_in_future(n=1)
    def f(a, b=2):
        return a + b

    assert f(1) == 3
    with pytest.warns(DeprecationWarning):
        assert f(1, 2) == 3
