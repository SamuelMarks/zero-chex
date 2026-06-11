"""Tests for misc."""

import pytest
import zero_jax as jax
import ml_switcheroo.ops as jnp

from zero_chex import (
    block_until_chexify_assertions_complete,
    chexify,
    create_deprecated_function_alias,
    dataclass,
    disable_asserts,
    enable_asserts,
    fake_jit,
    fake_pmap,
    fake_pmap_and_jit,
    if_args_not_none,
    params_product,
    restrict_backends,
    set_n_cpu_devices,
    warn_deprecated_function,
    warn_only_n_pos_args_in_future,
    with_jittable_assertions,
)


def test_assert_max_traces():
    pass


def test_assert_numerical_grads():
    def f(x):
        return x**2

    f(1)


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
        res = jax.pmap(lambda x: x)(jnp.ones((1,)))
        expected = jnp.ones((1,))
        res_data = (
            res._tensor.data
            if hasattr(res, "_tensor")
            else (res.data if hasattr(res, "data") else res)
        )
        expected_data = (
            expected._tensor.data
            if hasattr(expected, "_tensor")
            else (expected.data if hasattr(expected, "data") else expected)
        )
        assert __import__("numpy").array_equal(res_data, expected_data)
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
    pass


def test_mappable_dataclass():
    pass


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
            pass
    with pytest.raises(ValueError):
        with restrict_backends(allowed=["cpu"], forbidden=["cpu"]):
            pass


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


def test_misc_coverage():
    from zero_chex import warn_only_n_pos_args_in_future, get_err_regex

    def dummy_fn():
        pass

    wrapper = warn_only_n_pos_args_in_future(dummy_fn)
    import warnings

    with warnings.catch_warnings(record=True):
        wrapper()

    assert get_err_regex("test") == "test"
