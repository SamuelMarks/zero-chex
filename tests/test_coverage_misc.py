import pytest
import zero_chex as chex


def test_if_args_not_none():
    called = False

    def func(a, b):
        nonlocal called
        called = True

    chex.if_args_not_none(func, 1, 2)
    assert called

    called = False
    chex.if_args_not_none(func, 1, None)
    assert not called


def test_warn_only_n_pos_args_in_future():
    @chex.warn_only_n_pos_args_in_future(n=1)
    def func(a, b=2):
        return a + b

    # Actually raising the warning is tested in warnings_test.py. We just cover the decorator logic.
    assert func(1, 2) == 3


def test_dataclass_methods():
    @chex.dataclass
    class MyDC:
        x: int

    # Try using mappable logic
    @chex.mappable_dataclass
    @chex.dataclass(mappable_dataclass=False)
    class MapDC:
        x: int

    obj2 = MapDC(x=2)
    assert obj2["x"] == 2


def test_fake_pmap():
    with chex.fake_pmap():
        pass


def test_fake_jit():
    with chex.fake_jit():
        pass


def test_if_args_not_none_no_func():
    with pytest.raises(TypeError):
        chex.if_args_not_none()


def test_warn_only_n_pos_args_in_future_edge():
    wrapper = chex.warn_only_n_pos_args_in_future()

    @wrapper
    def func(a):
        return a

    assert func(1) == 1


def test_fake_pmap_and_jit():
    with chex.fake_pmap_and_jit():
        pass


def test_create_deprecated_function_alias():
    def old_func():
        return 1

    new_func = chex.create_deprecated_function_alias(old_func, "new_func", "old_func")
    assert new_func() == 1


def test_clear_trace_counter():
    chex.clear_trace_counter()


def test_assert_max_traces():
    def f():
        pass

    chex.assert_max_traces(f)()


def test_block_until_chexify_assertions_complete():
    chex.block_until_chexify_assertions_complete()


def test_restrict_backends():
    @chex.restrict_backends(allowed=["cpu"])
    def foo():
        pass

    foo()


def test_set_n_cpu_devices():
    chex.set_n_cpu_devices(2)


def test_assert_numerical_grads():
    chex.assert_numerical_grads(lambda x: x, (1,), 1)


def test_disable_enable_asserts():
    chex.disable_asserts()
    chex.enable_asserts()


def test_params_product():
    list(chex.params_product([1], [2]))


def test_if_args_not_none_with_kwargs():
    # Calling the internal version since the public wrapper doesn't work out of the box with the mock.
    flag = False

    def inner(*args, **kwargs):
        nonlocal flag
        flag = True

    chex._src.misc.if_args_not_none(inner, 1, 2, c=3)
    assert flag

    flag = False
    chex._src.misc.if_args_not_none(inner, 1, None, c=3)
    assert not flag


def test_assert_max_traces_wrapper():
    @chex.assert_max_traces(n=1)
    def func():
        pass

    func()
    assert getattr(func, "__wrapped__", None) is not None


def test_chexify_wait_checks():
    # cover line 64
    @chex.chexify
    def func():
        pass

    func()
    func.wait_checks()


def test_fake_pmap_impl():
    chex._src.misc._fake_pmap_impl(lambda: None)


def test_mappable_dataclass_getitem_len():
    # cover lines 149, 152
    @chex.mappable_dataclass
    @chex.dataclass(mappable_dataclass=False)
    class MyMap:
        x: int

    obj = MyMap(x=1)
    assert obj["x"] == 1
    assert len(obj) == 1
