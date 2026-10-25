"""Tests for dataclasses, trace counters, chexify, numerical grads, context managers, and deprecation utils."""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import warnings
from collections.abc import Sequence

import pytest
from ml_switcheroo_ir import DType

from zero_chex import (
    ChexifyChecks,
    TestCase,
    assert_max_traces,
    assert_numerical_grads,
    assert_scalar_positive,
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
    get_err_regex,
    if_args_not_none,
    ify,
    mappable_dataclass,
    params_product,
    register_dataclass_type_with_jax_tree_util,
    restrict_backends,
    set_n_cpu_devices,
    variants,
    warn_deprecated_function,
    warn_keyword_args_only_in_future,
    warn_only_n_pos_args_in_future,
    with_jittable_assertions,
)
from zero_chex._src.tree_util import _CUSTOM_PYTREE_NODES, tree_flatten, tree_unflatten
from zero_chex.dataclasses import (
    MappableDataclassMixin,
    _get_field_names,
    _get_field_values,
)
from zero_chex.mocks import (
    FakeContext,
    _abstract_arg_signature,
    _fake_pmap_impl,
    _invoke_differentiable,
    _scalar_from_numeric,
    _TaylorJet,
    are_asserts_enabled,
)
from zero_chex.types import MappableDataclassProtocol, Numeric, PyTree, PyTreeDef, ZeroArray


def test_assert_toggles_and_max_traces() -> None:
    """Test enable_asserts, disable_asserts, assert_max_traces, and clear_trace_counter."""
    assert are_asserts_enabled()
    disable_asserts()
    assert not are_asserts_enabled()
    enable_asserts()
    assert are_asserts_enabled()

    @assert_max_traces(n=2)
    def fn_limit(x: int) -> int:
        """Add one to x.

        Args:
            x: Input integer.

        Returns:
            Incremented integer.
        """
        return x + 1

    assert fn_limit(1) == 2
    assert fn_limit(2) == 3
    with pytest.raises(AssertionError, match="Function is traced > 2 times"):
        fn_limit(3)

    clear_trace_counter()
    assert fn_limit(4) == 5

    @assert_max_traces(1)
    def fn_pos_limit(x: int) -> int:
        """Positional integer limit decorator test.

        Args:
            x: Input integer.

        Returns:
            Incremented integer.
        """
        return x + 1

    assert fn_pos_limit(10) == 11
    with pytest.raises(AssertionError, match="Function is traced > 1 times"):
        fn_pos_limit(20)

    @assert_max_traces
    def fn_unbounded(x: int) -> int:
        """Return x unchanged.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    assert fn_unbounded(10) == 10


def test_chexify_and_with_jittable_assertions() -> None:
    """Test @chexify, @with_jittable_assertions, wait_checks, and runtime error checks."""

    @chexify
    def f1(x: int) -> int:
        """Double x after checking positivity.

        Args:
            x: Input integer.

        Returns:
            Doubled integer.
        """
        assert_scalar_positive(x)
        return x * 2

    @chexify(async_check=False, errors=ChexifyChecks.all)
    def f2(x: float) -> float:
        """Compute 10.0 / x.

        Args:
            x: Divisor float.

        Returns:
            Quotient float.
        """
        return 10.0 / x

    @chexify(async_check=True, errors=ChexifyChecks.all)
    def f_nan(flag: int) -> float:
        """Produce NaN, complex NaN, or index error based on flag.

        Args:
            flag: Selector integer.

        Returns:
            Float or complex value.
        """
        if flag == 1:
            return float("nan")
        if flag == 2:
            items = [1.0, 2.0]
            return items[99]
        return 1.0

    @chexify(async_check=True, errors=ChexifyChecks.nan)
    def f_complex_nan(bad: bool) -> complex:
        """Return complex with NaN imaginary part when bad is True.

        Args:
            bad: Whether to return complex NaN.

        Returns:
            Complex scalar.
        """
        return complex(1.0, float("nan")) if bad else complex(1.0, 2.0)

    @chexify(async_check=False, errors=ChexifyChecks.nan)
    def f_unhandled_errs(mode: int) -> float:
        """Raise errors not enabled in self._errors to test pass-through re-raising.

        Args:
            mode: Error selector.

        Returns:
            Float value.
        """
        if mode == 1:
            divisor = mode - 1
            return 1.0 / divisor
        if mode == 2:
            seq = [1.0]
            return seq[mode]
        if mode == 3:
            raise AssertionError("raw user assertion")
        return 1.0

    f3 = with_jittable_assertions(lambda x: x + 5, async_check=True)
    f4 = with_jittable_assertions(async_check=False)(lambda x: x + 10)
    f5 = ify(lambda x: x + 7, async_check=True)
    f6 = ify(async_check=False)(lambda x: x + 14)

    assert f1(3) == 6
    f1.wait_checks()
    with pytest.raises(AssertionError, match="must be positive"):
        f1(-1)
    with pytest.raises(AssertionError, match="must be positive"):
        f1.wait_checks()
    f1.wait_checks()

    assert f2(2.0) == 5.0
    f2.wait_checks()
    with pytest.raises(AssertionError, match="division by zero"):
        f2(0.0)
    f2.wait_checks()

    assert f_nan(0) == 1.0
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        f_nan(1)
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        f_nan.wait_checks()

    with pytest.raises(AssertionError, match="index out-of-bounds"):
        f_nan(2)
    with pytest.raises(AssertionError, match="index out-of-bounds"):
        f_nan.wait_checks()

    assert f_complex_nan(False) == 1.0 + 2.0j
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        f_complex_nan(True)
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        block_until_chexify_assertions_complete()
    block_until_chexify_assertions_complete()

    assert f_unhandled_errs(0) == 1.0
    with pytest.raises(ZeroDivisionError):
        f_unhandled_errs(1)
    with pytest.raises(IndexError):
        f_unhandled_errs(2)
    with pytest.raises(AssertionError, match="raw user assertion"):
        f_unhandled_errs(3)

    assert f3(3) == 8
    f3.wait_checks()
    assert f4(3) == 13
    assert f5(3) == 10
    f5.wait_checks()
    assert f6(3) == 17


def test_taylor_jet_and_assert_numerical_grads() -> None:
    """Test _TaylorJet arithmetic and assert_numerical_grads on scalar, array, PyTree, and higher-order cases."""
    j_var = _TaylorJet.variable(2.0, 3)
    j_zero = _TaylorJet.variable(2.0, 0)
    assert j_zero.order == 0
    assert j_var.derivative(-1) == 0.0
    assert j_var.derivative(99) == 0.0
    assert float(+j_var) == 2.0
    assert float(-j_var) == -2.0

    # Test jet arithmetic with different orders and scalar operands
    j_short = _TaylorJet.constant(3.0, 1)
    assert float(j_var + j_short) == 5.0
    assert float(3.0 + j_var) == 5.0
    assert float(j_var - 1.0) == 1.0
    assert float(10.0 - j_var) == 8.0
    assert float(3.0 * j_var) == 6.0
    assert (j_var**0).derivative(0) == 1.0
    assert (j_var**-1).derivative(1) == pytest.approx(-0.25)
    assert (j_var**1.5).derivative(1) == pytest.approx(1.5 * (2.0**0.5))
    assert (1.0 / j_var).derivative(1) == pytest.approx(-0.25)
    assert (j_short / j_var).derivative(0) == pytest.approx(1.5)
    assert math.isnan((j_var / 0.0).derivative(0))
    j_neg = _TaylorJet.variable(-2.0, 2)
    assert math.isnan((j_neg**1.5).derivative(0))

    assert _scalar_from_numeric(_TaylorJet.constant(5.0, 1)) == 5.0
    assert _scalar_from_numeric(3.0) == 3.0
    assert _scalar_from_numeric(2 + 0j) == 2.0
    assert _scalar_from_numeric(ZeroArray(4.0)) == 4.0
    assert _scalar_from_numeric(ZeroArray(4 + 0j)) == 4.0
    with pytest.raises(AssertionError, match="Expected real scalar"):
        _scalar_from_numeric(2 + 5j)
    with pytest.raises(AssertionError, match="Expected real scalar"):
        _scalar_from_numeric(ZeroArray(2 + 5j))

    class _DummyShapeOnly:
        """Dummy ShapeOnlyProtocol object for testing _scalar_from_numeric fallback."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return empty shape."""
            return ()

    dummy_shape: Numeric = _DummyShapeOnly()
    assert _scalar_from_numeric(dummy_shape) == 0.0
    assert _scalar_from_numeric(ZeroArray([1.0], shape=(0,))) == 0.0

    sig_str = _abstract_arg_signature(
        (ZeroArray(1.0), _DummyShapeOnly(), {"a": ZeroArray(2.0), "b": 3}, 5),
        {"k1": ZeroArray(1.0), "k2": "v"},
    )
    assert "arr:" in sig_str and "tree:" in sig_str

    def algebraic_poly(x: Numeric, y: Numeric) -> Numeric:
        """Compute x^3 + 2*y^2 directly using algebraic operators so _TaylorJet propagates.

        Args:
            x: First numeric input.
            y: Second numeric input.

        Returns:
            Polynomial evaluation.
        """
        if isinstance(x, (int, float, _TaylorJet)) and isinstance(y, (int, float, _TaylorJet)):
            return x**3 + 2.0 * (y**2)
        vx = _scalar_from_numeric(x)
        vy = _scalar_from_numeric(y)
        return vx * vx * vx + 2.0 * vy * vy

    def const_scalar_fn(x: Numeric) -> Numeric:
        """Return constant 5.0 regardless of x.

        Args:
            x: Input numeric value.

        Returns:
            Constant 5.0.
        """
        _ = x
        return 5.0

    def type_checking_fn(x: Numeric) -> Numeric:
        """Raise TypeError on _TaylorJet to exercise fallback derivative path.

        Args:
            x: Input numeric value.

        Returns:
            x^2 as float.
        """
        if isinstance(x, _TaylorJet):
            raise TypeError("Requires float")
        vx = _scalar_from_numeric(x)
        return vx * vx

    assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=1, atol=1e-3)
    assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=2, atol=1e-2)
    assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=3, atol=5e-2)
    assert_numerical_grads(const_scalar_fn, [2.0], order=1, atol=1e-4)
    assert_numerical_grads(type_checking_fn, [2.0], order=1, atol=1e-3)
    assert_numerical_grads(algebraic_poly, [ZeroArray([1.5, 2.5], shape=(2,)), 1.0], order=1, atol=1e-2)

    def pytree_fn(tree_arg: PyTree) -> PyTree:
        """Compute squared leaves for a dict PyTree argument.

        Args:
            tree_arg: Dictionary PyTree with numeric leaves.

        Returns:
            Dictionary with squared float values.
        """
        if isinstance(tree_arg, dict):
            a_val = _scalar_from_numeric(tree_arg["a"])
            b_val = _scalar_from_numeric(tree_arg["b"])
            return {"a": a_val * a_val, "b": 2.0 * b_val}
        return 0.0

    assert_numerical_grads(pytree_fn, [{"a": 3.0, "b": ZeroArray([1.0, 2.0], shape=(2,))}], order=1, atol=1e-2)
    assert _invoke_differentiable(pytree_fn, [{"a": 2.0, "b": 3.0}]) == {"a": 4.0, "b": 6.0}

    class _CorrectAnalyticalFn:
        """Callable with an exact analytical .grad() method."""

        def __call__(self, *args: Numeric) -> Numeric:
            """Compute x^2 + 3*y.

            Args:
                *args: Input numeric arguments.

            Returns:
                Computed scalar value.
            """
            x = _scalar_from_numeric(args[0])
            y = _scalar_from_numeric(args[1])
            return x * x + 3.0 * y

        def grad(self, *args: Numeric) -> Sequence[Numeric]:
            """Return exact gradient (2*x, 3.0).

            Args:
                *args: Input numeric arguments.

            Returns:
                Sequence of analytical gradients.
            """
            x = _scalar_from_numeric(args[0])
            return [2.0 * x, 3.0]

    class _WrongAnalyticalSingleFn:
        """Callable with an incorrect single-arg analytical .grad() method."""

        def __call__(self, *args: Numeric) -> Numeric:
            """Compute x^2.

            Args:
                *args: Input numeric arguments.

            Returns:
                Computed scalar value.
            """
            x = _scalar_from_numeric(args[0])
            return x * x

        def grad(self, *args: Numeric) -> Numeric:
            """Return wrong gradient 100.0 instead of 2*x.

            Args:
                *args: Input numeric arguments.

            Returns:
                Incorrect scalar gradient.
            """
            _ = args
            return 100.0

    class _NaNAnalyticalFn:
        """Callable whose analytical .grad() returns NaN."""

        def __call__(self, *args: Numeric) -> Numeric:
            """Compute x.

            Args:
                *args: Input numeric arguments.

            Returns:
                Computed scalar value.
            """
            return _scalar_from_numeric(args[0])

        def grad(self, *args: Numeric) -> Numeric:
            """Return NaN gradient.

            Args:
                *args: Input numeric arguments.

            Returns:
                NaN scalar gradient.
            """
            _ = args
            return float("nan")

    assert_numerical_grads(_CorrectAnalyticalFn(), [3.0, 4.0], order=1, atol=1e-3)
    with pytest.raises(AssertionError, match="Gradient mismatch"):
        assert_numerical_grads(_WrongAnalyticalSingleFn(), [3.0], order=1, atol=1e-3)
    with pytest.raises(AssertionError, match="Analytical gradient is non-finite"):
        assert_numerical_grads(_NaNAnalyticalFn(), [3.0], order=1, atol=1e-3)

    with pytest.raises(AssertionError, match="non-finite values at base"):
        assert_numerical_grads(lambda _x: float("nan"), [1.0])

    def _nan_on_perturbed(x: Numeric) -> Numeric:
        """Return 0.0 at x == 1.0 and NaN otherwise.

        Args:
            x: Input numeric value.

        Returns:
            0.0 at base point or NaN when perturbed.
        """
        v = _scalar_from_numeric(x)
        return 0.0 if v == 1.0 else float("nan")

    with pytest.raises(AssertionError, match="Numerical gradient is non-finite"):
        assert_numerical_grads(_nan_on_perturbed, [1.0])

    with pytest.raises(ValueError, match="order must be at least 1"):
        assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=0)

    with pytest.raises(AssertionError, match="Tolerance atol must be non-negative"):
        assert_numerical_grads(algebraic_poly, [2.0, 3.0], atol=-1.0)

    with pytest.raises(
        ValueError,
        match=r"\[Chex\] Assertion failed: bad atol -2\.0",
    ):
        assert_numerical_grads(
            algebraic_poly,
            [2.0, 3.0],
            atol=-2.0,
            custom_message="bad atol {}",
            custom_message_format_vars=(-2.0,),
            include_default_message=False,
            exception_type=ValueError,
        )
    with pytest.raises(AssertionError, match=r"Tolerance atol must be non-negative, got -1\.0\. \[extra\]"):
        assert_numerical_grads(
            algebraic_poly,
            [2.0, 3.0],
            atol=-1.0,
            custom_message="extra",
            include_default_message=True,
        )


def test_fake_contexts_and_backend_restrictions() -> None:
    """Test fake_jit, fake_pmap, fake_pmap_and_jit, restrict_backends, and set_n_cpu_devices."""
    ctx = FakeContext()
    assert not ctx.active
    ctx.start()
    ctx2 = FakeContext()
    ctx2.start()
    assert ctx2.active
    ctx2.stop()
    ctx.stop()

    @assert_max_traces(n=1)
    def traced_fn(x: int) -> int:
        """Increment x.

        Args:
            x: Input integer.

        Returns:
            x + 1.
        """
        return x + 1

    with fake_jit() as j_ctx:
        assert j_ctx.active is True
        assert traced_fn(1) == 2
        assert traced_fn(2) == 3
    assert j_ctx.active is False

    class PmapSampleTest(TestCase):
        """TestCase verifying fake_pmap and fake_pmap_and_jit with @variants."""

        @variants(with_pmap=True, with_jit=True)
        def run_step(self, val: int) -> int:
            """Execute step under active variant.

            Args:
                val: Input integer.

            Returns:
                val * 2.
            """

            def _double(v: int) -> int:
                """Double v.

                Args:
                    v: Input integer.

                Returns:
                    v * 2.
                """
                return v * 2

            fn = self.variant(_double)
            return fn(val)

    pmap_test = PmapSampleTest()
    with fake_pmap() as p_ctx:
        assert p_ctx.active is True
        assert pmap_test.run_step(4) == 8
    assert p_ctx.active is False

    with fake_pmap_and_jit() as pj_ctx:
        assert pj_ctx.active is True
        assert pmap_test.run_step(5) == 10
    assert pj_ctx.active is False

    def _id(x: int) -> int:
        """Identity helper.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    assert _fake_pmap_impl(_id)(5) == 5

    with restrict_backends(allowed=["cpu"]):
        pass
    with restrict_backends(forbidden=["tpu"]):
        pass
    with pytest.raises(ValueError, match="Must specify either"):
        with restrict_backends():
            pass
    with pytest.raises(ValueError, match="Cannot specify both"):
        with restrict_backends(allowed=["cpu"], forbidden=["gpu"]):
            pass

    set_n_cpu_devices(2)
    set_n_cpu_devices(None)
    with pytest.raises(ValueError, match="must be >= 1"):
        set_n_cpu_devices(0)


def test_dataclasses_and_pytree_registration() -> None:
    """Test @dataclass, @mappable_dataclass, kw_only, Mapping conformance, and PyTree registration."""

    @dataclass
    class Point(MappableDataclassMixin):
        """2D point dataclass using generated __init__."""

        x: int = 0
        y: int = 0

    p = Point(1, 2)
    assert isinstance(p, collections.abc.Mapping)
    assert isinstance(p, MappableDataclassProtocol)
    assert p["x"] == 1
    assert p["y"] == 2
    with pytest.raises(KeyError):
        _ = p["z"]
    assert list(p) == ["x", "y"]
    assert len(p) == 2
    assert tuple(p.keys()) == ("x", "y")
    assert tuple(p.values()) == (1, 2)
    assert tuple(p.items()) == (("x", 1), ("y", 2))
    assert dict(p) == {"x": 1, "y": 2}
    p2 = p.replace(x=10)
    assert p2.x == 10 and p2.y == 2
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        p.replace(unknown=99)
    tup = p.to_tuple()
    assert tup == ((1, 2), ("x", "y"))
    assert Point.from_tuple(tup) == p

    leaves, treedef = tree_flatten(p)
    assert leaves == [1, 2]
    assert tree_unflatten(treedef, [30, 40]) == Point(30, 40)

    # Verify nested dataclasses containing ZeroArray and list/dict fields are NOT corrupted by astuple
    @dataclass
    class ContainerPoint(MappableDataclassMixin):
        """Dataclass holding a ZeroArray and a list."""

        arr: ZeroArray
        seq: list[int]

    arr_leaf = ZeroArray([1.0, 2.0], shape=(2,), dtype=DType.float32)
    nested_p = ContainerPoint(arr_leaf, [10, 20])
    assert isinstance(nested_p["arr"], ZeroArray)
    assert isinstance(nested_p["seq"], list)
    nested_leaves, _ = tree_flatten(nested_p)
    assert len(nested_leaves) == 3
    assert isinstance(nested_leaves[0], ZeroArray)

    # Test kw_only=True with mappable_dataclass=True and mappable_dataclass=False
    @dataclass(kw_only=True)
    class KwPoint:
        """Keyword-only mappable dataclass."""

        x: int = 0
        y: int = 0

    kp = KwPoint(x=3, y=4)
    assert kp.x == 3 and kp.y == 4
    pos_pair: Sequence[int] = [3, 4]
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        KwPoint(*pos_pair)

    @dataclass(kw_only=True)
    class KwPreMappablePoint(MappableDataclassMixin):
        """Keyword-only dataclass already inheriting MappableDataclassMixin."""

        x: int = 0
        y: int = 0

    kpm = KwPreMappablePoint(x=5, y=6)
    assert kpm.x == 5 and kpm.y == 6
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        KwPreMappablePoint(*pos_pair)

    @dataclass(kw_only=True, mappable_dataclass=False)
    class KwNonMappablePoint:
        """Keyword-only non-mappable dataclass."""

        x: int = 0
        y: int = 0

    knp = KwNonMappablePoint(x=7, y=8)
    assert knp.x == 7 and knp.y == 8
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        KwNonMappablePoint(*pos_pair)

    # Test unflatten when custom_cls is not in _CUSTOM_PYTREE_NODES
    @dataclass(mappable_dataclass=True, registered_with_pytree=False)
    class UnregisteredPoint(MappableDataclassMixin):
        """Mappable dataclass not registered in _CUSTOM_PYTREE_NODES."""

        a: int = 0
        b: int = 0

    up = UnregisteredPoint(5, 6)
    up_leaves, up_def = tree_flatten(up)
    assert up_leaves == [5, 6]
    assert tree_unflatten(up_def, [7, 8]) == UnregisteredPoint(7, 8)

    fallback_def = PyTreeDef(node_type="unknown", children=(PyTreeDef(),))
    with pytest.raises(ValueError, match="Unknown PyTreeDef node_type"):
        tree_unflatten(fallback_def, [99])

    @dataclass(mappable_dataclass=False, registered_with_pytree=True)
    class NonMappablePoint:
        """Dataclass with mappable_dataclass=False."""

        x: int = 0
        y: int = 0

    nmp = NonMappablePoint(3, 4)
    assert nmp.x == 3 and nmp.y == 4
    mappable_dataclass(NonMappablePoint)
    register_dataclass_type_with_jax_tree_util(NonMappablePoint)

    @dataclass(mappable_dataclass=False, registered_with_pytree=False)
    class RawDataclassPoint:
        """Dataclass with both mappable_dataclass=False and registered_with_pytree=False."""

        x: int = 1
        y: int = 2

    raw_pt = RawDataclassPoint(9, 10)
    assert raw_pt.x == 9 and raw_pt.y == 10

    class PlainBox:
        """Plain class converted by mappable_dataclass and register_dataclass_type_with_jax_tree_util."""

        val: int = 0

    class AnotherPlainBox:
        """Another plain class converted directly by mappable_dataclass."""

        val: int = 0

    mappable_dataclass(AnotherPlainBox)
    reg_box = register_dataclass_type_with_jax_tree_util(PlainBox)
    mappable_dataclass(reg_box)
    assert issubclass(reg_box, MappableDataclassProtocol)
    cls_entry = _CUSTOM_PYTREE_NODES[reg_box]
    _, unflatten_fn = cls_entry
    rebuilt_box = unflatten_fn((), (42,))
    assert isinstance(rebuilt_box, MappableDataclassProtocol)
    assert dict(zip(rebuilt_box.__chex_mappable_fields__(), rebuilt_box.__chex_field_values__())) == {"val": 42}

    # Test MappableDataclassMixin fallbacks on a non-dataclass subclass
    class _BareMixinUser(MappableDataclassMixin):
        """Non-dataclass class inheriting directly from MappableDataclassMixin."""

        def __init__(self, **kwargs: PyTree) -> None:
            """Initialize bare mixin user.

            Args:
                **kwargs: Ignored keyword arguments.
            """
            _ = kwargs

    bare = _BareMixinUser()
    assert bare.__chex_mappable_fields__() == ()
    assert bare.__chex_field_values__() == ()
    assert _get_field_names(bare) == ()
    assert _get_field_values(bare) == ()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        bare.replace(bad=1)

    @dataclasses.dataclass
    class _StdDataclass:
        """Standard Python dataclass not inheriting from MappableDataclassMixin."""

        v: int = 5

    std_inst = _StdDataclass(5)
    assert MappableDataclassMixin.replace(std_inst, v=9).v == 9


def test_utils_params_product_and_deprecations() -> None:
    """Test params_product, if_args_not_none, get_err_regex, and deprecation decorators."""
    unnamed = params_product([1, 2], ["a", "b"])
    assert unnamed == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]

    unnamed_tuples = params_product([(1, 2)], [(3,)])
    assert unnamed_tuples == [(1, 2, 3)]

    named = params_product([("case1", 10), ("case2", 20)], [{"sub": (30, 40)}], named=True)
    assert len(named) == 2

    named_mapping = params_product({"k1": 1, "k2": (2, 3)}, [10], named=True)
    assert len(named_mapping) == 2

    def _add_opt(a: int | None, b: int | None) -> int:
        """Add two optional integers.

        Args:
            a: First integer.
            b: Second integer.

        Returns:
            Sum of a and b.
        """
        return (a or 0) + (b or 0)

    assert if_args_not_none(_add_opt, 2, 3) == 5
    assert if_args_not_none(_add_opt, None, 3) is None
    assert get_err_regex("shape (2, 3)") == r"shape\ \(2,\ 3\)"

    def _plus1(x: int) -> int:
        """Add 1 to x.

        Args:
            x: Input integer.

        Returns:
            x + 1.
        """
        return x + 1

    alias_fn = create_deprecated_function_alias(_plus1, "new_fn", "old_fn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert alias_fn(4) == 5
        assert len(caught) == 1

    @warn_deprecated_function
    def dep1(x: int) -> int:
        """Return x from a deprecated function without args.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    @warn_deprecated_function(replacement="better_fn")
    def dep2(x: int) -> int:
        """Return x from a deprecated function with replacement.

        Args:
            x: Input integer.

        Returns:
            Input integer.
        """
        return x

    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        assert dep1(1) == 1
        assert dep2(2) == 2
        assert len(caught2) == 2

    @warn_only_n_pos_args_in_future(n=1)
    def pos_fn(a: int, b: int = 2, *, c: int = 3) -> int:
        """Accept up to 2 positional args and 1 keyword-only arg, warn if > 1 positional.

        Args:
            a: First integer.
            b: Second integer.
            c: Keyword-only integer.

        Returns:
            Sum of a, b, and c.
        """
        return a + b + c

    @warn_only_n_pos_args_in_future
    def pos_fn_bare(a: int, b: int = 2) -> int:
        """Bare decorator version.

        Args:
            a: First integer.
            b: Second integer.

        Returns:
            Sum of a and b.
        """
        return a + b

    @warn_keyword_args_only_in_future
    def kw_fn(a: int = 1) -> int:
        """Keyword-only in future.

        Args:
            a: Keyword integer.

        Returns:
            Input integer.
        """
        return a

    @warn_keyword_args_only_in_future()
    def kw_fn_parens(a: int = 1) -> int:
        """Keyword-only in future with parentheses.

        Args:
            a: Keyword integer.

        Returns:
            Input integer.
        """
        return a

    with warnings.catch_warnings(record=True) as caught3:
        warnings.simplefilter("always")
        assert pos_fn(1, b=2) == 6
        assert pos_fn(1, 2) == 6
        assert pos_fn_bare(1, 2) == 3
        assert kw_fn(a=5) == 5
        assert kw_fn(5) == 5
        assert kw_fn_parens(5) == 5
        assert len(caught3) == 4

    extra_args: tuple[int, ...] = (2, 3)
    with pytest.raises(TypeError, match="takes at most 2 arguments"):
        pos_fn(1, *extra_args)

    from zero_chex._src import (
        asserts_chexify as src_chexify,
    )
    from zero_chex._src import (
        dataclass as src_dc,
    )
    from zero_chex._src import (
        dimensions as src_dims,
    )
    from zero_chex._src import (
        fake as src_fake,
    )
    from zero_chex._src import (
        restrict_backends as src_rb,
    )
    from zero_chex._src import (
        warnings as src_warn,
    )

    assert src_chexify.chexify is chexify
    assert src_chexify.ify is ify
    assert src_dc.dataclass is dataclass
    assert src_dims.Dimensions.__name__ == "Dimensions"
    assert src_fake.fake_jit is fake_jit
    assert src_rb.restrict_backends is restrict_backends
    assert src_warn.warn_deprecated_function is warn_deprecated_function
