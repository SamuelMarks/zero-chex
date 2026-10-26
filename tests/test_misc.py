"""Tests for dataclasses, trace counters, chexify, numerical grads, context managers, and deprecation utils."""

from __future__ import annotations

import collections.abc
import math
import warnings
from collections.abc import Sequence

import pytest
import zero_jax
import zero_jax.numpy as jnp
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from ml_switcheroo_ir import DType, LogicalNode, TensorSpec
from zero_flax.nnx.state import State
from zero_flax.nnx.variables import Param
from zero_grain import SharedMemoryArray, SharedMemoryArrayMetadata
from zero_jax._types import PyTree as JaxPyTree
from zero_optax import ScaleByAdamState

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
from zero_chex._src.tree_util import tree_flatten, tree_unflatten
from zero_chex.dataclasses import MappableDataclassMixin
from zero_chex.mocks import (
    FakeContext,
    MaxTracesWrapper,
    _invoke_differentiable,
    _scalar_from_numeric,
    _TaylorJet,
    _TaylorJetArray,
    are_asserts_enabled,
)
from zero_chex.types import DataclassInstanceProtocol, MappableDataclassProtocol, Numeric, PyTree, PyTreeDef, ZeroArray


class _ShapeOnlyHolder:
    """Helper class implementing ShapeOnlyProtocol without ArrayProtocol."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return a valid shape tuple."""
        return (2, 2)


def test_assert_toggles_and_max_traces() -> None:
    """Test enable_asserts, disable_asserts, assert_max_traces, and abstract signature caching."""
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
    assert fn_limit(1) == 2
    assert fn_limit(2) == 3
    with pytest.raises(AssertionError, match="Function is traced > 2 times"):
        fn_limit(3)

    clear_trace_counter()
    assert fn_limit(4) == 5

    @assert_max_traces(n=1)
    def _traced_arr_inner(arr: ZeroArray | jnp.ndarray) -> ZeroArray | jnp.ndarray:
        """Identity function wrapped in assert_max_traces(n=1).

        Args:
            arr: Input array.

        Returns:
            Unchanged array.
        """
        return arr

    @zero_jax.jit
    def fn_jit_arr(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """JIT-compiled wrapper around _traced_arr_inner.

        Args:
            *args: Positional array arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Output array.
        """
        _ = kwargs
        first = args[0]
        assert isinstance(first, jnp.ndarray)
        res = _traced_arr_inner(first)
        assert isinstance(res, jnp.ndarray)
        return res

    a1 = jnp.zeros((2, 3))
    a2 = jnp.ones((2, 3))
    a3 = jnp.zeros((4, 5))
    out1 = fn_jit_arr(a1)
    out2 = fn_jit_arr(a2)
    assert isinstance(out1, jnp.ndarray) and out1.shape == (2, 3)
    assert isinstance(out2, jnp.ndarray) and out2.shape == (2, 3)
    with pytest.raises(AssertionError, match="Function is traced > 1 times!"):
        fn_jit_arr(a3)

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
    def fn_unbounded(x: PyTree, **kw: PyTree) -> PyTree:
        """Accept any ecosystem PyTree argument and return it unchanged.

        Args:
            x: Input PyTree.
            **kw: Keyword PyTree arguments.

        Returns:
            Input x.
        """
        _ = kw
        return x

    shm_arr = SharedMemoryArray((2,), "float32")
    shm_meta = SharedMemoryArrayMetadata(name="s", dtype="float32", shape=(2,))
    try:
        assert fn_unbounded(10) == 10
        assert fn_unbounded(Param(jnp.ones((2,)))) is not None
        assert fn_unbounded(shm_arr) is shm_arr
        assert fn_unbounded(shm_meta) is shm_meta
        assert fn_unbounded(TensorSpec(shape=(2,), dtype=DType.float32)) is not None
        assert fn_unbounded(LogicalNode(id="n1", op_type="Relu", shape_metadata=(2,))) is not None
        assert fn_unbounded(_ShapeOnlyHolder()) is not None
        adam_state = ScaleByAdamState(count=as_tensor_obj(1), mu=1.0, nu=1.0)
        assert fn_unbounded({"a": State({"w": Param(1.0)}), "b": adam_state}) is not None
    finally:
        shm_arr.unlink_on_del()

    static_wrapped = MaxTracesWrapper(
        lambda x, y, *, mode="a": x,
        max_traces=2,
        static_argnums=1,
        static_argnames="mode",
    )
    assert static_wrapped(ZeroArray.zeros((2,)), 1, mode="a") == ZeroArray.zeros((2,))
    assert static_wrapped(ZeroArray.ones((2,)), 1, mode="a") == ZeroArray.ones((2,))
    assert static_wrapped(ZeroArray.zeros((2,)), 2, mode="b") == ZeroArray.zeros((2,))
    with pytest.raises(AssertionError, match="Function is traced > 2 times"):
        static_wrapped(ZeroArray.zeros((2,)), 3, mode="c")


def test_chexify_and_with_jittable_assertions() -> None:
    """Test @chexify, @with_jittable_assertions, wait_checks isolation, and LogicalNode checks."""

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

    @chexify
    def f_other(x: int) -> int:
        """Triple x after checking positivity.

        Args:
            x: Input integer.

        Returns:
            Tripled integer.
        """
        assert_scalar_positive(x)
        return x * 3

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
    def f_nan(flag: int) -> PyTree:
        """Produce NaN, complex NaN, LogicalNode error, or index error based on flag.

        Args:
            flag: Selector integer.

        Returns:
            PyTree output.
        """
        if flag == 1:
            return float("nan")
        if flag == 2:
            items = [1.0, 2.0]
            return items[99]
        if flag == 3:
            return LogicalNode(id="div0", op_type="Div", attributes={"divisor": 0})
        if flag == 4:
            return LogicalNode(id="oob", op_type="Gather", attributes={"index": 5}, shape_metadata=(2,))
        if flag == 5:
            return LogicalNode(id="nan_node", op_type="Constant", attributes={"value": float("nan")})
        return [
            State({"p": Param(jnp.array([1.0]))}),
            LogicalNode(id="ok", op_type="Gather", attributes={"index": 0}, shape_metadata=(2,)),
        ]

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
        f_other(-2)
    # Waiting on f1 must not wipe f_other's pending error
    with pytest.raises(AssertionError, match="must be positive"):
        f1.wait_checks()
    f1.wait_checks()
    with pytest.raises(AssertionError, match="must be positive"):
        f_other.wait_checks()
    f_other.wait_checks()

    assert f2(2.0) == 5.0
    f2.wait_checks()
    with pytest.raises(AssertionError, match="division by zero"):
        f2(0.0)
    f2.wait_checks()

    assert f_nan(0) is not None
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        f_nan(1)
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        f_nan.wait_checks()

    with pytest.raises(AssertionError, match="index out-of-bounds"):
        f_nan(2)
    with pytest.raises(AssertionError, match="index out-of-bounds"):
        f_nan.wait_checks()

    with pytest.raises(AssertionError, match="division by zero check failed in LogicalNode"):
        f_nan(3)
    with pytest.raises(AssertionError, match="division by zero check failed in LogicalNode"):
        f_nan.wait_checks()

    with pytest.raises(AssertionError, match="index out-of-bounds check failed in LogicalNode"):
        f_nan(4)
    with pytest.raises(AssertionError, match="index out-of-bounds check failed in LogicalNode"):
        f_nan.wait_checks()

    with pytest.raises(AssertionError, match="NaN/Inf check failed on LogicalNode"):
        f_nan(5)
    with pytest.raises(AssertionError, match="NaN/Inf check failed on LogicalNode"):
        f_nan.wait_checks()

    assert f_complex_nan(False) == 1.0 + 2.0j
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        f_complex_nan(True)
    with pytest.raises(AssertionError, match="NaN/Inf check failed"):
        block_until_chexify_assertions_complete()
    block_until_chexify_assertions_complete()
    f_complex_nan.wait_checks()

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
    disable_asserts()
    try:
        disabled_res = f_nan(1)
        assert isinstance(disabled_res, float) and math.isnan(disabled_res)
    finally:
        enable_asserts()


def test_taylor_jet_and_assert_numerical_grads() -> None:
    """Test _TaylorJet elementary functions, _TaylorJetArray, and assert_numerical_grads."""
    j_var = _TaylorJet.variable(2.0, 3)
    j_zero = _TaylorJet.variable(2.0, 0)
    assert j_zero.order == 0
    assert j_var.derivative(-1) == 0.0
    assert j_var.derivative(99) == 0.0
    assert float(+j_var) == 2.0
    assert float(-j_var) == -2.0
    assert float(abs(-j_var)) == 2.0
    assert float(abs(j_var)) == 2.0
    assert j_var == 2.0
    assert j_var == _TaylorJet.constant(2.0, 1)
    assert j_var != "not_a_jet"
    assert hash(j_var) == hash(_TaylorJet(j_var.coeffs))
    assert j_var < 3.0 and j_var <= 2.0 and j_var > 1.0 and j_var >= 2.0

    # Elementary transcendental functions on _TaylorJet
    assert j_var.exp().derivative(1) == pytest.approx(math.exp(2.0))
    assert j_var.log().derivative(1) == pytest.approx(0.5)
    assert math.isnan(_TaylorJet.variable(-1.0, 2).log().derivative(0))
    assert j_var.sqrt().derivative(1) == pytest.approx(0.5 / math.sqrt(2.0))
    assert j_var.sin().derivative(1) == pytest.approx(math.cos(2.0))
    assert j_var.cos().derivative(1) == pytest.approx(-math.sin(2.0))
    assert j_var.tanh().derivative(1) == pytest.approx(1.0 - math.tanh(2.0) ** 2)

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

    # Test _TaylorJetArray properties and operations
    ja = _TaylorJetArray([_TaylorJet.variable(2.0, 1), _TaylorJet.constant(3.0, 1)], shape=(2,))
    assert ja.shape == (2,)
    assert ja.dtype == DType.float32
    assert ja.size == 2
    assert ja.ndim == 1
    assert len(ja) == 2
    assert ja[0] == 2.0
    assert ja.to_list() == [2.0, 3.0]
    assert list(ja) == [_TaylorJet.variable(2.0, 1), _TaylorJet.constant(3.0, 1)]
    assert (ja + ja).to_list() == [4.0, 6.0]
    assert (1.0 + ja).to_list() == [3.0, 4.0]
    assert (ja - 1.0).to_list() == [1.0, 2.0]
    assert (10.0 - ja).to_list() == [8.0, 7.0]
    assert (2.0 * ja).to_list() == [4.0, 6.0]
    assert (ja * ZeroArray([2.0], shape=(1,))).to_list() == [4.0, 6.0]
    assert (ja * ZeroArray([2.0, 3.0], shape=(2,))).to_list() == [4.0, 9.0]
    assert (ja / 2.0).to_list() == [1.0, 1.5]
    assert (-ja).to_list() == [-2.0, -3.0]
    assert (ja**2).to_list() == [4.0, 9.0]

    assert _scalar_from_numeric(_TaylorJet.constant(5.0, 1)) == 5.0
    assert _scalar_from_numeric(Param(3.5)) == 3.5
    assert _scalar_from_numeric(3.0) == 3.0
    assert _scalar_from_numeric(2 + 0j) == 2.0
    assert _scalar_from_numeric(ZeroArray(4.0)) == 4.0
    assert _scalar_from_numeric(ZeroArray(4 + 0j)) == 4.0
    with pytest.raises(AssertionError, match="Expected real scalar"):
        _scalar_from_numeric(2 + 5j)
    with pytest.raises(AssertionError, match="Expected real scalar"):
        _scalar_from_numeric(ZeroArray(2 + 5j))
    with pytest.raises(ValueError, match="Cannot extract scalar from empty array"):
        _scalar_from_numeric(ZeroArray([1.0], shape=(0,)))
    with pytest.raises(TypeError, match="Expected numeric scalar"):
        _scalar_from_numeric(_ShapeOnlyHolder())

    def algebraic_poly(x: Numeric, y: Numeric) -> Numeric:
        """Compute sum(x^3) + 2*sum(y^2) directly using algebraic operators so _TaylorJet and _TaylorJetArray propagate.

        Args:
            x: First numeric input.
            y: Second numeric input.

        Returns:
            Polynomial evaluation.
        """
        if isinstance(x, (int, float, _TaylorJet)) and isinstance(y, (int, float, _TaylorJet)):
            return x**3 + 2.0 * (y**2)
        if isinstance(x, _TaylorJetArray) and isinstance(y, (int, float, _TaylorJet)):
            acc: float | _TaylorJet = 0.0
            for j in x:
                acc = acc + (j**3)
            return acc + 2.0 * (y**2)
        vx = sum(float(v) ** 3 for v in _invoke_float_elements(x))
        vy = sum(2.0 * (float(v) ** 2) for v in _invoke_float_elements(y))
        return vx + vy

    def _invoke_float_elements(val: Numeric) -> list[float]:
        """Extract float elements from a Numeric scalar or array.

        Args:
            val: Numeric input.

        Returns:
            List of float values.
        """
        if isinstance(val, ZeroArray):
            return [float(v.real) if isinstance(v, complex) else float(v) for v in val.to_list()]
        return [_scalar_from_numeric(val)]

    def const_scalar_fn(x: Numeric) -> Numeric:
        """Return constant 5.0 regardless of x.

        Args:
            x: Input numeric value.

        Returns:
            Constant 5.0.
        """
        _ = x
        return 5.0

    assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=1, atol=1e-3)
    assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=2, atol=1e-2)
    assert_numerical_grads(algebraic_poly, [2.0, 3.0], order=3, atol=5e-2)
    assert_numerical_grads(const_scalar_fn, [2.0], order=1, atol=1e-4)
    assert_numerical_grads(algebraic_poly, [ZeroArray([1.5, 2.5], shape=(2,)), 1.0], order=1, atol=1e-2)

    # Test zero_jax.grad integration and detection of broken custom_vjp rules
    def jax_square_sum(x: PyTree) -> PyTree:
        """Compute sum(x * x) using zero_jax.numpy.

        Args:
            x: Input JaxNdarray.

        Returns:
            Scalar sum of squares.
        """
        assert isinstance(x, jnp.ndarray)
        res = jnp.sum(x * x)
        assert isinstance(res, jnp.ndarray)
        return res

    assert_numerical_grads(jax_square_sum, [jnp.array([1.5, -2.0])], order=1, atol=1e-2)

    @zero_jax.custom_vjp
    def broken_custom_vjp(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
        """Primal function x^2 with a deliberately broken custom VJP rule.

        Args:
            *args: Positional scalar JaxNdarray.
            **kwargs: Unused keyword arguments.

        Returns:
            Squared scalar.
        """
        _ = kwargs
        x = args[0]
        assert isinstance(x, jnp.ndarray)
        out = x * x
        assert isinstance(out, jnp.ndarray)
        return out

    def _broken_fwd(
        *args: JaxPyTree,
        **kwargs: JaxPyTree,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for broken_custom_vjp.

        Args:
            *args: Input scalar.
            **kwargs: Unused keyword arguments.

        Returns:
            Pair of (primal_out, residual).
        """
        _ = kwargs
        x = args[0]
        assert isinstance(x, jnp.ndarray)
        out = x * x
        assert isinstance(out, jnp.ndarray)
        return out, x

    def _broken_bwd(
        *args: JaxPyTree,
        **kwargs: JaxPyTree,
    ) -> tuple[jnp.ndarray]:
        """Deliberately incorrect backward pass returning 50 * g * res instead of 2 * g * res.

        Args:
            *args: Saved residual x and cotangent g.
            **kwargs: Unused keyword arguments.

        Returns:
            1-tuple with wrong gradient.
        """
        _ = kwargs
        res = args[0]
        g = args[1]
        assert isinstance(res, jnp.ndarray) and isinstance(g, jnp.ndarray)
        wrong = g * (res * 50.0)
        assert isinstance(wrong, jnp.ndarray)
        return (wrong,)

    broken_custom_vjp.defvjp(_broken_fwd, _broken_bwd)
    with pytest.raises(AssertionError, match="Gradient mismatch"):
        assert_numerical_grads(broken_custom_vjp, [jnp.array(3.0)], order=1, atol=1e-2)

    def pytree_fn(tree_arg: PyTree) -> PyTree:
        """Compute squared leaves for a dict PyTree argument using algebraic ops.

        Args:
            tree_arg: Dictionary PyTree with numeric leaves.

        Returns:
            Dictionary with transformed leaves.
        """
        assert isinstance(tree_arg, dict)
        a_leaf = tree_arg["a"]
        b_leaf = tree_arg["b"]
        assert isinstance(a_leaf, (int, float, _TaylorJet))
        assert isinstance(b_leaf, (int, float, ZeroArray, _TaylorJet, _TaylorJetArray))
        if isinstance(b_leaf, ZeroArray):
            b_out: PyTree = ZeroArray(
                [2.0 * float(v.real if isinstance(v, complex) else v) for v in b_leaf.to_list()], shape=b_leaf.shape
            )
        else:
            b_out = 2.0 * b_leaf
        return {"a": a_leaf**2, "b": b_out}

    assert_numerical_grads(pytree_fn, [{"a": 3.0, "b": ZeroArray([1.0, 2.0], shape=(2,))}], order=1, atol=1e-2)
    assert _invoke_differentiable(pytree_fn, [{"a": 2.0, "b": 3.0}]) == {"a": 4.0, "b": 6.0}

    class _CorrectAnalyticalFn:
        """Callable with an exact analytical .grad() method."""

        def __call__(self, *args: PyTree) -> PyTree:
            """Compute x^2 + 3*y.

            Args:
                *args: Input numeric arguments.

            Returns:
                Computed scalar value.
            """
            x = _scalar_from_numeric(args[0])
            y = _scalar_from_numeric(args[1])
            return x * x + 3.0 * y

        def grad(self, *args: PyTree) -> Sequence[Numeric]:
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

        def __call__(self, *args: PyTree) -> PyTree:
            """Compute x^2.

            Args:
                *args: Input numeric arguments.

            Returns:
                Computed scalar value.
            """
            x = _scalar_from_numeric(args[0])
            return x * x

        def grad(self, *args: PyTree) -> Numeric:
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

        def __call__(self, *args: PyTree) -> PyTree:
            """Compute x.

            Args:
                *args: Input numeric arguments.

            Returns:
                Computed scalar value.
            """
            return _scalar_from_numeric(args[0])

        def grad(self, *args: PyTree) -> Numeric:
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
    with fake_pmap(jit_result=True, fake_parallel_axis=True) as p_ctx:
        assert p_ctx.active is True
        assert pmap_test.run_step(4) == 8

        def _pmap_double(*args: JaxPyTree, **kwargs: JaxPyTree) -> JaxPyTree:
            """Double first array argument under pmap.

            Args:
                *args: Input array arguments.
                **kwargs: Unused keyword arguments.

            Returns:
                Doubled array.
            """
            _ = kwargs
            first = args[0]
            assert isinstance(first, jnp.ndarray)
            res = first * 2
            assert isinstance(res, jnp.ndarray)
            return res

        pmapped_arr_fn = zero_jax.pmap(_pmap_double)
        res_arr = pmapped_arr_fn(jnp.array([1.0, 2.0]))
        res_arr2 = pmapped_arr_fn(jnp.array([3.0, 4.0]))
        assert isinstance(res_arr, jnp.ndarray) and res_arr.shape == (2,)
        assert isinstance(res_arr2, jnp.ndarray) and res_arr2.shape == (2,)
    assert p_ctx.active is False

    with fake_pmap_and_jit() as pj_ctx:
        assert pj_ctx.active is True
        assert pmap_test.run_step(5) == 10
    assert pj_ctx.active is False

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
    jax_leaves, jax_treedef = zero_jax.tree_util.tree_flatten(p)
    assert jax_leaves == [1, 2]
    assert zero_jax.tree_util.tree_unflatten(jax_treedef, [30, 40]) == Point(30, 40)

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
        """Dataclass with mappable_dataclass=False and registered_with_pytree=True."""

        x: int = 0
        y: int = 0

    nmp = NonMappablePoint(3, 4)
    assert nmp.x == 3 and nmp.y == 4
    assert not isinstance(nmp, MappableDataclassProtocol)
    nmp_leaves, nmp_def = tree_flatten(nmp)
    assert nmp_leaves == [3, 4]
    assert tree_unflatten(nmp_def, [11, 22]) == NonMappablePoint(11, 22)

    mapped_nmp_cls = mappable_dataclass(NonMappablePoint)
    mapped_nmp = mapped_nmp_cls(5, 6)
    assert isinstance(mapped_nmp, MappableDataclassProtocol)
    assert mapped_nmp.__chex_mappable_fields__() == ("x", "y")
    assert mapped_nmp.__chex_field_values__() == (5, 6)

    @dataclass(mappable_dataclass=False, registered_with_pytree=False)
    class RawDataclassPoint:
        """Dataclass with both mappable_dataclass=False and registered_with_pytree=False."""

        x: int = 1
        y: int = 2

    raw_pt = RawDataclassPoint(9, 10)
    assert raw_pt.x == 9 and raw_pt.y == 10
    raw_leaves, _ = tree_flatten(raw_pt)
    assert raw_leaves == [raw_pt]

    class PlainBox:
        """Plain class converted by register_dataclass_type_with_jax_tree_util and mappable_dataclass."""

        val: int = 0

        def __init__(self, val: int = 0) -> None:
            """Initialize PlainBox with val.

            Args:
                val: Box integer value.
            """
            self.val = val

    class AnotherPlainBox:
        """Another plain class converted directly by mappable_dataclass."""

        val: int = 0

    mapped_another_cls = mappable_dataclass(AnotherPlainBox)
    assert issubclass(mapped_another_cls, MappableDataclassProtocol)
    reg_box = register_dataclass_type_with_jax_tree_util(PlainBox)
    box_inst = reg_box(42)
    assert isinstance(box_inst, DataclassInstanceProtocol)
    box_leaves, box_def = tree_flatten(box_inst)
    assert box_leaves == [42]
    rebuilt_box = tree_unflatten(box_def, [99])
    assert rebuilt_box == reg_box(99)


def test_utils_params_product_and_deprecations() -> None:
    """Test params_product, if_args_not_none, get_err_regex, deprecation decorators, and _src behavioral parity."""
    unnamed = params_product([1, 2], ["a", "b"])
    assert unnamed == [(1, "a"), (1, "b"), (2, "a"), (2, "b")]

    unnamed_tuples = params_product([(1, 2)], [(3,)])
    assert unnamed_tuples == [(1, 2, 3)]

    named = params_product([("case1", 10), ("case2", 20)], [{"sub": (30, 40)}], named=True)
    assert named == [
        ("case1_{'sub': (30, 40)}", 10, {"sub": (30, 40)}),
        ("case2_{'sub': (30, 40)}", 20, {"sub": (30, 40)}),
    ]

    named_mapping = params_product({"k1": 1, "k2": (2, 3)}, [10], named=True)
    assert named_mapping == [("k1_10", 1, 10), ("k2_10", 2, 3, 10)]

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
    alias_default = create_deprecated_function_alias(_plus1, "new_fn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert alias_fn(4) == 5
        assert alias_default(4) == 5
        assert len(caught) == 2
        assert "old_fn" in str(caught[0].message)
        assert "This function is deprecated; please use new_fn instead." in str(caught[1].message)

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

    src_wrapped = src_chexify.chexify(lambda x: x + 1)
    assert src_wrapped(2) == 3
    src_wrapped.wait_checks()

    @src_dc.dataclass
    class _SrcPair(MappableDataclassMixin):
        """Dataclass created via zero_chex._src.dataclass."""

        u: int = 1
        v: int = 2

    assert dict(_SrcPair(3, 4)) == {"u": 3, "v": 4}
    d_obj = src_dims.Dimensions(B=4, T=8)
    assert d_obj["BT"] == (4, 8)
    with src_fake.fake_jit() as fj:
        assert fj.active
    with src_rb.restrict_backends(allowed=["cpu"]):
        pass
    with warnings.catch_warnings(record=True) as caught_src:
        warnings.simplefilter("always")
        dep_src = src_warn.warn_deprecated_function(lambda x: x)
        assert dep_src(9) == 9
        assert len(caught_src) == 1
