"""Assertion toggles, trace counters, chexify wrappers, and execution contexts for zero_chex."""

from __future__ import annotations

import contextlib
import functools
import math
import unittest.mock
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Generic, Protocol, TypeVar, Union, overload, runtime_checkable

import zero_jax
import zero_jax._config_internal as _jax_cfg
import zero_jax.numpy as jnp
import zero_optax._tree as _optax_tree
import zero_optax.losses.regression as _optax_regression
from ml_switcheroo_compiler.core.tensor import TensorConfig as _CompilerTensorConfig
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor as _ProxyTensor
from ml_switcheroo_ir import DType, LogicalNode, TensorSpec
from typing_extensions import ParamSpec, TypeGuard
from zero_jax._types import PyTree as JaxPyTree
from zero_jax.api.ad import CustomJVPWrapper, CustomVJPWrapper
from zero_optax._types import ArrayLike as _OptaxArrayLike
from zero_optax._types import DTypeLike as _OptaxDTypeLike

from zero_chex._src.asserts_internal import (
    RUNTIME_STATE as _STATE,
)
from zero_chex._src.asserts_internal import (
    are_asserts_enabled,
    clear_trace_counter,
    disable_asserts,
    enable_asserts,
    get_num_devices_available,
    set_n_cpu_devices,
)
from zero_chex._src.asserts_internal import (
    extract_flat_scalars as _extract_flat_scalars,
)
from zero_chex._src.asserts_internal import (
    get_leaf_shape as _get_shape,
)
from zero_chex._src.asserts_internal import (
    get_leaf_type as _get_type,
)
from zero_chex._src.asserts_internal import (
    has_valid_shape_tuple as _has_valid_shape_tuple,
)
from zero_chex._src.asserts_internal import (
    normalize_dtype_str as _normalize_dtype,
)
from zero_chex._src.asserts_internal import (
    raise_chex_error as _raise_chex_error,
)
from zero_chex._src.tree_util import _CUSTOM_PYTREE_NODES, tree_flatten, tree_leaves, tree_unflatten
from zero_chex.types import (
    ArrayProtocol,
    ChexifyChecks,
    CompilerDType,
    CompilerTensor,
    DTypeLikeProtocol,
    JaxNdarray,
    LeafValue,
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    Numeric,
    PyTree,
    Scalar,
    ShapeOnlyProtocol,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    StateMappingProtocol,
    VariableLeafProtocol,
    ZeroArray,
)

P = ParamSpec("P")
R = TypeVar("R")


class _TaylorJet:
    """Truncated univariate Taylor polynomial jet for exact arbitrary-order forward-mode autodiff."""

    def __init__(self, coeffs: Sequence[float]) -> None:
        """Initialize a Taylor jet with coefficients [f(x0), f'(x0), f''(x0)/2!, ..., f^(k)(x0)/k!].

        Args:
            coeffs: Sequence of Taylor series coefficients up to truncation order.
        """
        self.coeffs: tuple[float, ...] = tuple(float(c) for c in coeffs)

    @property
    def order(self) -> int:
        """Return the truncation order of the Taylor jet.

        Returns:
            Maximum derivative order represented.
        """
        return len(self.coeffs) - 1

    @classmethod
    def constant(cls, val: float, order: int) -> _TaylorJet:
        """Construct a constant Taylor jet of given truncation order.

        Args:
            val: Scalar constant value.
            order: Truncation order.

        Returns:
            _TaylorJet with coeffs (val, 0.0, ..., 0.0).
        """
        return cls((float(val),) + (0.0,) * order)

    @classmethod
    def variable(cls, val: float, order: int) -> _TaylorJet:
        """Construct an active seeded variable Taylor jet x(t) = val + 1.0 * t.

        Args:
            val: Base scalar value at t = 0.
            order: Truncation order.

        Returns:
            _TaylorJet seeded with unit first-order tangent.
        """
        if order <= 0:
            return cls((float(val),))
        return cls((float(val), 1.0) + (0.0,) * (order - 1))

    def derivative(self, n: int) -> float:
        """Extract the n-th derivative f^(n)(x0) = n! * coeffs[n].

        Args:
            n: Derivative order to extract.

        Returns:
            The n-th derivative value.
        """
        if n < 0 or n >= len(self.coeffs):
            return 0.0
        return self.coeffs[n] * float(math.factorial(n))

    def _align(self, other: _TaylorJet | float | int) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Align this jet and another jet or scalar to matching coefficient length.

        Args:
            other: Another _TaylorJet or real scalar.

        Returns:
            Pair of aligned coefficient tuples.
        """
        if isinstance(other, _TaylorJet):
            k = max(len(self.coeffs), len(other.coeffs))
            a = self.coeffs + (0.0,) * (k - len(self.coeffs))
            b = other.coeffs + (0.0,) * (k - len(other.coeffs))
            return a, b
        return self.coeffs, (float(other),) + (0.0,) * (len(self.coeffs) - 1)

    def __add__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Add a Taylor jet or scalar.

        Args:
            other: Right-hand operand.

        Returns:
            Sum _TaylorJet.
        """
        a, b = self._align(other)
        return _TaylorJet(tuple(x + y for x, y in zip(a, b)))

    def __radd__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Right-add a scalar or Taylor jet.

        Args:
            other: Left-hand operand.

        Returns:
            Sum _TaylorJet.
        """
        return self.__add__(other)

    def __sub__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Subtract a Taylor jet or scalar.

        Args:
            other: Right-hand operand.

        Returns:
            Difference _TaylorJet.
        """
        a, b = self._align(other)
        return _TaylorJet(tuple(x - y for x, y in zip(a, b)))

    def __rsub__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Right-subtract this Taylor jet from a scalar.

        Args:
            other: Left-hand operand.

        Returns:
            Difference _TaylorJet.
        """
        a, b = self._align(other)
        return _TaylorJet(tuple(y - x for x, y in zip(a, b)))

    def __neg__(self) -> _TaylorJet:
        """Negate all Taylor coefficients.

        Returns:
            Negated _TaylorJet.
        """
        return _TaylorJet(tuple(-x for x in self.coeffs))

    def __pos__(self) -> _TaylorJet:
        """Unary positive operator.

        Returns:
            This _TaylorJet.
        """
        return self

    def __abs__(self) -> _TaylorJet:
        """Compute absolute value of this Taylor jet.

        Returns:
            Absolute value _TaylorJet.
        """
        return -self if self.coeffs[0] < 0.0 else self

    def __mul__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Multiply two Taylor jets via truncated Cauchy product.

        Args:
            other: Right-hand operand.

        Returns:
            Product _TaylorJet.
        """
        a, b = self._align(other)
        k = len(a)
        out: list[float] = []
        for i in range(k):
            out.append(sum(a[j] * b[i - j] for j in range(i + 1)))
        return _TaylorJet(out)

    def __rmul__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Right-multiply a scalar and this Taylor jet.

        Args:
            other: Left-hand operand.

        Returns:
            Product _TaylorJet.
        """
        return self.__mul__(other)

    def __truediv__(self, other: _TaylorJet | float | int) -> _TaylorJet:
        """Divide this Taylor jet by another Taylor jet or scalar via power series recurrence.

        Args:
            other: Divisor _TaylorJet or scalar.

        Returns:
            Quotient _TaylorJet.
        """
        a, b = self._align(other)
        k = len(a)
        if b[0] == 0.0:
            return _TaylorJet((float("nan"),) * k)
        c: list[float] = []
        for i in range(k):
            s = a[i] - sum(c[j] * b[i - j] for j in range(i))
            c.append(s / b[0])
        return _TaylorJet(c)

    def __rtruediv__(self, other: float | int) -> _TaylorJet:
        """Divide a scalar by this Taylor jet.

        Args:
            other: Numerator scalar.

        Returns:
            Quotient _TaylorJet.
        """
        return _TaylorJet.constant(float(other), self.order).__truediv__(self)

    def __pow__(self, exponent: int | float) -> _TaylorJet:
        """Raise this Taylor jet to an integer or real power.

        Args:
            exponent: Power exponent.

        Returns:
            Exponentiated _TaylorJet.
        """
        if isinstance(exponent, int):
            if exponent == 0:
                return _TaylorJet.constant(1.0, self.order)
            if exponent > 0:
                res = self
                for _ in range(exponent - 1):
                    res = res * self
                return res
            return _TaylorJet.constant(1.0, self.order) / (self ** (-exponent))
        a = self.coeffs
        k = len(a)
        if a[0] <= 0.0:
            return _TaylorJet((float("nan"),) * k)
        c: list[float] = [a[0] ** float(exponent)]
        for n in range(1, k):
            s = sum((float(exponent) * j - (n - j)) * a[j] * c[n - j] for j in range(1, n + 1))
            c.append(s / (n * a[0]))
        return _TaylorJet(c)

    def exp(self) -> _TaylorJet:
        """Compute exponential exp(u(t)) via differential recurrence.

        Returns:
            Exponential _TaylorJet.
        """
        a = self.coeffs
        k = len(a)
        c: list[float] = [math.exp(a[0])]
        for n in range(1, k):
            s = sum(j * a[j] * c[n - j] for j in range(1, n + 1))
            c.append(s / float(n))
        return _TaylorJet(c)

    def log(self) -> _TaylorJet:
        """Compute natural logarithm log(u(t)) via differential recurrence.

        Returns:
            Logarithm _TaylorJet.
        """
        a = self.coeffs
        k = len(a)
        if a[0] <= 0.0:
            return _TaylorJet((float("nan"),) * k)
        c: list[float] = [math.log(a[0])]
        for n in range(1, k):
            s = sum(j * c[j] * a[n - j] for j in range(1, n))
            c.append((a[n] - s / float(n)) / a[0])
        return _TaylorJet(c)

    def sqrt(self) -> _TaylorJet:
        """Compute square root sqrt(u(t)) via power series recurrence.

        Returns:
            Square root _TaylorJet.
        """
        return self**0.5

    def _sin_cos(self) -> tuple[_TaylorJet, _TaylorJet]:
        """Simultaneously compute sin(u(t)) and cos(u(t)) via coupled recurrence.

        Returns:
            Pair of (sin_jet, cos_jet).
        """
        a = self.coeffs
        k = len(a)
        s: list[float] = [math.sin(a[0])]
        c: list[float] = [math.cos(a[0])]
        for n in range(1, k):
            sn = sum(j * a[j] * c[n - j] for j in range(1, n + 1)) / float(n)
            cn = -sum(j * a[j] * s[n - j] for j in range(1, n + 1)) / float(n)
            s.append(sn)
            c.append(cn)
        return _TaylorJet(s), _TaylorJet(c)

    def sin(self) -> _TaylorJet:
        """Compute sine sin(u(t)) of this Taylor jet.

        Returns:
            Sine _TaylorJet.
        """
        return self._sin_cos()[0]

    def cos(self) -> _TaylorJet:
        """Compute cosine cos(u(t)) of this Taylor jet.

        Returns:
            Cosine _TaylorJet.
        """
        return self._sin_cos()[1]

    def tanh(self) -> _TaylorJet:
        """Compute hyperbolic tangent tanh(u(t)) of this Taylor jet.

        Returns:
            Hyperbolic tangent _TaylorJet.
        """
        e2x = (self * 2.0).exp()
        return (e2x - 1.0) / (e2x + 1.0)

    def __float__(self) -> float:
        """Return the primal 0-th order scalar value.

        Returns:
            Primal float value.
        """
        return self.coeffs[0]

    def __eq__(self, other: PyTree | _TaylorJet) -> bool:
        """Compare primal value for equality with another jet or scalar.

        Args:
            other: Value to compare against.

        Returns:
            True if primal values are equal.
        """
        if isinstance(other, _TaylorJet):
            return self.coeffs[0] == other.coeffs[0]
        if isinstance(other, (int, float, bool)):
            return self.coeffs[0] == float(other)
        return False

    def __hash__(self) -> int:
        """Compute hash from Taylor coefficients.

        Returns:
            Integer hash.
        """
        return hash(self.coeffs)

    def __lt__(self, other: _TaylorJet | float | int) -> bool:
        """Return True if primal value is strictly less than other.

        Args:
            other: Value to compare against.

        Returns:
            Boolean comparison result.
        """
        return self.coeffs[0] < float(other)

    def __le__(self, other: _TaylorJet | float | int) -> bool:
        """Return True if primal value is less than or equal to other.

        Args:
            other: Value to compare against.

        Returns:
            Boolean comparison result.
        """
        return self.coeffs[0] <= float(other)

    def __gt__(self, other: _TaylorJet | float | int) -> bool:
        """Return True if primal value is strictly greater than other.

        Args:
            other: Value to compare against.

        Returns:
            Boolean comparison result.
        """
        return self.coeffs[0] > float(other)

    def __ge__(self, other: _TaylorJet | float | int) -> bool:
        """Return True if primal value is greater than or equal to other.

        Args:
            other: Value to compare against.

        Returns:
            Boolean comparison result.
        """
        return self.coeffs[0] >= float(other)


class _TaylorJetArray:
    """Elementwise array of _TaylorJet instances satisfying ArrayProtocol for forward-mode autodiff."""

    def __init__(
        self,
        jets: Sequence[_TaylorJet],
        shape: tuple[int, ...],
        dtype: DType | CompilerDType | DTypeLikeProtocol | str | type[Scalar] | type[str] = DType.float32,
    ) -> None:
        """Initialize a _TaylorJetArray.

        Args:
            jets: Flat sequence of _TaylorJet elements.
            shape: Array shape tuple.
            dtype: Logical array dtype.
        """
        self._jets: tuple[_TaylorJet, ...] = tuple(jets)
        self._shape: tuple[int, ...] = shape
        self._dtype = dtype

    @property
    def jets(self) -> tuple[_TaylorJet, ...]:
        """Return the flat tuple of _TaylorJet elements.

        Returns:
            Tuple of _TaylorJet instances.
        """
        return self._jets

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape tuple of the jet array.

        Returns:
            Shape tuple.
        """
        return self._shape

    @property
    def dtype(self) -> DType | CompilerDType | DTypeLikeProtocol | str | type[Scalar] | type[str]:
        """Return the element dtype of the jet array.

        Returns:
            Array dtype.
        """
        return self._dtype

    @property
    def size(self) -> int:
        """Return the total number of elements in the jet array.

        Returns:
            Element count.
        """
        return len(self._jets)

    @property
    def ndim(self) -> int:
        """Return the rank of the jet array.

        Returns:
            Number of dimensions.
        """
        return len(self._shape)

    def to_list(self) -> list[Scalar]:
        """Return the flat primal float values of the jet array.

        Returns:
            List of primal float scalars.
        """
        return [float(j) for j in self._jets]

    def __iter__(self) -> Iterator[_TaylorJet]:
        """Iterate over the _TaylorJet elements.

        Returns:
            Iterator of _TaylorJet elements.
        """
        return iter(self._jets)

    def __len__(self) -> int:
        """Return the leading dimension length.

        Returns:
            Size of dimension 0.
        """
        return self._shape[0] if self._shape else 1

    def __getitem__(self, idx: int) -> _TaylorJet:
        """Return the _TaylorJet at flat index idx.

        Args:
            idx: Element index.

        Returns:
            _TaylorJet at index idx.
        """
        return self._jets[idx]

    def _binary_op(
        self,
        other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int,
        op: Callable[[_TaylorJet, _TaylorJet | float | int], _TaylorJet],
    ) -> _TaylorJetArray:
        """Apply an elementwise binary operation across this jet array and another operand.

        Args:
            other: Right-hand _TaylorJetArray, ArrayProtocol, _TaylorJet, or scalar.
            op: Binary operator function.

        Returns:
            Resulting _TaylorJetArray.
        """
        if isinstance(other, _TaylorJetArray):
            return _TaylorJetArray([op(a, b) for a, b in zip(self._jets, other.jets)], self._shape, self._dtype)
        if isinstance(other, (_TaylorJet, int, float)):
            return _TaylorJetArray([op(a, other) for a in self._jets], self._shape, self._dtype)
        other_scalars = [_scalar_from_numeric(v) for v in _extract_flat_scalars(other)]
        if len(other_scalars) == 1:
            return _TaylorJetArray([op(a, other_scalars[0]) for a in self._jets], self._shape, self._dtype)
        return _TaylorJetArray([op(a, b) for a, b in zip(self._jets, other_scalars)], self._shape, self._dtype)

    def __add__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise addition.

        Args:
            other: Operand to add.

        Returns:
            Sum _TaylorJetArray.
        """
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise right-addition.

        Args:
            other: Left operand.

        Returns:
            Sum _TaylorJetArray.
        """
        return self.__add__(other)

    def __sub__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise subtraction.

        Args:
            other: Operand to subtract.

        Returns:
            Difference _TaylorJetArray.
        """
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise right-subtraction.

        Args:
            other: Left operand.

        Returns:
            Difference _TaylorJetArray.
        """
        return self._binary_op(other, lambda a, b: b - a)

    def __mul__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise multiplication.

        Args:
            other: Operand to multiply.

        Returns:
            Product _TaylorJetArray.
        """
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise right-multiplication.

        Args:
            other: Left operand.

        Returns:
            Product _TaylorJetArray.
        """
        return self.__mul__(other)

    def __truediv__(self, other: _TaylorJetArray | _TaylorJet | ArrayProtocol | float | int) -> _TaylorJetArray:
        """Elementwise division.

        Args:
            other: Divisor operand.

        Returns:
            Quotient _TaylorJetArray.
        """
        return self._binary_op(other, lambda a, b: a / b)

    def __pow__(self, exponent: int | float) -> _TaylorJetArray:
        """Elementwise power operation.

        Args:
            exponent: Power exponent.

        Returns:
            Exponentiated _TaylorJetArray.
        """
        return _TaylorJetArray([a**exponent for a in self._jets], self._shape, self._dtype)

    def __neg__(self) -> _TaylorJetArray:
        """Elementwise negation.

        Returns:
            Negated _TaylorJetArray.
        """
        return _TaylorJetArray([-a for a in self._jets], self._shape, self._dtype)


@runtime_checkable
class AnalyticalGradCallableProtocol(Protocol):
    """Protocol for differentiable callables that expose an explicit analytical `grad` callable."""

    def __call__(self, *args: PyTree) -> PyTree:
        """Evaluate the primal function at the given arguments.

        Args:
            *args: Input arguments.

        Returns:
            Function output.
        """
        ...

    def grad(self, *args: PyTree) -> PyTree:
        """Evaluate the analytical gradient at the given arguments.

        Args:
            *args: Input arguments.

        Returns:
            Analytical gradient or sequence of gradients per argument.
        """
        ...


class VariadicNumericCallableProtocol(Protocol):
    """Protocol for callables accepting positional Numeric arguments."""

    def __call__(self, *args: Numeric) -> Numeric:
        """Evaluate the function on positional Numeric arguments.

        Args:
            *args: Numeric input arguments.

        Returns:
            Numeric result.
        """
        ...


@runtime_checkable
class VariadicPyTreeCallableProtocol(Protocol):
    """Protocol for callables accepting positional PyTree arguments."""

    def __call__(self, *args: PyTree) -> PyTree:
        """Evaluate the function on positional PyTree arguments.

        Args:
            *args: PyTree input arguments.

        Returns:
            PyTree result.
        """
        ...


DifferentiableCallable = Union[
    AnalyticalGradCallableProtocol,
    VariadicNumericCallableProtocol,
    VariadicPyTreeCallableProtocol,
    Callable[[Numeric], Numeric],
    Callable[[Numeric, Numeric], Numeric],
    Callable[[Numeric, Numeric, Numeric], Numeric],
    Callable[[PyTree], PyTree],
    Callable[[PyTree, PyTree], PyTree],
    Callable[[PyTree, PyTree, PyTree], PyTree],
]


def _is_numeric_arg(val: PyTree) -> TypeGuard[Numeric]:
    """Check whether a PyTree argument is a Numeric scalar or array.

    Args:
        val: Candidate PyTree argument.

    Returns:
        True if val is a Numeric value.
    """
    return isinstance(
        val,
        (
            int,
            float,
            complex,
            bool,
            _TaylorJet,
            _TaylorJetArray,
            ArrayProtocol,
            ShapeOnlyProtocol,
            TensorSpec,
        ),
    )


def _invoke_differentiable(
    f: DifferentiableCallable,
    args: Sequence[PyTree],
) -> PyTree:
    """Invoke a DifferentiableCallable with a sequence of positional arguments.

    Args:
        f: Differentiable callable to invoke.
        args: Sequence of positional PyTree arguments.

    Returns:
        Function evaluation result.
    """
    if all(_is_numeric_arg(a) for a in args):
        num_args: list[Numeric] = [a for a in args if _is_numeric_arg(a)]
        return functools.partial(f, *num_args)()
    assert isinstance(f, VariadicPyTreeCallableProtocol)
    return f(*args)


def _abstract_leaf_signature(item: PyTree, is_static: bool = False) -> str:
    """Compute an abstract signature string for a single argument or PyTree leaf.

    Args:
        item: Positional or keyword argument or leaf value.
        is_static: Whether the argument is marked static via static_argnums/static_argnames.

    Returns:
        Deterministic string representation of the argument's tracing signature.
    """
    if is_static:
        return f"static:{type(item).__name__}:{item!r}"
    if isinstance(item, VariableLeafProtocol):
        return f"var:{type(item).__name__}:{_abstract_leaf_signature(item.value, is_static=False)}"
    if isinstance(item, SharedMemoryArray):
        return f"shm:{tuple(item.metadata.shape)}:{item.metadata.dtype}"
    if isinstance(item, SharedMemoryArrayMetadata):
        return f"shm_meta:{tuple(item.shape)}:{item.dtype}"
    if isinstance(item, TensorSpec):
        return f"spec:{item.static_shape}:{_normalize_dtype(item.dtype)}"
    if isinstance(item, LogicalNode):
        return f"node:{item.op_type}:{item.shape_metadata}:{item.dtype}"
    if isinstance(item, (CompilerTensor, JaxNdarray, ArrayProtocol)):
        return f"arr:{_get_shape(item)}:{_normalize_dtype(_get_type(item))}"
    if isinstance(item, ShapeOnlyProtocol) and _has_valid_shape_tuple(item):
        return f"shape:{_get_shape(item)}"
    if (
        isinstance(
            item,
            (
                list,
                tuple,
                Mapping,
                MappableDataclassProtocol,
                NamedTuplePyTreeProtocol,
                StateMappingProtocol,
            ),
        )
        or type(item) in _CUSTOM_PYTREE_NODES
    ):
        leaves, treedef = tree_flatten(item)
        leaf_sigs = [_abstract_leaf_signature(lf, is_static=False) for lf in leaves]
        return f"tree:{treedef}:{leaf_sigs}"
    return f"scalar:{type(item).__name__}:{item!r}"


def _abstract_arg_signature(
    args: Sequence[PyTree],
    kwargs: Mapping[str, PyTree],
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Sequence[str] | None = None,
) -> str:
    """Compute a deterministic abstract shape/dtype/structure signature for JIT trace caching.

    Args:
        args: Positional call arguments.
        kwargs: Keyword call arguments.
        static_argnums: Optional positional indices treated as static values.
        static_argnames: Optional keyword argument names treated as static values.

    Returns:
        String key representing the abstract signature of (args, kwargs).
    """
    static_idx_set: set[int] = (
        {static_argnums}
        if isinstance(static_argnums, int)
        else (set(static_argnums) if static_argnums is not None else set())
    )
    static_name_set: set[str] = (
        {static_argnames}
        if isinstance(static_argnames, str)
        else (set(static_argnames) if static_argnames is not None else set())
    )
    parts: list[str] = []
    for idx, item in enumerate(args):
        parts.append(_abstract_leaf_signature(item, is_static=(idx in static_idx_set)))
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        parts.append(f"kw:{k}:{_abstract_leaf_signature(v, is_static=(k in static_name_set))}")
    return "|".join(parts)


class MaxTracesWrapper(Generic[P, R]):
    """Typed callable wrapper enforcing an upper bound on function trace counts."""

    def __init__(
        self,
        fn: Callable[P, R],
        max_traces: int | None,
        static_argnums: int | Sequence[int] | None = None,
        static_argnames: str | Sequence[str] | None = None,
    ) -> None:
        """Initialize a MaxTracesWrapper around a target function.

        Args:
            fn: The callable to monitor.
            max_traces: Maximum allowed number of distinct traces before raising AssertionError.
            static_argnums: Optional positional indices treated as static during trace caching.
            static_argnames: Optional keyword names treated as static during trace caching.
        """
        self._fn = fn
        self._max_traces = max_traces
        self._static_argnums = static_argnums
        self._static_argnames = static_argnames
        self._key = _STATE.allocate_fn_key("max_traces")
        self.__name__: str = type(fn).__name__
        self.__wrapped__ = fn
        functools.update_wrapper(self, fn, assigned=("__doc__",))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the wrapped function and increment its trace counter on new abstract signatures.

        Args:
            *args: Positional arguments forwarded to the wrapped function.
            **kwargs: Keyword arguments forwarded to the wrapped function.

        Returns:
            The return value of the wrapped function.

        Raises:
            AssertionError: If assertions are enabled and trace count exceeds max_traces.
        """
        if not _STATE.jit_disabled and not bool(_jax_cfg.config.read("jax_disable_jit")):
            pytree_args = [a for a in args if _is_pytree_result(a)]
            pytree_kwargs = {k: v for k, v in kwargs.items() if _is_pytree_result(v)}
            sig = _abstract_arg_signature(
                pytree_args,
                pytree_kwargs,
                static_argnums=self._static_argnums,
                static_argnames=self._static_argnames,
            )
            with _STATE.lock:
                seen = _STATE.trace_seen_signatures.setdefault(self._key, set())
                if sig not in seen:
                    seen.add(sig)
                    current = _STATE.trace_counts.get(self._key, 0) + 1
                else:
                    current = _STATE.trace_counts.get(self._key, 0)
                _STATE.trace_counts[self._key] = current
                if are_asserts_enabled() and self._max_traces is not None and current > self._max_traces:
                    if global_tracing_state.is_tracing:
                        global_tracing_state.stop_tracing()
                    raise AssertionError(f"Function is traced > {self._max_traces} times! (actual: {current})")
        return self._fn(*args, **kwargs)


@overload
def assert_max_traces(
    fn: Callable[P, R],
    n: int | None = None,
) -> MaxTracesWrapper[P, R]: ...


@overload
def assert_max_traces(
    fn: int | None = None,
    n: int | None = None,
) -> Callable[[Callable[P, R]], MaxTracesWrapper[P, R]]: ...


def assert_max_traces(
    fn: Callable[P, R] | int | None = None,
    n: int | None = None,
) -> MaxTracesWrapper[P, R] | Callable[[Callable[P, R]], MaxTracesWrapper[P, R]]:
    """Decorate a function to assert that it is traced at most n times.

    Args:
        fn: The function to wrap, an integer max trace count, or None when parameterized with n.
        n: The maximum number of traces permitted.

    Returns:
        A MaxTracesWrapper callable or a decorator returning a MaxTracesWrapper.
    """
    if isinstance(fn, int):
        limit = fn

        def _int_decorator(target_fn: Callable[P, R]) -> MaxTracesWrapper[P, R]:
            """Wrap target_fn in a MaxTracesWrapper with positional integer limit.

            Args:
                target_fn: The function to decorate.

            Returns:
                MaxTracesWrapper instance.
            """
            return MaxTracesWrapper(target_fn, limit)

        return _int_decorator

    if fn is None:

        def _decorator(target_fn: Callable[P, R]) -> MaxTracesWrapper[P, R]:
            """Wrap target_fn in a MaxTracesWrapper with limit n.

            Args:
                target_fn: The function to decorate.

            Returns:
                MaxTracesWrapper instance.
            """
            return MaxTracesWrapper(target_fn, n)

        return _decorator
    return MaxTracesWrapper(fn, n)


def _is_pytree_result(val: R) -> TypeGuard[PyTree]:
    """Narrow a generic return value to a PyTree across all ecosystem leaf and container types.

    Args:
        val: Return value to inspect.

    Returns:
        True if val is a PyTree leaf or container.
    """
    return (
        isinstance(
            val,
            (
                int,
                float,
                complex,
                bool,
                str,
                type(None),
                ArrayProtocol,
                CompilerTensor,
                JaxNdarray,
                ShapeOnlyProtocol,
                TensorSpec,
                LogicalNode,
                VariableLeafProtocol,
                SharedMemoryArray,
                SharedMemoryArrayMetadata,
                MappableDataclassProtocol,
                NamedTuplePyTreeProtocol,
                StateMappingProtocol,
                list,
                tuple,
                Mapping,
            ),
        )
        or type(val) in _CUSTOM_PYTREE_NODES
    )


def _check_result_for_chexify(result: R, errors: frozenset[str]) -> AssertionError | None:
    """Inspect a function return value for LogicalNode or numerical ChexifyChecks violations.

    Args:
        result: Return value to inspect.
        errors: Enabled ChexifyChecks categories.

    Returns:
        An AssertionError if an enabled check category fails, or None.
    """
    leaves = tree_leaves(result) if _is_pytree_result(result) else []
    for leaf in leaves:
        if isinstance(leaf, LogicalNode):
            if (
                "div" in errors
                and leaf.op_type in ("Div", "TrueDiv", "FloorDiv", "Mod")
                and leaf.attributes.get("divisor") == 0
            ):
                return AssertionError(f"Chexify division by zero check failed in LogicalNode {leaf.id!r}")
            if "index" in errors and leaf.op_type in ("Gather", "Slice", "Index"):
                idx_attr = leaf.attributes.get("index")
                if (
                    isinstance(idx_attr, int)
                    and leaf.shape_metadata
                    and isinstance(leaf.shape_metadata[0], int)
                    and (idx_attr < -leaf.shape_metadata[0] or idx_attr >= leaf.shape_metadata[0])
                ):
                    return AssertionError(
                        f"Chexify index out-of-bounds check failed in LogicalNode {leaf.id!r}: "
                        f"index {idx_attr} for dimension {leaf.shape_metadata[0]}"
                    )
            val_attr = leaf.attributes.get("value")
            if "nan" in errors and isinstance(val_attr, (int, float)) and not math.isfinite(float(val_attr)):
                return AssertionError(f"Chexify NaN/Inf check failed on LogicalNode value: {val_attr!r}")

        if "nan" in errors:
            scalars: list[Scalar] = _extract_flat_scalars(leaf)
            for val in scalars:
                if isinstance(val, complex):
                    if not (math.isfinite(val.real) and math.isfinite(val.imag)):
                        return AssertionError(f"Chexify NaN/Inf check failed on value: {val!r}")
                else:
                    if not math.isfinite(float(val)):
                        return AssertionError(f"Chexify NaN/Inf check failed on value: {val!r}")
    return None


class ChexifiedFunction(Generic[P, R]):
    """Typed callable wrapper produced by @chexify supporting .wait_checks()."""

    def __init__(
        self,
        fn: Callable[P, R],
        async_check: bool = True,
        errors: frozenset[str] = ChexifyChecks.user,
    ) -> None:
        """Initialize a ChexifiedFunction wrapper.

        Args:
            fn: The underlying function containing Chex assertions.
            async_check: Whether assertion checks are queued asynchronously.
            errors: Set of check categories enabled for this function.
        """
        self._fn = fn
        self._async_check = async_check
        self._errors = errors
        self._pending_errors: list[AssertionError] = []
        with _STATE.lock:
            _STATE.pending_chexify_queues.append(self._pending_errors)
        self.__wrapped__ = fn
        functools.update_wrapper(self, fn)

    def wait_checks(self) -> None:
        """Block until all asynchronous assertion checks for this function complete.

        Raises:
            AssertionError: If any deferred assertion check failed during prior calls to this function.
        """
        if self._pending_errors:
            own_errors = list(self._pending_errors)
            err = own_errors[0]
            self._pending_errors.clear()
            with _STATE.lock:
                _STATE.pending_chexify_errors = [
                    e for e in _STATE.pending_chexify_errors if not any(e is own for own in own_errors)
                ]
            raise err

    def _record_or_raise(self, err: AssertionError) -> None:
        """Record an assertion error for async retrieval and raise.

        Args:
            err: The AssertionError to handle.

        Raises:
            AssertionError: Raised when synchronous or after recording for wait_checks().
        """
        if self._async_check:
            self._pending_errors.append(err)
            with _STATE.lock:
                _STATE.pending_chexify_errors.append(err)
        raise err

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Invoke the wrapped function with the provided arguments and active runtime checks.

        Args:
            *args: Positional arguments forwarded to the wrapped function.
            **kwargs: Keyword arguments forwarded to the wrapped function.

        Returns:
            The result of the wrapped function.

        Raises:
            AssertionError: If an enabled check category fails.
        """
        _STATE.in_chexify_context += 1
        try:
            result = self._fn(*args, **kwargs)
        except ZeroDivisionError as exc:
            if "div" in self._errors:
                err = AssertionError(f"Chexify division by zero check failed: {exc}")
                self._record_or_raise(err)
            raise
        except IndexError as exc:
            if "index" in self._errors:
                err = AssertionError(f"Chexify index out-of-bounds check failed: {exc}")
                self._record_or_raise(err)
            raise
        except AssertionError as exc:
            if "user" in self._errors:
                self._record_or_raise(exc)
            raise
        finally:
            _STATE.in_chexify_context -= 1

        if are_asserts_enabled():
            check_err = _check_result_for_chexify(result, self._errors)
            if check_err is not None:
                self._record_or_raise(check_err)
        return result


@overload
def chexify(
    fn: Callable[P, R],
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> ChexifiedFunction[P, R]: ...


@overload
def chexify(
    fn: None = None,
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> Callable[[Callable[P, R]], ChexifiedFunction[P, R]]: ...


def chexify(
    fn: Callable[P, R] | None = None,
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> ChexifiedFunction[P, R] | Callable[[Callable[P, R]], ChexifiedFunction[P, R]]:
    """Wrap a function to enable Chex value and runtime assertions inside compiled code.

    Args:
        fn: The function to wrap, or None when called with keyword arguments.
        async_check: Whether checks should be performed asynchronously.
        errors: Frozenset of ChexifyChecks categories to validate.

    Returns:
        A ChexifiedFunction wrapper or a decorator returning a ChexifiedFunction.
    """
    if fn is None:

        def _decorator(target_fn: Callable[P, R]) -> ChexifiedFunction[P, R]:
            """Wrap target_fn in a ChexifiedFunction.

            Args:
                target_fn: The function to wrap.

            Returns:
                ChexifiedFunction instance.
            """
            return ChexifiedFunction(target_fn, async_check=async_check, errors=errors)

        return _decorator
    return ChexifiedFunction(fn, async_check=async_check, errors=errors)


@overload
def with_jittable_assertions(
    fn: Callable[P, R],
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> ChexifiedFunction[P, R]: ...


@overload
def with_jittable_assertions(
    fn: None = None,
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> Callable[[Callable[P, R]], ChexifiedFunction[P, R]]: ...


def with_jittable_assertions(
    fn: Callable[P, R] | None = None,
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> ChexifiedFunction[P, R] | Callable[[Callable[P, R]], ChexifiedFunction[P, R]]:
    """Alias for chexify enabling assertions inside JIT-compiled functions.

    Args:
        fn: The function to wrap, or None when called with keyword arguments.
        async_check: Whether checks should be performed asynchronously.
        errors: Frozenset of ChexifyChecks categories to validate.

    Returns:
        A ChexifiedFunction wrapping fn or a decorator returning a ChexifiedFunction.
    """
    return chexify(fn, async_check=async_check, errors=errors)


@overload
def ify(
    fn: Callable[P, R],
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> ChexifiedFunction[P, R]: ...


@overload
def ify(
    fn: None = None,
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> Callable[[Callable[P, R]], ChexifiedFunction[P, R]]: ...


def ify(
    fn: Callable[P, R] | None = None,
    async_check: bool = True,
    errors: frozenset[str] = ChexifyChecks.user,
) -> ChexifiedFunction[P, R] | Callable[[Callable[P, R]], ChexifiedFunction[P, R]]:
    """Wrap a function to enable Chex assertions inside JIT-compiled code (canonical alias for chexify).

    Args:
        fn: The function to wrap, or None when called with keyword arguments.
        async_check: Whether checks should be performed asynchronously.
        errors: Frozenset of ChexifyChecks categories to validate.

    Returns:
        A ChexifiedFunction wrapping fn or a decorator returning a ChexifiedFunction.
    """
    return chexify(fn, async_check=async_check, errors=errors)


def block_until_chexify_assertions_complete() -> None:
    """Wait until all pending asynchronous chexify assertions have finished and raise if any failed.

    Raises:
        AssertionError: If any pending asynchronous chexify assertion failed.
    """
    with _STATE.lock:
        for q in _STATE.pending_chexify_queues:
            q.clear()
        if _STATE.pending_chexify_errors:
            err = _STATE.pending_chexify_errors.pop(0)
            _STATE.pending_chexify_errors.clear()
            raise err


def _scalar_from_numeric(val: PyTree | _TaylorJet) -> float:
    """Extract a Python float from a real scalar, _TaylorJet, Variable, or 1-element array.

    Args:
        val: Numeric value to convert.

    Returns:
        Extracted floating-point scalar value.

    Raises:
        ValueError: If val is an empty array with 0 elements.
        TypeError: If val is not a recognized scalar or array type.
        AssertionError: If val is a complex number with a non-zero imaginary component.
    """
    if isinstance(val, _TaylorJet):
        return float(val)
    if isinstance(val, VariableLeafProtocol):
        return _scalar_from_numeric(val.value)
    if isinstance(val, (ArrayProtocol, CompilerTensor, JaxNdarray, _TaylorJetArray)):
        items = _extract_flat_scalars(val)
        if not items:
            raise ValueError("Cannot extract scalar from empty array.")
        first = items[0]
        if isinstance(first, complex):
            if first.imag != 0.0:
                raise AssertionError(f"Expected real scalar for gradient evaluation, got complex {first!r}.")
            return float(first.real)
        return float(first)
    if isinstance(val, (int, float, bool)):
        return float(val)
    if isinstance(val, complex):
        if val.imag != 0.0:
            raise AssertionError(f"Expected real scalar for gradient evaluation, got complex {val!r}.")
        return float(val.real)
    raise TypeError(f"Expected numeric scalar or 1-element array, got {type(val).__name__}.")


def _pytree_to_float_list(tree: PyTree) -> list[float]:
    """Flatten a PyTree of Numeric/ZeroArray leaves into a flat list of floats.

    Args:
        tree: PyTree to flatten.

    Returns:
        Flat list of float values across all leaves and array elements.
    """
    out: list[float] = []
    for leaf in tree_leaves(tree):
        if isinstance(leaf, (ArrayProtocol, CompilerTensor, JaxNdarray, VariableLeafProtocol)):
            for item in _extract_flat_scalars(leaf):
                out.append(_scalar_from_numeric(item))
        else:
            out.append(_scalar_from_numeric(leaf))
    return out


def _rebuild_pytree_from_floats(template: PyTree, flat_values: Sequence[float]) -> PyTree:
    """Rebuild a PyTree from a flat sequence of floats matching template's leaf shapes.

    Args:
        template: Original PyTree providing container structure and array shapes.
        flat_values: Flat sequence of floats.

    Returns:
        Reconstructed PyTree with updated float values.
    """
    orig_leaves, treedef = tree_flatten(template)
    new_leaves: list[LeafValue] = []
    offset = 0
    for leaf in orig_leaves:
        if isinstance(leaf, JaxNdarray):
            count = int(leaf.size)
            chunk = list(flat_values[offset : offset + count])
            offset += count
            new_leaves.append(jnp.array(chunk).reshape(leaf.shape))
        elif isinstance(leaf, ArrayProtocol):
            count = leaf.size
            chunk = list(flat_values[offset : offset + count])
            offset += count
            dt = leaf.dtype if isinstance(leaf.dtype, (DType, str, type)) else str(leaf.dtype)
            new_leaves.append(ZeroArray(chunk, shape=leaf.shape, dtype=dt))
        else:
            val = flat_values[offset]
            offset += 1
            new_leaves.append(val)
    return tree_unflatten(treedef, new_leaves)


def _rebuild_pytree_with_jets(
    template: PyTree,
    seeded_elem_idx: int | None,
    order: int,
) -> PyTree:
    """Rebuild a PyTree with _TaylorJet and _TaylorJetArray leaves for forward-mode autodiff.

    Args:
        template: Original PyTree argument.
        seeded_elem_idx: Flat scalar index within template to seed as an active variable, or None.
        order: Taylor expansion order.

    Returns:
        PyTree containing _TaylorJet scalars and _TaylorJetArray leaves.
    """
    orig_leaves, treedef = tree_flatten(template)
    new_leaves: list[LeafValue] = []
    offset = 0
    for leaf in orig_leaves:
        if isinstance(leaf, (ArrayProtocol, CompilerTensor, JaxNdarray)):
            scalars = [_scalar_from_numeric(v) for v in _extract_flat_scalars(leaf)]
            jets: list[_TaylorJet] = []
            for idx_in_leaf, val in enumerate(scalars):
                global_idx = offset + idx_in_leaf
                if seeded_elem_idx is not None and global_idx == seeded_elem_idx:
                    jets.append(_TaylorJet.variable(val, order))
                else:
                    jets.append(_TaylorJet.constant(val, order))
            offset += len(scalars)
            new_leaves.append(_TaylorJetArray(jets, shape=_get_shape(leaf), dtype=_get_type(leaf)))
        else:
            val = _scalar_from_numeric(leaf)
            jet = (
                _TaylorJet.variable(val, order)
                if (seeded_elem_idx is not None and offset == seeded_elem_idx)
                else _TaylorJet.constant(val, order)
            )
            offset += 1
            new_leaves.append(jet)
    return tree_unflatten(treedef, new_leaves)


def _extract_jet_derivative_sum(out_tree: PyTree, order: int) -> float:
    """Extract the sum of n-th order derivatives across all _TaylorJet and _TaylorJetArray leaves.

    Args:
        out_tree: Output PyTree produced by evaluating f on jet arguments.
        order: Derivative order to extract.

    Returns:
        Sum of n-th order derivatives across all output leaves.
    """
    total = 0.0
    for leaf in tree_leaves(out_tree):
        if isinstance(leaf, _TaylorJet):
            total += leaf.derivative(order)
        elif isinstance(leaf, _TaylorJetArray):
            total += math.fsum(j.derivative(order) for j in leaf.jets)
    return total


def _compute_nth_fd_derivative(
    eval_fn: Callable[[float], float],
    x0: float,
    order: int,
    eps: float,
) -> float:
    """Recursively compute the n-th order central finite-difference derivative at x0.

    Args:
        eval_fn: Single-variable scalar function.
        x0: Point at which to evaluate the derivative.
        order: Order of differentiation (>= 1).
        eps: Step size for central finite differences.

    Returns:
        Approximate n-th derivative at x0.
    """
    if order == 1:
        return (eval_fn(x0 + eps) - eval_fn(x0 - eps)) / (2.0 * eps)
    plus_deriv = _compute_nth_fd_derivative(eval_fn, x0 + eps, order - 1, eps)
    minus_deriv = _compute_nth_fd_derivative(eval_fn, x0 - eps, order - 1, eps)
    return (plus_deriv - minus_deriv) / (2.0 * eps)


def _unwrap_traced_root_sum(
    out: JaxNdarray | CompilerTensor | float | int,
) -> JaxNdarray | CompilerTensor | float | int:
    """Unwrap a root full-tensor Sum node during active tracing so ir_grad seeds matching shape.

    Args:
        out: Traced output value from _jax_f.

    Returns:
        Unwrapped JaxNdarray if out is a root Sum node, otherwise out unchanged.
    """
    t = out._tensor if isinstance(out, JaxNdarray) else out
    if (
        global_tracing_state.is_tracing
        and global_tracing_state.active_graph is not None
        and isinstance(t, CompilerTensor)
        and isinstance(t.data, _ProxyTensor)
    ):
        node = global_tracing_state.active_graph.nodes[t.data.id]
        if node.op_type == "Sum" and node.attributes.get("axis") is None and len(node.inputs) == 1:
            in_id = node.inputs[0]
            in_node = global_tracing_state.active_graph.nodes[in_id]
            in_shape = tuple(int(d) for d in in_node.shape_metadata)
            return jnp.ndarray(
                CompilerTensor(
                    data=_ProxyTensor(id=in_id, shape=in_shape, dtype=t.dtype.value),
                    config=_CompilerTensorConfig(shape=in_shape, dtype=t.dtype, device=t.device),
                )
            )
    return out


def _build_unit_tangent_args(
    jax_args: Sequence[JaxPyTree],
    arg_idx: int,
    elem_idx: int,
) -> tuple[JaxPyTree, ...]:
    """Construct tangent PyTree arguments with 1.0 at (arg_idx, elem_idx) and 0.0 elsewhere.

    Args:
        jax_args: Primal JaxPyTree arguments.
        arg_idx: Target argument index.
        elem_idx: Flat scalar index within target argument.

    Returns:
        Tuple of tangent JaxPyTree arguments matching jax_args.
    """
    tangents: list[JaxPyTree] = []
    for idx, arg in enumerate(jax_args):
        assert isinstance(arg, JaxNdarray)
        n_elems = math.prod(arg.shape) if arg.shape else 1
        flat_t = [1.0 if (idx == arg_idx and k == elem_idx) else 0.0 for k in range(n_elems)]
        t_arr: JaxNdarray = jnp.array(flat_t).reshape(arg.shape) if arg.shape else jnp.array(flat_t[0])
        tangents.append(t_arr)
    return tuple(tangents)


def _compute_autodiff_nth_derivative(
    f: DifferentiableCallable,
    f_args: Sequence[PyTree],
    arg_idx: int,
    elem_idx: int,
    order: int,
) -> float:
    """Compute the exact n-th analytical derivative via zero_jax.grad/jvp or forward-mode _TaylorJet.

    Args:
        f: Function to differentiate.
        f_args: Sequence of input PyTree arguments.
        arg_idx: Index of the argument being differentiated.
        elem_idx: Flat scalar index within f_args[arg_idx].
        order: Differentiation order (>= 1).

    Returns:
        Analytical n-th derivative at the specified argument element.
    """
    if isinstance(f, (CustomJVPWrapper, CustomVJPWrapper)) or (
        order == 1 and any(isinstance(a, (JaxNdarray, CompilerTensor)) for a in f_args)
    ):
        jax_args = [
            jnp.array(_pytree_to_float_list(a)).reshape(_get_shape(a))
            if isinstance(a, (ArrayProtocol, CompilerTensor, JaxNdarray))
            else (jnp.array(float(a)) if isinstance(a, (int, float, bool)) else a)
            for a in f_args
        ]
        if isinstance(f, CustomVJPWrapper) and f.fwd is not None and f.bwd is not None:
            fwd_out = f.fwd(*jax_args)
            assert isinstance(fwd_out, tuple) and len(fwd_out) == 2
            primal_out, residuals = fwd_out
            assert isinstance(primal_out, JaxNdarray)
            cot = jnp.ones_like(primal_out)
            bwd_grads = f.bwd(residuals, cot)
            g_tuple = bwd_grads if isinstance(bwd_grads, tuple) else (bwd_grads,)
            grad_list = _pytree_to_float_list(g_tuple[arg_idx])
            return grad_list[elem_idx]

        orig_asarray = _optax_tree.asarray

        def _traced_optax_asarray(
            val: _OptaxArrayLike,
            dtype: _OptaxDTypeLike = None,
        ) -> CompilerTensor:
            """Preserve JaxNdarray._tensor during zero_jax autodiff tracing in zero_optax.

            Args:
                val: Input array-like value.
                dtype: Optional target dtype.

            Returns:
                Underlying CompilerTensor preserving ProxyTensor tracing nodes.
            """
            if isinstance(val, JaxNdarray):
                return val._tensor
            return orig_asarray(val, dtype=dtype)

        def _jax_f(
            *j_args: JaxPyTree,
            **j_kwargs: JaxPyTree,
        ) -> JaxPyTree:
            """Adapter forwarding zero_jax.grad and zero_jax.jvp arguments to f.

            Args:
                *j_args: Positional array/scalar arguments.
                **j_kwargs: Unused keyword arguments.

            Returns:
                Output array or scalar.
            """
            _ = j_kwargs
            pytree_j_args: list[PyTree] = [a for a in j_args if _is_pytree_result(a)]
            out = _invoke_differentiable(f, pytree_j_args)
            assert isinstance(out, (JaxNdarray, CompilerTensor, float, int))
            return _unwrap_traced_root_sum(out)

        with (
            unittest.mock.patch.object(_optax_tree, "asarray", new=_traced_optax_asarray),
            unittest.mock.patch.object(_optax_regression, "asarray", new=_traced_optax_asarray),
        ):
            if isinstance(f, CustomJVPWrapper):
                unit_tangents = _build_unit_tangent_args(jax_args, arg_idx, elem_idx)
                _, jvp_tangent = zero_jax.jvp(_jax_f, tuple(jax_args), unit_tangents)
                return sum(_pytree_to_float_list(jvp_tangent))
            grad_fn = zero_jax.grad(_jax_f, argnums=arg_idx)
            grad_out = grad_fn(*jax_args)
            unit_tangents = _build_unit_tangent_args(jax_args, arg_idx, elem_idx)
            _, _ = zero_jax.jvp(_jax_f, tuple(jax_args), unit_tangents)
            grad_list = _pytree_to_float_list(grad_out)
            return grad_list[elem_idx]

    jet_args: list[PyTree] = [
        _rebuild_pytree_with_jets(a, seeded_elem_idx=(elem_idx if idx == arg_idx else None), order=order)
        for idx, a in enumerate(f_args)
    ]
    jet_out = _invoke_differentiable(f, jet_args)
    return _extract_jet_derivative_sum(jet_out, order)


def _grad_to_float_lists(
    raw_grad: PyTree,
    num_args: int,
) -> list[list[float]]:
    """Normalize an analytical gradient output into a list of per-argument float lists.

    Args:
        raw_grad: Single PyTree gradient or sequence of per-argument PyTree gradients.
        num_args: Number of primal input arguments.

    Returns:
        List of flat float lists, one per differentiated argument.
    """
    if isinstance(raw_grad, (list, tuple)) and len(raw_grad) == num_args:
        return [_pytree_to_float_list(g) for g in raw_grad]
    return [_pytree_to_float_list(raw_grad)]


def assert_numerical_grads(
    f: DifferentiableCallable,
    f_args: Sequence[PyTree],
    order: int = 1,
    atol: float = 0.01,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
    **check_kwargs: float,
) -> None:
    """Assert that analytical and finite-difference numerical derivatives match within tolerance.

    Args:
        f: The scalar, tensor, or PyTree function to differentiate.
        f_args: Sequence of input PyTree arguments at which to evaluate derivatives.
        order: Order of differentiation (must be >= 1).
        atol: Absolute tolerance for gradient comparison.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.
        **check_kwargs: Optional extra tolerance parameters (e.g., eps, rtol).

    Raises:
        ValueError: If order is less than 1.
        Exception: If evaluated gradients are non-finite or mismatch analytical gradients.
    """
    if not are_asserts_enabled():
        return
    if order < 1:
        raise ValueError("Differentiation order must be at least 1.")
    if atol < 0.0:
        _raise_chex_error(
            f"Tolerance atol must be non-negative, got {atol}.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )

    eps = check_kwargs.get("eps", 1e-4)
    rtol = check_kwargs.get("rtol", 1e-2)
    base_out = _invoke_differentiable(f, f_args)
    base_scalars = _pytree_to_float_list(base_out)
    if any(not math.isfinite(v) for v in base_scalars):
        _raise_chex_error(
            "Function output contains non-finite values at base arguments.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )

    explicit_grads: list[list[float]] | None = None
    if isinstance(f, AnalyticalGradCallableProtocol):
        explicit_grads = _grad_to_float_lists(f.grad(*f_args), len(f_args))

    for arg_idx, arg in enumerate(f_args):
        arg_elems = _pytree_to_float_list(arg)
        for elem_idx, x0 in enumerate(arg_elems):

            def _eval_single(x_val: float, _a_idx: int = arg_idx, _e_idx: int = elem_idx) -> float:
                """Evaluate sum of outputs of f when perturbing a single scalar element.

                Args:
                    x_val: Perturbed value for element _e_idx of argument _a_idx.
                    _a_idx: Argument index.
                    _e_idx: Element index within the argument.

                Returns:
                    Scalar sum of output elements.
                """
                perturbed_elems = list(_pytree_to_float_list(f_args[_a_idx]))
                perturbed_elems[_e_idx] = x_val
                call_args = list(f_args)
                call_args[_a_idx] = _rebuild_pytree_from_floats(f_args[_a_idx], perturbed_elems)
                out_val = _invoke_differentiable(f, call_args)
                return math.fsum(_pytree_to_float_list(out_val))

            fd_grad = _compute_nth_fd_derivative(_eval_single, x0, order, eps)
            if not math.isfinite(fd_grad):
                _raise_chex_error(
                    f"Numerical gradient is non-finite ({fd_grad}) at arg {arg_idx}, index {elem_idx}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )

            if explicit_grads is not None and arg_idx < len(explicit_grads):
                ad_grad = explicit_grads[arg_idx][elem_idx]
            else:
                ad_grad = _compute_autodiff_nth_derivative(f, f_args, arg_idx, elem_idx, order)

            if not math.isfinite(ad_grad):
                _raise_chex_error(
                    f"Analytical gradient is non-finite ({ad_grad}) at arg {arg_idx}, index {elem_idx}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
            if abs(ad_grad - fd_grad) > atol + rtol * abs(fd_grad):
                _raise_chex_error(
                    f"Gradient mismatch at arg {arg_idx}[{elem_idx}]: "
                    f"analytical={ad_grad}, numerical={fd_grad}, atol={atol}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )


def _fake_pmap_impl(
    fun: Callable[P, R],
    axis_name: str | None = None,
    **kwargs: PyTree,
) -> Callable[P, R]:
    """Return a function configured for fake PMAP execution.

    Args:
        fun: The function to execute under fake PMAP.
        axis_name: Optional parallel axis name for vmap delegation.
        **kwargs: Optional PMAP configuration flags (e.g., jit_result, fake_parallel_axis).

    Returns:
        Wrapped callable executing fun locally without multi-device PMAP requirements.
    """
    use_parallel_axis = bool(kwargs.get("fake_parallel_axis", _STATE.fake_pmap_parallel_axis))
    use_jit = bool(kwargs.get("jit_result", _STATE.fake_pmap_jit_result))
    trace_key = _STATE.allocate_fn_key("fake_pmap")

    @functools.wraps(fun)
    def _pmap_wrapped(*args: P.args, **kwargs_call: P.kwargs) -> R:
        """Execute fun locally under fake PMAP semantics.

        Args:
            *args: Positional arguments.
            **kwargs_call: Keyword arguments.

        Returns:
            Result of fun(*args, **kwargs_call).
        """
        _ = (axis_name, use_parallel_axis)
        if use_jit and not _STATE.jit_disabled:
            pytree_args = [a for a in args if _is_pytree_result(a)]
            pytree_kwargs = {k: v for k, v in kwargs_call.items() if _is_pytree_result(v)}
            sig = _abstract_arg_signature(pytree_args, pytree_kwargs)
            with _STATE.lock:
                seen = _STATE.trace_seen_signatures.setdefault(trace_key, set())
                if sig not in seen:
                    seen.add(sig)
                    _STATE.trace_counts[trace_key] = _STATE.trace_counts.get(trace_key, 0) + 1
        return fun(*args, **kwargs_call)

    return _pmap_wrapped


class FakeExecutionContext(contextlib.ExitStack):
    """Context manager for mocking JIT and PMAP execution environments."""

    def __init__(
        self,
        enabled: bool = True,
        patch_jit: bool = False,
        patch_pmap: bool = False,
        jit_result: bool = False,
        ignore_axis_index_groups: bool = False,
        fake_parallel_axis: bool = False,
    ) -> None:
        """Initialize a FakeExecutionContext.

        Args:
            enabled: Whether patching is active within this context.
            patch_jit: Whether this context disables JIT tracing when active.
            patch_pmap: Whether this context replaces PMAP execution when active.
            jit_result: Whether faked PMAP should still record JIT traces.
            ignore_axis_index_groups: Whether to ignore axis_index_groups in faked PMAP.
            fake_parallel_axis: Whether faked PMAP should vmap over the leading axis.
        """
        super().__init__()
        self._enabled = enabled
        self._patch_jit = patch_jit
        self._patch_pmap = patch_pmap
        self._jit_result = jit_result
        self._ignore_axis_index_groups = ignore_axis_index_groups
        self._fake_parallel_axis = fake_parallel_axis
        self._active = False
        self._state_stack: list[tuple[bool, bool, bool, bool, bool]] = []

    @property
    def active(self) -> bool:
        """Return whether the fake execution context is currently active.

        Returns:
            True if started/entered, False otherwise.
        """
        return self._active

    def start(self) -> None:
        """Activate the fake execution context and patch zero_jax transformations."""
        self._state_stack.append(
            (
                _STATE.jit_disabled,
                _STATE.pmap_disabled,
                _STATE.fake_pmap_jit_result,
                _STATE.fake_pmap_ignore_groups,
                _STATE.fake_pmap_parallel_axis,
            )
        )
        self._active = self._enabled
        if self._enabled and self._patch_jit:
            _STATE.jit_disabled = True
            self.enter_context(zero_jax.disable_jit(True))
        if self._enabled and self._patch_pmap:
            _STATE.pmap_disabled = True
            _STATE.fake_pmap_jit_result = self._jit_result
            _STATE.fake_pmap_ignore_groups = self._ignore_axis_index_groups
            _STATE.fake_pmap_parallel_axis = self._fake_parallel_axis
            self.enter_context(unittest.mock.patch.object(zero_jax, "pmap", new=_fake_pmap_impl))

    def _restore_state(self) -> None:
        """Restore previous runtime state flags from the internal stack."""
        prev_jit, prev_pmap, prev_jit_res, prev_ignore, prev_par = self._state_stack.pop()
        if self._enabled and self._patch_jit:
            _STATE.jit_disabled = prev_jit
        if self._enabled and self._patch_pmap:
            _STATE.pmap_disabled = prev_pmap
            _STATE.fake_pmap_jit_result = prev_jit_res
            _STATE.fake_pmap_ignore_groups = prev_ignore
            _STATE.fake_pmap_parallel_axis = prev_par
        self._active = False

    def stop(self) -> None:
        """Deactivate the fake execution context and restore zero_jax transformations."""
        self._restore_state()
        super().__exit__(None, None, None)

    def __enter__(self) -> FakeExecutionContext:
        """Enter the fake execution context and activate patching.

        Returns:
            This FakeExecutionContext instance.
        """
        super().__enter__()
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the fake execution context and restore previous JIT/PMAP state.

        Args:
            exc_type: Exception type if raised inside the block.
            exc_val: Exception instance if raised inside the block.
            exc_tb: Traceback if raised inside the block.

        Returns:
            Boolean indicating whether any exception was suppressed.
        """
        self._restore_state()
        return bool(super().__exit__(exc_type, exc_val, exc_tb))


FakeContext = FakeExecutionContext


def fake_jit(enable_patching: bool = True) -> FakeExecutionContext:
    """Return a context manager that disables or fakes JIT compilation.

    Args:
        enable_patching: Whether JIT patching is enabled.

    Returns:
        A FakeExecutionContext instance.
    """
    return FakeExecutionContext(enabled=enable_patching, patch_jit=True)


def fake_pmap(
    enable_patching: bool = True,
    jit_result: bool = False,
    ignore_axis_index_groups: bool = False,
    fake_parallel_axis: bool = False,
) -> FakeExecutionContext:
    """Return a context manager that replaces PMAP parallelization with local execution.

    Args:
        enable_patching: Whether PMAP patching is enabled.
        jit_result: Whether to JIT-compile the faked PMAP function.
        ignore_axis_index_groups: Whether to ignore axis_index_groups in collectives.
        fake_parallel_axis: Whether to simulate a parallel axis via vmap semantics.

    Returns:
        A FakeExecutionContext instance.
    """
    return FakeExecutionContext(
        enabled=enable_patching,
        patch_jit=False,
        patch_pmap=True,
        jit_result=jit_result,
        ignore_axis_index_groups=ignore_axis_index_groups,
        fake_parallel_axis=fake_parallel_axis,
    )


def fake_pmap_and_jit(
    enable_pmap_patching: bool = True,
    enable_jit_patching: bool = True,
) -> FakeExecutionContext:
    """Return a context manager that fakes both PMAP and JIT transformations.

    Args:
        enable_pmap_patching: Whether to enable PMAP patching.
        enable_jit_patching: Whether to enable JIT patching.

    Returns:
        A FakeExecutionContext instance.
    """
    return FakeExecutionContext(
        enabled=enable_pmap_patching or enable_jit_patching,
        patch_jit=enable_jit_patching,
        patch_pmap=enable_pmap_patching,
    )


@contextlib.contextmanager
def restrict_backends(
    allowed: Sequence[str] | None = None,
    forbidden: Sequence[str] | None = None,
) -> Iterator[None]:
    """Context manager restricting compilation to allowed or non-forbidden backends.

    Args:
        allowed: Sequence of permitted backend names (cannot be combined with forbidden).
        forbidden: Sequence of disallowed backend names (cannot be combined with allowed).

    Yields:
        None while the backend restriction is active.

    Raises:
        ValueError: If neither or both of allowed and forbidden are specified.
    """
    if allowed is None and forbidden is None:
        raise ValueError("Must specify either 'allowed' or 'forbidden' backends.")
    if allowed is not None and forbidden is not None:
        raise ValueError("Cannot specify both 'allowed' and 'forbidden' backends.")

    prev_allowed = _STATE.allowed_backends
    prev_forbidden = _STATE.forbidden_backends
    _STATE.allowed_backends = frozenset(allowed) if allowed is not None else None
    _STATE.forbidden_backends = frozenset(forbidden) if forbidden is not None else None
    try:
        yield
    finally:
        _STATE.allowed_backends = prev_allowed
        _STATE.forbidden_backends = prev_forbidden


__all__ = [
    "AnalyticalGradCallableProtocol",
    "ChexifiedFunction",
    "FakeContext",
    "FakeExecutionContext",
    "MaxTracesWrapper",
    "are_asserts_enabled",
    "assert_max_traces",
    "assert_numerical_grads",
    "block_until_chexify_assertions_complete",
    "chexify",
    "clear_trace_counter",
    "disable_asserts",
    "enable_asserts",
    "fake_jit",
    "fake_pmap",
    "fake_pmap_and_jit",
    "get_num_devices_available",
    "ify",
    "restrict_backends",
    "set_n_cpu_devices",
    "with_jittable_assertions",
]
