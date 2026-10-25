"""Assertion toggles, trace counters, chexify wrappers, and execution contexts for zero_chex."""

from __future__ import annotations

import contextlib
import functools
import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Generic, Protocol, TypeVar, Union, overload, runtime_checkable

from ml_switcheroo_ir import TensorSpec
from typing_extensions import ParamSpec, TypeGuard

from zero_chex._src.tree_util import tree_flatten, tree_leaves, tree_unflatten
from zero_chex.types import (
    ArrayProtocol,
    ChexifyChecks,
    DType,
    LeafValue,
    MappableDataclassProtocol,
    Numeric,
    PyTree,
    Scalar,
    ShapeOnlyProtocol,
    ZeroArray,
    _extract_flat_scalars,
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

    def __float__(self) -> float:
        """Return the primal 0-th order scalar value.

        Returns:
            Primal float value.
        """
        return self.coeffs[0]


@runtime_checkable
class AnalyticalGradCallableProtocol(Protocol):
    """Protocol for differentiable callables that expose an explicit analytical `grad` callable."""

    def __call__(self, *args: Numeric) -> Numeric:
        """Evaluate the primal function at the given arguments.

        Args:
            *args: Input arguments.

        Returns:
            Function output.
        """
        ...

    def grad(self, *args: Numeric) -> Numeric | Sequence[Numeric]:
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
    return isinstance(val, (int, float, complex, bool, _TaylorJet, ArrayProtocol, ShapeOnlyProtocol, TensorSpec))


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


def _abstract_arg_signature(args: tuple[PyTree, ...], kwargs: Mapping[str, PyTree]) -> str:
    """Compute a deterministic abstract shape/dtype/structure signature for JIT trace caching.

    Args:
        args: Positional call arguments.
        kwargs: Keyword call arguments.

    Returns:
        String key representing the abstract signature of (args, kwargs).
    """
    parts: list[str] = []
    for item in args:
        if isinstance(item, ArrayProtocol):
            parts.append(f"arr:{item.shape}:{item.dtype}")
        elif isinstance(item, ShapeOnlyProtocol):
            parts.append(f"shape:{item.shape}")
        elif isinstance(item, (list, tuple, Mapping, MappableDataclassProtocol)):
            leaves, treedef = tree_flatten(item)
            leaf_sigs = [
                f"arr:{lf.shape}:{lf.dtype}" if isinstance(lf, ArrayProtocol) else f"val:{type(lf).__name__}"
                for lf in leaves
            ]
            parts.append(f"tree:{treedef}:{leaf_sigs}")
        else:
            parts.append(f"scalar:{type(item).__name__}")
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if isinstance(v, ArrayProtocol):
            parts.append(f"kw:{k}:arr:{v.shape}:{v.dtype}")
        else:
            parts.append(f"kw:{k}:{type(v).__name__}:{v!r}")
    return "|".join(parts)


class _AssertionRuntimeState:
    """Thread-safe container for global assertion state, trace counters, and device configs."""

    def __init__(self) -> None:
        """Initialize the default assertion runtime state."""
        self.enabled: bool = True
        self.lock: threading.Lock = threading.Lock()
        self.trace_counts: dict[str, int] = {}
        self.trace_seen_signatures: dict[str, set[str]] = {}
        self._next_fn_token: int = 0
        self.n_cpu_devices: int = 1
        self.n_gpu_devices: int = 1
        self.n_tpu_devices: int = 1
        self.allowed_backends: frozenset[str] | None = None
        self.forbidden_backends: frozenset[str] | None = None
        self.jit_disabled: bool = False
        self.pmap_disabled: bool = False
        self.pending_chexify_errors: list[AssertionError] = []

    def allocate_fn_key(self, fn_name: str) -> str:
        """Allocate a collision-free trace key independent of garbage-collected object ids.

        Args:
            fn_name: Function name for diagnostics.

        Returns:
            Unique string key for tracking trace counts.
        """
        with self.lock:
            self._next_fn_token += 1
            return f"{fn_name}#{self._next_fn_token}"


_STATE = _AssertionRuntimeState()


def are_asserts_enabled() -> bool:
    """Return whether Chex assertions are currently enabled globally.

    Returns:
        True if assertions are enabled, False if disabled via disable_asserts().
    """
    return _STATE.enabled


def disable_asserts() -> None:
    """Disable all Chex assertions globally."""
    _STATE.enabled = False


def enable_asserts() -> None:
    """Enable all Chex assertions globally."""
    _STATE.enabled = True


def clear_trace_counter() -> None:
    """Reset all recorded function trace counters and pending chexify errors."""
    with _STATE.lock:
        _STATE.trace_counts.clear()
        _STATE.trace_seen_signatures.clear()
        _STATE.pending_chexify_errors.clear()
        _STATE.jit_disabled = False
        _STATE.pmap_disabled = False


class MaxTracesWrapper(Generic[P, R]):
    """Typed callable wrapper enforcing an upper bound on function trace/invocation counts."""

    def __init__(self, fn: Callable[P, R], max_traces: int | None) -> None:
        """Initialize a MaxTracesWrapper around a target function.

        Args:
            fn: The callable to monitor.
            max_traces: Maximum allowed number of invocations before raising AssertionError.
        """
        self._fn = fn
        self._max_traces = max_traces
        self._key = _STATE.allocate_fn_key("max_traces")
        self.__wrapped__ = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the wrapped function and increment its trace counter.

        Args:
            *args: Positional arguments forwarded to the wrapped function.
            **kwargs: Keyword arguments forwarded to the wrapped function.

        Returns:
            The return value of the wrapped function.

        Raises:
            AssertionError: If assertions are enabled and trace count exceeds max_traces.
        """
        if not _STATE.jit_disabled:
            with _STATE.lock:
                current = _STATE.trace_counts.get(self._key, 0) + 1
                _STATE.trace_counts[self._key] = current
                if _STATE.enabled and self._max_traces is not None and current > self._max_traces:
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
    """Narrow a generic return value to a PyTree.

    Args:
        val: Return value to inspect.

    Returns:
        True if val is a PyTree leaf or container.
    """
    return isinstance(
        val,
        (
            int,
            float,
            complex,
            bool,
            str,
            type(None),
            ArrayProtocol,
            ShapeOnlyProtocol,
            TensorSpec,
            MappableDataclassProtocol,
            list,
            tuple,
            Mapping,
        ),
    )


def _check_result_for_nan(result: R) -> AssertionError | None:
    """Inspect a function return value for NaN or Inf leaves when ChexifyChecks.nan is active.

    Args:
        result: Return value to inspect.

    Returns:
        An AssertionError if a NaN/Inf value is detected, or None.
    """
    leaves = tree_leaves(result) if _is_pytree_result(result) else []
    for leaf in leaves:
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
        self.__wrapped__ = fn
        functools.update_wrapper(self, fn)

    def wait_checks(self) -> None:
        """Block until all asynchronous assertion checks for this function complete.

        Raises:
            AssertionError: If any deferred assertion check failed during prior calls.
        """
        if self._pending_errors:
            err = self._pending_errors.pop(0)
            self._pending_errors.clear()
            with _STATE.lock:
                _STATE.pending_chexify_errors.clear()
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

        if "nan" in self._errors and _STATE.enabled:
            nan_err = _check_result_for_nan(result)
            if nan_err is not None:
                self._record_or_raise(nan_err)
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
        if _STATE.pending_chexify_errors:
            err = _STATE.pending_chexify_errors.pop(0)
            _STATE.pending_chexify_errors.clear()
            raise err


def _scalar_from_numeric(val: PyTree | _TaylorJet) -> float:
    """Extract a Python float from a real scalar, _TaylorJet, or 1-element ArrayProtocol.

    Args:
        val: Numeric value to convert.

    Returns:
        Extracted floating-point scalar value.

    Raises:
        AssertionError: If val is a complex number with a non-zero imaginary component.
    """
    if isinstance(val, _TaylorJet):
        return float(val)
    if isinstance(val, ArrayProtocol):
        items = _extract_flat_scalars(val)
        first = items[0] if items else 0.0
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
    return 0.0


def _pytree_to_float_list(tree: PyTree) -> list[float]:
    """Flatten a PyTree of Numeric/ZeroArray leaves into a flat list of floats.

    Args:
        tree: PyTree to flatten.

    Returns:
        Flat list of float values across all leaves and array elements.
    """
    out: list[float] = []
    for leaf in tree_leaves(tree):
        if isinstance(leaf, ArrayProtocol):
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
        if isinstance(leaf, ArrayProtocol):
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


def _compute_autodiff_nth_derivative(
    f: DifferentiableCallable,
    f_args: Sequence[PyTree],
    arg_idx: int,
    elem_idx: int,
    x0: float,
    order: int,
    eps: float,
) -> float:
    """Compute the exact n-th analytical derivative via forward-mode _TaylorJet propagation.

    If f is a scalar function supporting algebraic operations on _TaylorJet (e.g., polynomials,
    rational functions), this evaluates f on a seeded _TaylorJet and extracts the exact n-th
    Taylor coefficient without finite differences. For functions that require ZeroArray or
    non-jet types, falls back to high-order Richardson extrapolation at distinct step scale.

    Args:
        f: Function to differentiate.
        f_args: Sequence of input PyTree arguments.
        arg_idx: Index of the argument being differentiated.
        elem_idx: Flat scalar index within f_args[arg_idx].
        x0: Base scalar value.
        order: Differentiation order (>= 1).
        eps: Step size fallback for non-jet array functions.

    Returns:
        Analytical n-th derivative at x0.
    """
    target_arg = f_args[arg_idx]
    if isinstance(target_arg, (int, float, bool)) and all(isinstance(a, (int, float, bool)) for a in f_args):
        try:
            jet_args: list[PyTree] = [
                _TaylorJet.variable(x0, order)
                if idx == arg_idx
                else _TaylorJet.constant(_scalar_from_numeric(a), order)
                for idx, a in enumerate(f_args)
            ]
            jet_out = _invoke_differentiable(f, jet_args)
            return jet_out.derivative(order) if isinstance(jet_out, _TaylorJet) else 0.0
        except (TypeError, ValueError, AttributeError):
            pass

    def _eval_single(x_val: float) -> float:
        """Evaluate sum of output leaves when perturbing scalar element elem_idx of arg_idx.

        Args:
            x_val: Perturbed scalar value.

        Returns:
            Scalar sum of output leaves.
        """
        perturbed_elems = list(_pytree_to_float_list(f_args[arg_idx]))
        perturbed_elems[elem_idx] = x_val
        call_args = list(f_args)
        call_args[arg_idx] = _rebuild_pytree_from_floats(f_args[arg_idx], perturbed_elems)
        out_val = _invoke_differentiable(f, call_args)
        return math.fsum(_pytree_to_float_list(out_val))

    h1 = max(eps * 0.5, 1e-5)
    h2 = h1 * 0.5
    d1 = _compute_nth_fd_derivative(_eval_single, x0, order, h1)
    d2 = _compute_nth_fd_derivative(_eval_single, x0, order, h2)
    return (4.0 * d2 - d1) / 3.0


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


def _raise_grad_error(
    default_msg: str,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Format and raise an assertion error for assert_numerical_grads.

    Args:
        default_msg: Default assertion failure message.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include default_msg alongside custom_message.
        exception_type: Exception class to instantiate and raise.

    Raises:
        Exception: An instance of exception_type with the formatted message.
    """
    if custom_message is not None:
        formatted_custom = (
            custom_message.format(*custom_message_format_vars) if custom_message_format_vars else custom_message
        )
        final_msg = (
            f"{default_msg} [{formatted_custom}]"
            if include_default_message
            else f"[Chex] Assertion failed: {formatted_custom}"
        )
    else:
        final_msg = default_msg
    raise exception_type(final_msg)


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
    if not _STATE.enabled:
        return
    if order < 1:
        raise ValueError("Differentiation order must be at least 1.")
    if atol < 0.0:
        _raise_grad_error(
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
        _raise_grad_error(
            "Function output contains non-finite values at base arguments.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )

    explicit_grads: list[list[float]] | None = None
    if isinstance(f, AnalyticalGradCallableProtocol):
        num_f_args: list[Numeric] = [
            a if isinstance(a, (int, float, complex, bool, ArrayProtocol)) else 0.0 for a in f_args
        ]
        explicit_grads = _grad_to_float_lists(f.grad(*num_f_args), len(f_args))

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
                _raise_grad_error(
                    f"Numerical gradient is non-finite ({fd_grad}) at arg {arg_idx}, index {elem_idx}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )

            if explicit_grads is not None and arg_idx < len(explicit_grads):
                ad_grad = explicit_grads[arg_idx][elem_idx]
            else:
                ad_grad = _compute_autodiff_nth_derivative(f, f_args, arg_idx, elem_idx, x0, order, eps)

            if not math.isfinite(ad_grad):
                _raise_grad_error(
                    f"Analytical gradient is non-finite ({ad_grad}) at arg {arg_idx}, index {elem_idx}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
            if abs(ad_grad - fd_grad) > atol + rtol * abs(fd_grad):
                _raise_grad_error(
                    f"Gradient mismatch at arg {arg_idx}[{elem_idx}]: "
                    f"analytical={ad_grad}, numerical={fd_grad}, atol={atol}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )


class FakeExecutionContext(contextlib.ExitStack):
    """Context manager for mocking JIT and PMAP execution environments."""

    def __init__(
        self,
        enabled: bool = True,
        patch_jit: bool = False,
        patch_pmap: bool = False,
    ) -> None:
        """Initialize a FakeExecutionContext.

        Args:
            enabled: Whether patching is active within this context.
            patch_jit: Whether this context disables JIT tracing when active.
            patch_pmap: Whether this context disables PMAP execution when active.
        """
        super().__init__()
        self._enabled = enabled
        self._patch_jit = patch_jit
        self._patch_pmap = patch_pmap
        self._active = False
        self._state_stack: list[tuple[bool, bool]] = []

    @property
    def active(self) -> bool:
        """Return whether the fake execution context is currently active.

        Returns:
            True if started/entered, False otherwise.
        """
        return self._active

    def start(self) -> None:
        """Activate the fake execution context manually."""
        self._state_stack.append((_STATE.jit_disabled, _STATE.pmap_disabled))
        self._active = self._enabled
        if self._enabled and self._patch_jit:
            _STATE.jit_disabled = True
        if self._enabled and self._patch_pmap:
            _STATE.pmap_disabled = True

    def stop(self) -> None:
        """Deactivate the fake execution context manually."""
        prev_jit, prev_pmap = self._state_stack.pop()
        if self._enabled and self._patch_jit:
            _STATE.jit_disabled = prev_jit
        if self._enabled and self._patch_pmap:
            _STATE.pmap_disabled = prev_pmap
        self._active = False

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
        self.stop()
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


def _fake_pmap_impl(
    fun: Callable[P, R],
    **kwargs: bool | int | str | None,
) -> Callable[P, R]:
    """Return a function configured for fake PMAP execution.

    Args:
        fun: The function to execute under fake PMAP.
        **kwargs: Optional PMAP configuration flags.

    Returns:
        Wrapped callable executing fun locally without multi-device PMAP requirements.
    """
    _ = kwargs

    @functools.wraps(fun)
    def _pmap_wrapped(*args: P.args, **kwargs_call: P.kwargs) -> R:
        """Execute fun locally under fake PMAP semantics.

        Args:
            *args: Positional arguments.
            **kwargs_call: Keyword arguments.

        Returns:
            Result of fun(*args, **kwargs_call).
        """
        return fun(*args, **kwargs_call)

    return _pmap_wrapped


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
    _ = (ignore_axis_index_groups, fake_parallel_axis)
    return FakeExecutionContext(
        enabled=enable_patching,
        patch_jit=False if jit_result else False,
        patch_pmap=True,
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


def set_n_cpu_devices(n: int | None = None) -> None:
    """Configure the number of simulated host CPU devices.

    Args:
        n: Number of CPU devices to simulate (defaults to 1 if None).

    Raises:
        ValueError: If n is provided and is less than 1.
    """
    if n is not None and n < 1:
        raise ValueError(f"Number of CPU devices must be >= 1, got {n}.")
    _STATE.n_cpu_devices = n if n is not None else 1


def get_num_devices_available(platform: str) -> int:
    """Return the simulated number of available devices for a given platform.

    Args:
        platform: Hardware platform identifier ('cpu', 'gpu', or 'tpu').

    Returns:
        Number of simulated devices available on that platform.

    Raises:
        ValueError: If platform is not one of 'cpu', 'gpu', or 'tpu'.
    """
    if platform == "cpu":
        return _STATE.n_cpu_devices
    if platform == "gpu":
        return _STATE.n_gpu_devices
    if platform == "tpu":
        return _STATE.n_tpu_devices
    raise ValueError(f"Unknown device type: {platform}")


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
