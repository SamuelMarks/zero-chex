"""Assertion toggles, trace counters, chexify wrappers, and execution contexts for zero_chex."""

from __future__ import annotations

import contextlib
import functools
import math
import threading
from collections.abc import Callable, Iterator, Sequence
from types import TracebackType
from typing import Generic, Protocol, TypeVar, Union, overload, runtime_checkable

from typing_extensions import ParamSpec, TypeGuard

from zero_chex._src.tree_util import tree_leaves
from zero_chex.types import (
    ArrayProtocol,
    ChexifyChecks,
    Numeric,
    PyTree,
    Scalar,
    ShapeOnlyProtocol,
    ZeroArray,
)

P = ParamSpec("P")
R = TypeVar("R")


@runtime_checkable
class AnalyticalGradCallableProtocol(Protocol):
    """Protocol for differentiable callables that expose an explicit analytical `grad` callable."""

    def __call__(self, *args: Numeric) -> Numeric:
        """Evaluate the primal function at the given numeric arguments.

        Args:
            *args: Numeric input arguments.

        Returns:
            Numeric function output.
        """
        ...

    def grad(self, *args: Numeric) -> Numeric | Sequence[Numeric]:
        """Evaluate the analytical gradient at the given numeric arguments.

        Args:
            *args: Numeric input arguments.

        Returns:
            Analytical gradient or sequence of gradients per argument.
        """
        ...


class VariadicNumericCallableProtocol(Protocol):
    """Protocol for callables accepting variadic positional Numeric arguments."""

    def __call__(self, *args: Numeric) -> Numeric:
        """Evaluate the function on positional numeric arguments.

        Args:
            *args: Numeric input arguments.

        Returns:
            Numeric result.
        """
        ...


DifferentiableCallable = Union[
    AnalyticalGradCallableProtocol,
    VariadicNumericCallableProtocol,
    Callable[[Numeric], Numeric],
    Callable[[Numeric, Numeric], Numeric],
    Callable[[Numeric, Numeric, Numeric], Numeric],
]


class _AssertionRuntimeState:
    """Thread-safe container for global assertion state, trace counters, and device configs."""

    def __init__(self) -> None:
        """Initialize the default assertion runtime state."""
        self.enabled: bool = True
        self.lock: threading.Lock = threading.Lock()
        self.trace_counts: dict[int, int] = {}
        self.n_cpu_devices: int = 1
        self.n_gpu_devices: int = 1
        self.n_tpu_devices: int = 1
        self.allowed_backends: frozenset[str] | None = None
        self.forbidden_backends: frozenset[str] | None = None
        self.jit_disabled: bool = False
        self.pmap_disabled: bool = False
        self.pending_chexify_errors: list[AssertionError] = []


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
        self._fn_id = id(fn)
        self.__wrapped__ = fn

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
                current = _STATE.trace_counts.get(self._fn_id, 0) + 1
                _STATE.trace_counts[self._fn_id] = current
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
    return isinstance(val, (int, float, complex, bool, str, type(None), ArrayProtocol, list, tuple, dict))


def _check_result_for_nan(result: R) -> AssertionError | None:
    """Inspect a function return value for NaN or Inf leaves when ChexifyChecks.nan is active.

    Args:
        result: Return value to inspect.

    Returns:
        An AssertionError if a NaN/Inf value is detected, or None.
    """
    leaves = tree_leaves(result) if _is_pytree_result(result) else []
    for leaf in leaves:
        scalars: list[Scalar] = (
            list(leaf.to_list())
            if isinstance(leaf, ArrayProtocol)
            else ([leaf] if isinstance(leaf, (int, float, complex, bool)) else [])
        )
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

    def wait_checks(self) -> None:
        """Block until all asynchronous assertion checks for this function complete.

        Raises:
            AssertionError: If any deferred assertion check failed during prior calls.
        """
        if self._pending_errors:
            err = self._pending_errors.pop(0)
            self._pending_errors.clear()
            raise err

    def _record_or_raise(self, err: AssertionError) -> None:
        """Record an assertion error for async retrieval or raise immediately.

        Args:
            err: The AssertionError to handle.

        Raises:
            AssertionError: Always raised after recording if async_check is True or False.
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


def with_jittable_assertions(
    fn: Callable[P, R],
    async_check: bool = True,
) -> ChexifiedFunction[P, R]:
    """Alias for chexify enabling assertions inside JIT-compiled functions.

    Args:
        fn: The function to wrap.
        async_check: Whether checks should be performed asynchronously.

    Returns:
        A ChexifiedFunction wrapping fn.
    """
    return ChexifiedFunction(fn, async_check=async_check)


def block_until_chexify_assertions_complete() -> None:
    """Wait until all pending asynchronous chexify assertions have finished."""
    with _STATE.lock:
        _STATE.pending_chexify_errors.clear()


def _scalar_from_numeric(val: Numeric) -> float:
    """Extract a Python float from a real scalar or 1-element ArrayProtocol.

    Args:
        val: Numeric value to convert.

    Returns:
        Extracted floating-point scalar value.

    Raises:
        AssertionError: If val is a complex number with a non-zero imaginary component.
    """
    if isinstance(val, ArrayProtocol):
        items = val.to_list()
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


def _numeric_to_float_list(val: Numeric) -> list[float]:
    """Convert a Numeric scalar or ArrayProtocol into a flat list of floats.

    Args:
        val: Numeric value to flatten.

    Returns:
        List of floating-point values.
    """
    if isinstance(val, ArrayProtocol):
        return [_scalar_from_numeric(item) for item in val.to_list()]
    return [_scalar_from_numeric(val)]


def _rebuild_numeric(template: Numeric, values: Sequence[float]) -> Numeric:
    """Rebuild a Numeric scalar or ZeroArray matching the template's shape.

    Args:
        template: Original Numeric argument providing shape and type metadata.
        values: Flat sequence of float values.

    Returns:
        Reconstructed ZeroArray or float scalar.
    """
    if isinstance(template, ArrayProtocol):
        return ZeroArray(list(values), shape=template.shape, dtype=template.dtype)
    return values[0] if values else 0.0


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


def _compute_polynomial_analytic_deriv(
    eval_fn: Callable[[float], float],
    x0: float,
    order: int,
    base_eps: float,
) -> float:
    """Compute a high-precision Richardson-extrapolated reference derivative at x0.

    Args:
        eval_fn: Single-variable scalar function.
        x0: Evaluation point.
        order: Derivative order.
        base_eps: Base step size.

    Returns:
        Richardson-extrapolated derivative estimate.
    """
    h1 = max(base_eps * 0.5, 1e-5)
    h2 = h1 * 0.5
    d1 = _compute_nth_fd_derivative(eval_fn, x0, order, h1)
    d2 = _compute_nth_fd_derivative(eval_fn, x0, order, h2)
    return (4.0 * d2 - d1) / 3.0


def _grad_to_float_lists(
    raw_grad: Numeric | Sequence[Numeric],
) -> list[list[float]]:
    """Normalize an analytical gradient output into a list of per-argument float lists.

    Args:
        raw_grad: Single Numeric gradient or sequence of per-argument Numeric gradients.

    Returns:
        List of flat float lists, one per differentiated argument.
    """
    if isinstance(raw_grad, (int, float, complex, bool, ArrayProtocol, ShapeOnlyProtocol)):
        return [_numeric_to_float_list(raw_grad)]
    return [_numeric_to_float_list(g) for g in raw_grad]


def assert_numerical_grads(
    f: DifferentiableCallable,
    f_args: Sequence[Numeric],
    order: int = 1,
    atol: float = 0.01,
    **check_kwargs: float,
) -> None:
    """Assert that analytical and finite-difference numerical derivatives match within tolerance.

    Args:
        f: The scalar or tensor function to differentiate.
        f_args: Sequence of input arguments at which to evaluate derivatives.
        order: Order of differentiation (must be >= 1).
        atol: Absolute tolerance for gradient comparison.
        **check_kwargs: Optional extra tolerance parameters (e.g., eps, rtol).

    Raises:
        ValueError: If order is less than 1.
        AssertionError: If evaluated gradients are non-finite or mismatch analytical gradients.
    """
    if not _STATE.enabled:
        return
    if order < 1:
        raise ValueError("Differentiation order must be at least 1.")
    if atol < 0.0:
        raise AssertionError(f"Tolerance atol must be non-negative, got {atol}.")

    eps = check_kwargs.get("eps", 1e-4)
    rtol = check_kwargs.get("rtol", 1e-2)
    base_out = functools.partial(f, *f_args)()
    base_scalars = _numeric_to_float_list(base_out)
    if any(not math.isfinite(v) for v in base_scalars):
        raise AssertionError("Function output contains non-finite values at base arguments.")

    explicit_grads: list[list[float]] | None = None
    if isinstance(f, AnalyticalGradCallableProtocol):
        explicit_grads = _grad_to_float_lists(f.grad(*f_args))

    for arg_idx, arg in enumerate(f_args):
        arg_elems = _numeric_to_float_list(arg)
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
                perturbed_elems = list(_numeric_to_float_list(f_args[_a_idx]))
                perturbed_elems[_e_idx] = x_val
                call_args = list(f_args)
                call_args[_a_idx] = _rebuild_numeric(f_args[_a_idx], perturbed_elems)
                out_val = functools.partial(f, *call_args)()
                return sum(_numeric_to_float_list(out_val))

            fd_grad = _compute_nth_fd_derivative(_eval_single, x0, order, eps)
            if not math.isfinite(fd_grad):
                raise AssertionError(
                    f"Numerical gradient is non-finite ({fd_grad}) at arg {arg_idx}, index {elem_idx}."
                )

            if explicit_grads is not None and arg_idx < len(explicit_grads):
                ad_grad = explicit_grads[arg_idx][elem_idx]
            else:
                ad_grad = _compute_polynomial_analytic_deriv(_eval_single, x0, order, eps)

            if not math.isfinite(ad_grad):
                raise AssertionError(
                    f"Analytical gradient is non-finite ({ad_grad}) at arg {arg_idx}, index {elem_idx}."
                )
            if abs(ad_grad - fd_grad) > atol + rtol * abs(fd_grad):
                raise AssertionError(
                    f"Gradient mismatch at arg {arg_idx}[{elem_idx}]: "
                    f"analytical={ad_grad}, numerical={fd_grad}, atol={atol}."
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
        self._prev_jit_disabled = False
        self._prev_pmap_disabled = False

    @property
    def active(self) -> bool:
        """Return whether the fake execution context is currently active.

        Returns:
            True if started/entered, False otherwise.
        """
        return self._active

    def start(self) -> None:
        """Activate the fake execution context manually."""
        self._prev_jit_disabled = _STATE.jit_disabled
        self._prev_pmap_disabled = _STATE.pmap_disabled
        self._active = self._enabled
        if self._enabled and self._patch_jit:
            _STATE.jit_disabled = True
        if self._enabled and self._patch_pmap:
            _STATE.pmap_disabled = True

    def stop(self) -> None:
        """Deactivate the fake execution context manually."""
        if self._enabled and self._patch_jit:
            _STATE.jit_disabled = self._prev_jit_disabled
        if self._enabled and self._patch_pmap:
            _STATE.pmap_disabled = self._prev_pmap_disabled
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
        The callable fun.
    """
    _ = kwargs
    return fun


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
    _ = (jit_result, ignore_axis_index_groups, fake_parallel_axis)
    return FakeExecutionContext(enabled=enable_patching, patch_pmap=True)


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
    "_STATE",
    "_fake_pmap_impl",
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
    "restrict_backends",
    "set_n_cpu_devices",
    "with_jittable_assertions",
]
