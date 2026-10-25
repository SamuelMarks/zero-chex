"""Device availability, placement, host residence, and sharding assertions for zero_chex."""

from __future__ import annotations

from collections.abc import Sequence

from zero_chex._src.tree_util import tree_leaves
from zero_chex.assertions import _raise_chex_error
from zero_chex.mocks import _STATE, are_asserts_enabled, get_num_devices_available
from zero_chex.types import (
    DeviceProtocol,
    DevicesMethodProtocol,
    LeafValue,
    PyTree,
    Scalar,
    ShardingAttrProtocol,
    ShardingProtocol,
)


def _check_backend_permitted(backend: str | None, default_platform: str) -> None:
    """Validate that a target backend/platform is permitted under active restrict_backends.

    Args:
        backend: Explicit backend name or None.
        default_platform: Fallback platform name if backend is None.

    Raises:
        ValueError: If backend is not a recognized platform ('cpu', 'gpu', 'tpu').
        AssertionError: If the target backend is forbidden or excluded by restrict_backends.
    """
    target = backend if backend is not None else default_platform
    if target not in ("cpu", "gpu", "tpu"):
        raise ValueError(f"Unknown backend: {target}")
    if _STATE.forbidden_backends is not None and target in _STATE.forbidden_backends:
        raise AssertionError(f"Backend {target!r} is forbidden by restrict_backends.")
    if _STATE.allowed_backends is not None and target not in _STATE.allowed_backends:
        raise AssertionError(f"Backend {target!r} is not in allowed backends {_STATE.allowed_backends!r}.")


def _num_devices_available(platform: str, backend: str | None = None) -> int:
    """Query the number of available simulated devices for a given platform.

    Args:
        platform: Platform name ('cpu', 'gpu', or 'tpu').
        backend: Optional backend filter name.

    Returns:
        Integer count of available devices on the platform.

    Raises:
        ValueError: If platform is unknown.
    """
    if backend is not None:
        _check_backend_permitted(backend, platform)
    return get_num_devices_available(platform)


def assert_devices_available(
    n: int,
    devtype: str,
    backend: str | None = None,
    not_less_than: bool = False,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that n devices of devtype are available on the specified backend.

    Args:
        n: Expected number of available devices.
        devtype: Device platform string ('cpu', 'gpu', or 'tpu').
        backend: Optional backend identifier (defaults to devtype).
        not_less_than: If True, asserts available >= n; otherwise available == n.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        ValueError: If devtype or backend is not a recognized platform.
        Exception: If the available device count does not satisfy the condition.
    """
    if not are_asserts_enabled():
        return
    dtype_lower = devtype.lower()
    try:
        _check_backend_permitted(backend, dtype_lower)
    except AssertionError as exc:
        _raise_chex_error(
            str(exc),
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )
    available = _num_devices_available(dtype_lower)
    if not_less_than:
        if available < n:
            _raise_chex_error(
                f"Only {available} < {n} {dtype_lower.upper()}s available.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
    else:
        if available != n:
            _raise_chex_error(
                f"No {n} {dtype_lower.upper()}s available (found {available}).",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )


def assert_gpu_available(
    backend: str | None = None,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that at least one GPU device is available.

    Args:
        backend: Optional backend name to query.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If no GPU devices are available.
    """
    if not are_asserts_enabled():
        return
    try:
        _check_backend_permitted(backend, "gpu")
    except AssertionError as exc:
        _raise_chex_error(
            str(exc),
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )
    if _num_devices_available("gpu") < 1:
        _raise_chex_error(
            "No GPU devices available.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def assert_tpu_available(
    backend: str | None = None,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that at least one TPU device is available.

    Args:
        backend: Optional backend name to query.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If no TPU devices are available.
    """
    if not are_asserts_enabled():
        return
    try:
        _check_backend_permitted(backend, "tpu")
    except AssertionError as exc:
        _raise_chex_error(
            str(exc),
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )
    if _num_devices_available("tpu") < 1:
        _raise_chex_error(
            "No TPU devices available.",
            custom_message=custom_message,
            custom_message_format_vars=custom_message_format_vars,
            include_default_message=include_default_message,
            exception_type=exception_type,
        )


def _check_sharding(leaf: LeafValue) -> bool:
    """Determine whether a PyTree leaf has an active sharding configuration.

    Args:
        leaf: The PyTree leaf to inspect.

    Returns:
        True if leaf has a non-None ShardingProtocol with a non-empty device_set.
    """
    if not isinstance(leaf, ShardingAttrProtocol):
        return False
    sharding = leaf.sharding
    if sharding is None or not isinstance(sharding, ShardingProtocol):
        return False
    return len(sharding.device_set) > 0


def assert_tree_is_on_device(
    tree: PyTree,
    platform: Sequence[str] | str = ("gpu", "tpu"),
    device: DeviceProtocol | None = None,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that all leaves of a PyTree reside on the specified platform(s) and/or device.

    Args:
        tree: The PyTree whose leaves are checked.
        platform: Allowed platform string or sequence of platform strings.
        device: Optional specific DeviceProtocol instance required for all leaves.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If any leaf is not on a device or does not match platform/device.
    """
    if not are_asserts_enabled():
        return
    platforms = (platform,) if isinstance(platform, str) else tuple(platform)
    for leaf in tree_leaves(tree):
        if not isinstance(leaf, DevicesMethodProtocol):
            _raise_chex_error(
                f"Leaf {leaf!r} is not placed on any device.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
        devs = leaf.devices()
        if not devs:
            _raise_chex_error(
                f"Leaf {leaf!r} has an empty devices() tuple.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
        if platforms and any(d.platform not in platforms for d in devs):
            _raise_chex_error(
                f"Leaf on platforms {[d.platform for d in devs]}, expected all in {platforms}.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
        if device is not None and any(d != device for d in devs):
            _raise_chex_error(
                f"Leaf device mismatch: got {devs}, expected {device}.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )


def assert_tree_is_on_host(
    tree: PyTree,
    allow_cpu_device: bool = True,
    allow_sharded_arrays: bool = False,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that all leaves in a PyTree reside in host memory (or CPU devices).

    Args:
        tree: The PyTree whose leaves are checked.
        allow_cpu_device: Whether arrays on 'cpu' devices are accepted as host-resident.
        allow_sharded_arrays: Whether sharded CPU arrays are permitted.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If any leaf resides on an accelerator or violates host constraints.
    """
    if not are_asserts_enabled():
        return
    for leaf in tree_leaves(tree):
        if not allow_sharded_arrays and _check_sharding(leaf):
            _raise_chex_error(
                f"Sharded array {leaf!r} is not allowed on host.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
        if isinstance(leaf, DevicesMethodProtocol):
            devs = leaf.devices()
            if not allow_cpu_device and devs and all(d.platform == "cpu" for d in devs):
                _raise_chex_error(
                    f"CPU device array {leaf!r} is not allowed when allow_cpu_device=False.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )
            if any(d.platform != "cpu" for d in devs):
                _raise_chex_error(
                    f"Leaf {leaf!r} resides on non-CPU device {[d.platform for d in devs]}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )


def assert_tree_is_sharded(
    tree: PyTree,
    devices: Sequence[DeviceProtocol] | None = None,
    *,
    custom_message: str | None = None,
    custom_message_format_vars: Sequence[Scalar | str] = (),
    include_default_message: bool = True,
    exception_type: type[Exception] = AssertionError,
) -> None:
    """Assert that all leaves of a PyTree are sharded across the specified devices.

    Args:
        tree: The PyTree whose leaves are inspected.
        devices: Optional sequence of expected devices in the sharding.
        custom_message: Optional user-supplied error message or format string.
        custom_message_format_vars: Positional format arguments interpolated into custom_message.
        include_default_message: Whether to include the default error message alongside custom_message.
        exception_type: Exception class to raise on failure.

    Raises:
        Exception: If any leaf is unsharded or sharded across different devices.
    """
    if not are_asserts_enabled():
        return
    for leaf in tree_leaves(tree):
        if not _check_sharding(leaf):
            _raise_chex_error(
                f"Tree leaf {leaf!r} is not sharded.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
        if (
            devices is not None
            and isinstance(leaf, ShardingAttrProtocol)
            and isinstance(leaf.sharding, ShardingProtocol)
        ):
            expected_set = frozenset(devices)
            if leaf.sharding.device_set != expected_set:
                _raise_chex_error(
                    f"Leaf sharded across {leaf.sharding.device_set!r}, expected {expected_set!r}.",
                    custom_message=custom_message,
                    custom_message_format_vars=custom_message_format_vars,
                    include_default_message=include_default_message,
                    exception_type=exception_type,
                )


__all__ = [
    "assert_devices_available",
    "assert_gpu_available",
    "assert_tpu_available",
    "assert_tree_is_on_device",
    "assert_tree_is_on_host",
    "assert_tree_is_sharded",
]
