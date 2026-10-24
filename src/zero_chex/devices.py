"""Device availability, placement, host residence, and sharding assertions for zero_chex."""

from __future__ import annotations

from collections.abc import Sequence

from zero_chex._src.tree_util import tree_leaves
from zero_chex.mocks import _STATE, are_asserts_enabled, get_num_devices_available
from zero_chex.types import (
    DeviceProtocol,
    DevicesMethodProtocol,
    LeafValue,
    PmapSharding,
    PyTree,
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


def _num_devices_available(platform: str) -> int:
    """Return the number of available devices for a given platform.

    Args:
        platform: The hardware platform identifier ('cpu', 'gpu', or 'tpu').

    Returns:
        Integer count of available devices on the platform.

    Raises:
        ValueError: If platform is unknown.
    """
    return get_num_devices_available(platform)


def assert_devices_available(
    n: int,
    devtype: str = "cpu",
    backend: str | None = None,
    not_less_than: bool = False,
) -> None:
    """Assert that a specified number of devices are available on the given platform.

    Args:
        n: Expected number of devices.
        devtype: Hardware device type ('cpu', 'gpu', or 'tpu').
        backend: Optional backend identifier.
        not_less_than: If True, asserts available >= n; otherwise asserts available == n.

    Raises:
        ValueError: If devtype or backend is unrecognized.
        AssertionError: If the available device count does not satisfy the requirement.
    """
    if not are_asserts_enabled():
        return
    _check_backend_permitted(backend, devtype)
    av = _num_devices_available(devtype)
    if not_less_than:
        if av < n:
            raise AssertionError(f"Only {av} < {n} {devtype.upper()}s available.")
    elif av != n:
        raise AssertionError(f"No {n} {devtype.upper()}s available ({av} != {n}).")


def assert_gpu_available(backend: str | None = None) -> None:
    """Assert that at least one GPU device is available.

    Args:
        backend: Optional backend identifier.

    Raises:
        ValueError: If backend is unrecognized.
        AssertionError: If zero GPU devices are available or backend is restricted.
    """
    if not are_asserts_enabled():
        return
    _check_backend_permitted(backend, "gpu")
    if _num_devices_available("gpu") == 0:
        raise AssertionError("No GPU devices available.")


def assert_tpu_available(backend: str | None = None) -> None:
    """Assert that at least one TPU device is available.

    Args:
        backend: Optional backend identifier.

    Raises:
        ValueError: If backend is unrecognized.
        AssertionError: If zero TPU devices are available or backend is restricted.
    """
    if not are_asserts_enabled():
        return
    _check_backend_permitted(backend, "tpu")
    if _num_devices_available("tpu") == 0:
        raise AssertionError("No TPU devices available.")


def _check_sharding(leaf: LeafValue) -> bool:
    """Inspect whether a PyTree leaf has active PMAP/multi-device sharding.

    Args:
        leaf: The PyTree leaf to inspect.

    Returns:
        True if the leaf has a recognized sharding descriptor.
    """
    if isinstance(leaf, ShardingAttrProtocol):
        sharding = leaf.sharding
        if sharding is not None:
            return isinstance(sharding, (PmapSharding, ShardingProtocol))
    return False


def assert_tree_is_on_device(
    tree: PyTree,
    platform: str | Sequence[str] = ("gpu", "tpu"),
    device: DeviceProtocol | None = None,
) -> None:
    """Assert that all leaves of a PyTree reside on the specified platform or device.

    Args:
        tree: The PyTree whose leaves are checked.
        platform: Allowed platform string or sequence of platform strings.
        device: Specific DeviceProtocol instance required, if any.

    Raises:
        AssertionError: If any leaf is not a device array or resides on a disallowed platform/device.
    """
    if not are_asserts_enabled():
        return
    platforms: tuple[str, ...] = (platform,) if isinstance(platform, str) else tuple(platform)
    for leaf in tree_leaves(tree):
        if not isinstance(leaf, DevicesMethodProtocol):
            raise AssertionError(f"Leaf {leaf!r} is not placed on any device (missing .devices() method).")
        devs = leaf.devices()
        if platforms and devs and all(d.platform not in platforms for d in devs):
            raise AssertionError(f"Leaf {leaf!r} is on {[d.platform for d in devs]}, expected {platforms!r}.")
        if device is not None and devs and any(d != device for d in devs):
            raise AssertionError(f"Leaf {leaf!r} device mismatch: got {devs!r}, expected {device!r}.")


def assert_tree_is_on_host(
    tree: PyTree,
    allow_cpu_device: bool = True,
    allow_sharded_arrays: bool = False,
) -> None:
    """Assert that all leaves of a PyTree reside in host (CPU) memory.

    Args:
        tree: The PyTree whose leaves are checked.
        allow_cpu_device: Whether arrays on 'cpu' devices are accepted as host-resident.
        allow_sharded_arrays: Whether sharded CPU arrays are permitted.

    Raises:
        AssertionError: If any leaf resides on an accelerator or violates host constraints.
    """
    if not are_asserts_enabled():
        return
    for leaf in tree_leaves(tree):
        if not allow_sharded_arrays and _check_sharding(leaf):
            raise AssertionError(f"Sharded array {leaf!r} is not allowed on host.")
        if isinstance(leaf, DevicesMethodProtocol):
            devs = leaf.devices()
            if not allow_cpu_device and devs and all(d.platform == "cpu" for d in devs):
                raise AssertionError(f"CPU device array {leaf!r} is not allowed when allow_cpu_device=False.")
            if any(d.platform != "cpu" for d in devs):
                raise AssertionError(f"Leaf {leaf!r} resides on non-CPU device {[d.platform for d in devs]}.")


def assert_tree_is_sharded(
    tree: PyTree,
    devices: Sequence[DeviceProtocol] | None = None,
) -> None:
    """Assert that all leaves of a PyTree are sharded across the specified devices.

    Args:
        tree: The PyTree whose leaves are inspected.
        devices: Optional sequence of expected devices in the sharding.

    Raises:
        AssertionError: If any leaf is unsharded or sharded across different devices.
    """
    if not are_asserts_enabled():
        return
    for leaf in tree_leaves(tree):
        if not _check_sharding(leaf):
            raise AssertionError(f"Tree leaf {leaf!r} is not sharded.")
        if (
            devices is not None
            and isinstance(leaf, ShardingAttrProtocol)
            and isinstance(leaf.sharding, ShardingProtocol)
        ):
            expected_set = frozenset(devices)
            if leaf.sharding.device_set != expected_set:
                raise AssertionError(f"Leaf sharded across {leaf.sharding.device_set!r}, expected {expected_set!r}.")


__all__ = [
    "_check_sharding",
    "_num_devices_available",
    "assert_devices_available",
    "assert_gpu_available",
    "assert_tpu_available",
    "assert_tree_is_on_device",
    "assert_tree_is_on_host",
    "assert_tree_is_sharded",
]
