"""Device availability, placement, host residence, and sharding assertions for zero_chex."""

from __future__ import annotations

from collections.abc import Sequence

from ml_switcheroo_compiler.core.device import Device as CompilerDevice

from zero_chex._src.asserts_internal import (
    are_asserts_enabled,
)
from zero_chex._src.asserts_internal import (
    check_backend_permitted as _check_backend_permitted,
)
from zero_chex._src.asserts_internal import (
    num_devices_available as _num_devices_available,
)
from zero_chex._src.asserts_internal import (
    raise_chex_error as _raise_chex_error,
)
from zero_chex._src.tree_util import tree_leaves
from zero_chex.types import (
    CompilerTensor,
    Device,
    DeviceProtocol,
    DevicesMethodProtocol,
    FlaxVariable,
    GlobalArraySharding,
    JaxNdarray,
    LeafValue,
    NamedSharding,
    PartitionSpec,
    PyTree,
    Scalar,
    SerializedArray,
    ShardingAttrProtocol,
    ShardingProtocol,
    SingleDeviceSharding,
    VariableLeafProtocol,
)


def _coerce_device_item(
    item: DeviceProtocol | CompilerDevice | str | int | float | bool | None,
) -> DeviceProtocol:
    """Convert a device object or string identifier (e.g. 'cpu:0') into a DeviceProtocol.

    Args:
        item: DeviceProtocol, CompilerDevice, or string device identifier.

    Returns:
        Normalized DeviceProtocol instance.
    """
    if isinstance(item, DeviceProtocol):
        return Device(item.platform.lower(), item.id)
    if isinstance(item, CompilerDevice):
        return Device(str(item.device_type).split(".")[-1].lower(), item.index or 0)
    raw = str(item).strip().lower()
    if ":" in raw:
        plat, idx_str = raw.split(":", 1)
        return Device(plat, int(idx_str))
    return Device(raw if raw in ("cpu", "gpu", "tpu") else "cpu", 0)


def _extract_sharding_obj(
    leaf: LeafValue,
) -> ShardingProtocol | NamedSharding | PartitionSpec | GlobalArraySharding | SingleDeviceSharding | None:
    """Extract the active sharding descriptor from a leaf, Variable metadata, or inner array.

    Args:
        leaf: PyTree leaf to inspect.

    Returns:
        Sharding descriptor or None if unsharded.
    """
    if isinstance(leaf, FlaxVariable):
        meta_sharding = leaf.metadata.get("sharding")
        if isinstance(meta_sharding, Sequence) and not isinstance(meta_sharding, str) and len(meta_sharding) > 0:
            return PartitionSpec(*[str(axis) for axis in meta_sharding])
        inner_val = leaf.value
        if isinstance(
            inner_val,
            (JaxNdarray, ShardingAttrProtocol, FlaxVariable, VariableLeafProtocol),
        ):
            return _extract_sharding_obj(inner_val)
        return None
    if isinstance(leaf, JaxNdarray):
        raw_s = leaf._sharding
        if isinstance(
            raw_s,
            (NamedSharding, PartitionSpec, GlobalArraySharding, SingleDeviceSharding, ShardingProtocol),
        ):
            return raw_s
        return None
    if isinstance(leaf, ShardingAttrProtocol):
        return leaf.sharding
    return None


def _extract_sharding_device_set(leaf: LeafValue) -> frozenset[DeviceProtocol] | None:
    """Extract the frozenset of DeviceProtocol instances from a sharded leaf if available.

    Args:
        leaf: PyTree leaf to inspect.

    Returns:
        Frozenset of DeviceProtocol instances, or None if not device-set based.
    """
    sharding = _extract_sharding_obj(leaf)
    if isinstance(sharding, (NamedSharding, ShardingProtocol)):
        return frozenset(_coerce_device_item(d) for d in sharding.device_set)
    return None


def _extract_leaf_devices(leaf: LeafValue) -> Sequence[DeviceProtocol] | None:
    """Extract the sequence of devices on which a leaf resides.

    Args:
        leaf: PyTree leaf to inspect.

    Returns:
        Sequence of DeviceProtocol instances, or None if leaf is not device-placed.
    """
    if isinstance(leaf, VariableLeafProtocol):
        inner = leaf.value
        if isinstance(
            inner,
            (DevicesMethodProtocol, JaxNdarray, CompilerTensor, SerializedArray, VariableLeafProtocol),
        ):
            return _extract_leaf_devices(inner)
        return None
    if isinstance(leaf, DevicesMethodProtocol):
        return leaf.devices()
    if isinstance(leaf, JaxNdarray):
        sharding = leaf.sharding
        if isinstance(sharding, (SingleDeviceSharding, NamedSharding, ShardingProtocol)):
            return tuple(_coerce_device_item(d) for d in sharding.device_set)
        t_dev = leaf._tensor.device
        return (Device(str(t_dev.device_type).split(".")[-1].lower(), t_dev.index or 0),)
    if isinstance(leaf, CompilerTensor):
        c_dev = leaf.device
        return (Device(str(c_dev.device_type).split(".")[-1].lower(), c_dev.index or 0),)
    if isinstance(leaf, SerializedArray):
        return (Device("cpu", 0),)
    return None


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
        True if leaf has a non-None NamedSharding, PartitionSpec, GlobalArraySharding, or ShardingProtocol.
    """
    sharding = _extract_sharding_obj(leaf)
    if sharding is None or isinstance(sharding, SingleDeviceSharding):
        return False
    if isinstance(sharding, PartitionSpec):
        return len(sharding.axes) > 0 and any(p is not None for p in sharding.axes)
    if isinstance(sharding, GlobalArraySharding):
        return True
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
        devs = _extract_leaf_devices(leaf)
        if devs is None:
            _raise_chex_error(
                f"Leaf {leaf!r} is not placed on any device.",
                custom_message=custom_message,
                custom_message_format_vars=custom_message_format_vars,
                include_default_message=include_default_message,
                exception_type=exception_type,
            )
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
        inner = leaf.value if isinstance(leaf, VariableLeafProtocol) else leaf
        if isinstance(inner, SerializedArray):
            continue
        devs = _extract_leaf_devices(leaf)
        if devs is not None:
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
        if devices is not None:
            actual_set = _extract_sharding_device_set(leaf)
            if actual_set is not None:
                expected_set = frozenset(_coerce_device_item(d) for d in devices)
                if actual_set != expected_set:
                    _raise_chex_error(
                        f"Leaf sharded across {actual_set!r}, expected {expected_set!r}.",
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
