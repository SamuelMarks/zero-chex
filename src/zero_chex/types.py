"""Core type definitions, structural protocols, and concrete reference classes for zero_chex."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

from ml_switcheroo_ir import DType

Scalar = Union[int, float, complex, bool]
NestedScalarSequence = Sequence[Union[Scalar, "NestedScalarSequence"]]
if TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)


@runtime_checkable
class DTypeLikeProtocol(Protocol):
    """Protocol for objects exposing a data type name."""

    @property
    def name(self) -> str:
        """Return the canonical string name of the data type."""
        ...


@runtime_checkable
class DeviceProtocol(Protocol):
    """Protocol for compute device descriptors."""

    @property
    def platform(self) -> str:
        """Return the hardware platform identifier (e.g., 'cpu', 'gpu', 'tpu')."""
        ...

    @property
    def id(self) -> int:
        """Return the numerical device identifier."""
        ...


@runtime_checkable
class ShardingProtocol(Protocol):
    """Protocol for array sharding descriptors."""

    @property
    def device_set(self) -> frozenset[DeviceProtocol]:
        """Return the set of devices across which an array is sharded."""
        ...

    @property
    def is_fully_replicated(self) -> bool:
        """Return True if the array is replicated across all devices."""
        ...


@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol for n-dimensional array and tensor objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the array as a tuple of integers."""
        ...

    @property
    def dtype(self) -> DType | DTypeLikeProtocol | str | type[Scalar]:
        """Return the element data type of the array."""
        ...

    @property
    def size(self) -> int:
        """Return the total number of elements in the array."""
        ...

    @property
    def ndim(self) -> int:
        """Return the number of dimensions (rank) of the array."""
        ...

    @property
    def sharding(self) -> ShardingProtocol | None:
        """Return the sharding configuration of the array, if any."""
        ...

    def devices(self) -> Sequence[DeviceProtocol]:
        """Return the sequence of devices on which this array resides."""
        ...

    def to_list(self) -> list[Scalar]:
        """Return the flattened scalar elements of the array."""
        ...


@runtime_checkable
class ShapeOnlyProtocol(Protocol):
    """Protocol for lightweight array-like objects that expose at least a shape tuple."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape tuple of the array-like object."""
        ...


@runtime_checkable
class DTypeAttrProtocol(Protocol):
    """Protocol for objects that expose a dtype attribute."""

    @property
    def dtype(self) -> DType | DTypeLikeProtocol | str | type[Scalar]:
        """Return the dtype of the object."""
        ...


@runtime_checkable
class DevicesMethodProtocol(Protocol):
    """Protocol for objects that expose a devices() method."""

    def devices(self) -> Sequence[DeviceProtocol]:
        """Return the devices associated with the object."""
        ...


@runtime_checkable
class ShardingAttrProtocol(Protocol):
    """Protocol for objects that expose a sharding attribute."""

    @property
    def sharding(self) -> ShardingProtocol | None:
        """Return the sharding attribute of the object."""
        ...


@runtime_checkable
class MappableDataclassProtocol(Protocol):
    """Protocol for dataclasses registered with mappable dictionary-like access."""

    def __chex_mappable_fields__(self) -> tuple[str, ...]:
        """Return the ordered tuple of field names in the dataclass."""
        ...

    def __chex_field_values__(self) -> tuple[PyTree, ...]:
        """Return the ordered tuple of field values in the dataclass."""
        ...


class Device:
    """Concrete compute device descriptor implementing DeviceProtocol."""

    def __init__(self, platform: str = "cpu", id: int = 0) -> None:
        """Initialize a Device descriptor.

        Args:
            platform: The device platform name ('cpu', 'gpu', or 'tpu').
            id: The integer device index.
        """
        self._platform = platform
        self._id = id

    @property
    def platform(self) -> str:
        """Return the hardware platform identifier.

        Returns:
            The platform name string.
        """
        return self._platform

    @property
    def id(self) -> int:
        """Return the numerical device identifier.

        Returns:
            The integer device ID.
        """
        return self._id

    def __eq__(self, other: PyTree | PyTreeDef) -> bool:
        """Compare two Device instances for structural equality.

        Args:
            other: Another device to compare against.

        Returns:
            True if both have identical platform and id.
        """
        if isinstance(other, DeviceProtocol):
            return self._platform == other.platform and self._id == other.id
        return False

    def __hash__(self) -> int:
        """Compute hash for the Device instance.

        Returns:
            Integer hash value.
        """
        return hash((self._platform, self._id))

    def __repr__(self) -> str:
        """Return string representation of the Device.

        Returns:
            Formatted string representation.
        """
        return f"Device(platform={self._platform!r}, id={self._id})"


class PmapSharding:
    """Concrete sharding descriptor representing PMAP-style device sharding."""

    def __init__(
        self,
        devices: Sequence[DeviceProtocol] = (),
        is_fully_replicated: bool = False,
    ) -> None:
        """Initialize a PmapSharding descriptor.

        Args:
            devices: Sequence of devices participating in the sharding.
            is_fully_replicated: Whether the array is fully replicated.
        """
        resolved = tuple(devices) if devices else (Device("cpu", 0),)
        self._device_set: frozenset[DeviceProtocol] = frozenset(resolved)
        self._is_fully_replicated = is_fully_replicated

    @property
    def device_set(self) -> frozenset[DeviceProtocol]:
        """Return the frozenset of devices in this sharding.

        Returns:
            Frozenset of DeviceProtocol instances.
        """
        return self._device_set

    @property
    def is_fully_replicated(self) -> bool:
        """Return whether the sharding is fully replicated.

        Returns:
            True if fully replicated, False otherwise.
        """
        return self._is_fully_replicated


def _flatten_nested_data(
    data: NestedScalarSequence | Scalar,
) -> tuple[list[Scalar], tuple[int, ...]]:
    """Recursively flatten a scalar or nested sequence of scalars and infer its shape.

    Args:
        data: A scalar or nested sequence of scalars.

    Returns:
        A tuple of (flat_scalar_list, inferred_shape_tuple).

    Raises:
        ValueError: If nested sequences have ragged (inconsistent) sub-shapes.
    """
    if isinstance(data, (int, float, complex, bool)):
        return [data], ()
    if not data:
        return [], (0,)
    flat_items: list[Scalar] = []
    sub_shapes: list[tuple[int, ...]] = []
    for elem in data:
        sub_flat, sub_shape = _flatten_nested_data(elem)
        flat_items.extend(sub_flat)
        sub_shapes.append(sub_shape)
    first_sub = sub_shapes[0]
    if any(s != first_sub for s in sub_shapes[1:]):
        raise ValueError(f"Ragged nested sequence shapes are not allowed: {sub_shapes}")
    return flat_items, (len(data),) + first_sub


class ZeroArray:
    """Pure-Python n-dimensional array implementation satisfying ArrayProtocol."""

    def __init__(
        self,
        data: NestedScalarSequence | Scalar,
        shape: Sequence[int] | None = None,
        dtype: DType | DTypeLikeProtocol | str | type[Scalar] = DType.float32,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> None:
        """Initialize a ZeroArray instance.

        Args:
            data: Scalar or (optionally nested) sequence of scalar elements.
            shape: Explicit shape tuple. If None, inferred from data.
            dtype: Data type of the array elements.
            device: Device or sequence of devices where the array resides.
            sharding: Optional sharding descriptor for distributed arrays.

        Raises:
            ValueError: If explicit shape is incompatible with the number of provided elements.
        """
        self._flat, inferred_shape = _flatten_nested_data(data)

        self._shape: tuple[int, ...] = tuple(shape) if shape is not None else inferred_shape
        expected_size = math.prod(self._shape) if self._shape else 1
        if len(self._flat) == 1 and expected_size > 1:
            self._flat = self._flat * expected_size
        elif expected_size == 0:
            self._flat = []
        elif len(self._flat) != expected_size:
            raise ValueError(f"cannot reshape array of size {len(self._flat)} into shape {self._shape}")

        self._dtype: DType | DTypeLikeProtocol | str | type[Scalar] = dtype
        if device is None:
            self._devices: tuple[DeviceProtocol, ...] = (Device("cpu", 0),)
        elif isinstance(device, DeviceProtocol):
            self._devices = (device,)
        else:
            self._devices = tuple(device)
        self._sharding = sharding

    @classmethod
    def zeros(
        cls,
        shape: int | Sequence[int],
        dtype: DType | DTypeLikeProtocol | str | type[Scalar] = DType.float32,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> ZeroArray:
        """Create a ZeroArray filled with zeros.

        Args:
            shape: Dimension integer or sequence of dimension integers.
            dtype: Desired data type of the array.
            device: Optional device placement.
            sharding: Optional sharding configuration.

        Returns:
            A new ZeroArray filled with 0.0 or 0.
        """
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        fill: Scalar = 0 if dtype in (DType.int32, DType.int64, int, "int32", "int64") else 0.0
        total = math.prod(shape_tuple) if shape_tuple else 1
        return cls([fill] * total, shape=shape_tuple, dtype=dtype, device=device, sharding=sharding)

    @classmethod
    def ones(
        cls,
        shape: int | Sequence[int],
        dtype: DType | DTypeLikeProtocol | str | type[Scalar] = DType.float32,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> ZeroArray:
        """Create a ZeroArray filled with ones.

        Args:
            shape: Dimension integer or sequence of dimension integers.
            dtype: Desired data type of the array.
            device: Optional device placement.
            sharding: Optional sharding configuration.

        Returns:
            A new ZeroArray filled with 1.0 or 1.
        """
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        fill: Scalar = 1 if dtype in (DType.int32, DType.int64, int, "int32", "int64") else 1.0
        total = math.prod(shape_tuple) if shape_tuple else 1
        return cls([fill] * total, shape=shape_tuple, dtype=dtype, device=device, sharding=sharding)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape tuple of the array.

        Returns:
            Tuple of dimension sizes.
        """
        return self._shape

    @property
    def dtype(self) -> DType | DTypeLikeProtocol | str | type[Scalar]:
        """Return the data type of the array.

        Returns:
            The array data type.
        """
        return self._dtype

    @property
    def size(self) -> int:
        """Return the total number of elements in the array.

        Returns:
            Product of all dimensions in shape.
        """
        return math.prod(self._shape) if self._shape else 1

    @property
    def ndim(self) -> int:
        """Return the rank (number of dimensions) of the array.

        Returns:
            Length of the shape tuple.
        """
        return len(self._shape)

    @property
    def sharding(self) -> ShardingProtocol | None:
        """Return the sharding configuration of the array.

        Returns:
            ShardingProtocol instance or None.
        """
        return self._sharding

    def devices(self) -> Sequence[DeviceProtocol]:
        """Return the devices on which this array is placed.

        Returns:
            Tuple of DeviceProtocol instances.
        """
        return self._devices

    def to_list(self) -> list[Scalar]:
        """Return a flat list of the scalar elements in the array.

        Returns:
            List of scalar values.
        """
        return list(self._flat)

    def __iter__(self) -> Iterator[Scalar]:
        """Iterate over the flat scalar elements of the array.

        Returns:
            Iterator over scalar elements.
        """
        return iter(self._flat)

    def __len__(self) -> int:
        """Return the size of the leading dimension.

        Returns:
            Leading dimension length.

        Raises:
            TypeError: If the array is 0-dimensional.
        """
        if not self._shape:
            raise TypeError("len() of unsized 0-D ZeroArray")
        return self._shape[0]

    def __eq__(self, other: PyTree | PyTreeDef) -> bool:
        """Check structural and value equality with another array or scalar.

        Args:
            other: Another ZeroArray, ArrayProtocol, or Scalar.

        Returns:
            True if shapes and all elements are equal.
        """
        if isinstance(other, ArrayProtocol):
            return self._shape == other.shape and self._flat == list(other.to_list())
        if isinstance(other, (int, float, complex, bool)) and self.size == 1:
            return self._flat[0] == other
        return False

    def __hash__(self) -> int:
        """Compute hash for ZeroArray based on shape and flat values.

        Returns:
            Integer hash value.
        """
        return hash((self._shape, tuple(self._flat)))

    def __repr__(self) -> str:
        """Return string representation of the ZeroArray.

        Returns:
            Formatted string representation.
        """
        return f"ZeroArray(shape={self._shape}, dtype={self._dtype!r}, data={self._flat!r})"


class PyTreeDef:
    """Structural definition of a flattened PyTree hierarchy."""

    def __init__(
        self,
        node_type: str = "leaf",
        keys: tuple[str | int, ...] = (),
        children: tuple[PyTreeDef, ...] = (),
        custom_cls: type[MappableDataclassProtocol] | None = None,
    ) -> None:
        """Initialize a PyTreeDef node.

        Args:
            node_type: Identifier for the container kind ('leaf', 'list', 'tuple', 'dict', 'custom').
            keys: Mapping or field keys associated with children.
            children: Child PyTreeDef structures.
            custom_cls: Registered custom dataclass type if node_type is 'custom'.
        """
        self._node_type = node_type
        self._keys = keys
        self._children = children
        self._custom_cls = custom_cls

    @property
    def node_type(self) -> str:
        """Return the node container type string.

        Returns:
            Container type identifier.
        """
        return self._node_type

    @property
    def keys(self) -> tuple[str | int, ...]:
        """Return the keys associated with child nodes.

        Returns:
            Tuple of string or integer keys.
        """
        return self._keys

    @property
    def children(self) -> tuple[PyTreeDef, ...]:
        """Return the child PyTreeDef definitions.

        Returns:
            Tuple of child PyTreeDef instances.
        """
        return self._children

    @property
    def custom_cls(self) -> type[MappableDataclassProtocol] | None:
        """Return the custom dataclass type if applicable.

        Returns:
            Custom class type or None.
        """
        return self._custom_cls

    @property
    def num_leaves(self) -> int:
        """Return the total number of leaf nodes represented by this PyTreeDef.

        Returns:
            Integer leaf count.
        """
        if self._node_type == "leaf":
            return 1
        return sum(child.num_leaves for child in self._children)

    def __eq__(self, other: PyTree | PyTreeDef) -> bool:
        """Compare two PyTreeDef structures for exact structural equivalence.

        Args:
            other: Another PyTreeDef to compare with.

        Returns:
            True if both represent the exact same PyTree container structure.
        """
        if not isinstance(other, PyTreeDef):
            return False
        return (
            self._node_type == other.node_type
            and self._keys == other.keys
            and self._children == other.children
            and self._custom_cls == other.custom_cls
        )

    def __hash__(self) -> int:
        """Compute hash of the PyTreeDef structure.

        Returns:
            Integer hash value.
        """
        return hash((self._node_type, self._keys, self._children, self._custom_cls))

    def __repr__(self) -> str:
        """Return string representation of the PyTreeDef.

        Returns:
            Formatted string representation.
        """
        if self._node_type == "leaf":
            return "PyTreeDef(*)"
        child_str = ", ".join(repr(c) for c in self._children)
        return f"PyTreeDef({self._node_type}[{self._keys!r}]({child_str}))"


class ChexifyChecks:
    """Collection of check categories supported by chexify."""

    user: frozenset[str] = frozenset({"user"})
    nan: frozenset[str] = frozenset({"nan"})
    index: frozenset[str] = frozenset({"index"})
    div: frozenset[str] = frozenset({"div"})
    float: frozenset[str] = frozenset({"nan", "div"})
    automatic: frozenset[str] = frozenset({"nan", "index", "div"})
    all: frozenset[str] = frozenset({"user", "nan", "index", "div"})


Array = Union[ArrayProtocol, ShapeOnlyProtocol]
ArrayBatched = Union[ArrayProtocol, ShapeOnlyProtocol]
ArrayDevice = Union[ArrayProtocol, ShapeOnlyProtocol]
ArrayNumpy = Union[ArrayProtocol, ShapeOnlyProtocol]
ArraySharded = Union[ArrayProtocol, ShapeOnlyProtocol]
Numeric = Union[Array, Scalar]
Shape = Sequence[int]
ArrayDType = Union[DType, DTypeLikeProtocol, str, type[Scalar], type]
PRNGKey = Union[tuple[int, int], Array]
LeafValue = Union[
    Numeric,
    str,
    None,
    DeviceProtocol,
    DevicesMethodProtocol,
    ShardingAttrProtocol,
    DTypeAttrProtocol,
]

ArrayTree = Union[
    LeafValue,
    Iterable["ArrayTree"],
    Mapping[str, "ArrayTree"],
    MappableDataclassProtocol,
]
ArrayDeviceTree = Union[
    LeafValue,
    Iterable["ArrayDeviceTree"],
    Mapping[str, "ArrayDeviceTree"],
    MappableDataclassProtocol,
]
ArrayNumpyTree = Union[
    LeafValue,
    Iterable["ArrayNumpyTree"],
    Mapping[str, "ArrayNumpyTree"],
    MappableDataclassProtocol,
]
PyTree = Union[
    LeafValue,
    Iterable["PyTree"],
    Mapping[str, "PyTree"],
    MappableDataclassProtocol,
]

TDimMatcher = Union[int, set[int], tuple[int, ...], EllipsisType, None]
TShapeMatcher = Sequence[TDimMatcher]


def is_scalar(val: PyTree) -> bool:
    """Check whether a value is a Python scalar or 0-D array.

    Args:
        val: Value to inspect.

    Returns:
        True if val is a Python int, float, complex, bool, or 0-D array.
    """
    if isinstance(val, (int, float, complex, bool)):
        return True
    if isinstance(val, (ArrayProtocol, ShapeOnlyProtocol)):
        return val.shape == ()
    return False


def is_array(val: PyTree) -> bool:
    """Check whether a value satisfies the ArrayProtocol interface.

    Args:
        val: Value to inspect.

    Returns:
        True if val implements ArrayProtocol.
    """
    return isinstance(val, ArrayProtocol)


__all__ = [
    "Array",
    "ArrayBatched",
    "ArrayDType",
    "ArrayDevice",
    "ArrayDeviceTree",
    "ArrayNumpy",
    "ArrayNumpyTree",
    "ArrayProtocol",
    "ArraySharded",
    "ArrayTree",
    "ChexifyChecks",
    "DTypeAttrProtocol",
    "DTypeLikeProtocol",
    "Device",
    "DeviceProtocol",
    "DevicesMethodProtocol",
    "EllipsisType",
    "LeafValue",
    "MappableDataclassProtocol",
    "NestedScalarSequence",
    "Numeric",
    "PRNGKey",
    "PmapSharding",
    "PyTree",
    "PyTreeDef",
    "Scalar",
    "Shape",
    "ShapeOnlyProtocol",
    "ShardingAttrProtocol",
    "ShardingProtocol",
    "TDimMatcher",
    "TShapeMatcher",
    "ZeroArray",
    "is_array",
    "is_scalar",
]
