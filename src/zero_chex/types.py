"""Core type definitions, structural protocols, and concrete reference classes for zero_chex."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

from ml_switcheroo_compiler.core.dtype import DType as CompilerDType
from ml_switcheroo_compiler.core.tensor import Tensor as CompilerTensor
from ml_switcheroo_ir import DType, LogicalNode, SymInt, SymNode, TensorShape, TensorSpec
from zero_flax.nnx.variables import Variable as FlaxVariable
from zero_grain.shared_memory_array import SharedMemoryArray, SharedMemoryArrayMetadata
from zero_jax.numpy import ndarray as JaxNdarray


@runtime_checkable
class TaylorScalarProtocol(Protocol):
    """Protocol for forward-mode Taylor polynomial scalars supporting float conversion."""

    def __float__(self) -> float:
        """Return the primal floating-point value."""
        ...


Scalar = Union[int, float, complex, bool]
NestedScalarSequence = Sequence[Union[Scalar, "NestedScalarSequence"]]
if TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)

_VALID_DTYPE_NAMES: frozenset[str] = frozenset(
    {
        "bool",
        "bool_",
        "int",
        "int4",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint",
        "uint4",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "complex",
        "complex64",
        "complex128",
        "str",
    }
)


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
class ToListProtocol(Protocol):
    """Protocol for array objects exposing a zero-argument `to_list()` method."""

    def to_list(self) -> Sequence[Scalar]:
        """Return the flattened scalar elements of the array."""
        ...


@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol for n-dimensional array and tensor objects."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the array as a tuple of integers."""
        ...

    @property
    def dtype(self) -> DType | CompilerDType | DTypeLikeProtocol | str | type[Scalar]:
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


@runtime_checkable
class VariableLeafProtocol(Protocol):
    """Protocol for NNX-style Variable leaf containers wrapping an inner array or scalar value."""

    @property
    def value(self) -> PyTree:
        """Return the underlying array or scalar value stored in the variable."""
        ...

    def copy(self) -> VariableLeafProtocol:
        """Return a shallow copy of the variable container."""
        ...

    def replace(self, value: PyTree | None = None) -> VariableLeafProtocol:
        """Return a new variable container with an updated value.

        Args:
            value: Optional replacement value for the variable container.

        Returns:
            A new VariableLeafProtocol instance.
        """
        ...


@runtime_checkable
class NamedTuplePyTreeProtocol(Protocol):
    """Protocol for NamedTuple PyTree containers such as zero-optax OptState tuples."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """Return the tuple of field names in the NamedTuple."""
        ...

    @classmethod
    def _make(cls, iterable: Sequence[PyTree]) -> NamedTuplePyTreeProtocol:
        """Construct a new NamedTuple instance from a sequence of field values.

        Args:
            iterable: Sequence of field values matching `_fields`.

        Returns:
            Reconstructed NamedTuple instance.
        """
        ...

    def __len__(self) -> int:
        """Return the number of fields in the NamedTuple."""
        ...

    def __iter__(self) -> Iterator[PyTree]:
        """Iterate over the field values in the NamedTuple."""
        ...


@runtime_checkable
class OptStateLeafProtocol(Protocol):
    """Protocol for zero-optax dataclass OptState leaf containers exposing `_fields`."""

    @property
    def _fields(self) -> tuple[str, ...]:
        """Return the tuple of field names defined on the state class."""
        ...


@runtime_checkable
class StateMappingProtocol(Protocol):
    """Protocol for custom mapping PyTree containers such as Flax State and FrozenDict."""

    def __init__(self, mapping: Mapping[str, PyTree] | None = None) -> None:
        """Initialize the mapping container from a dictionary of child entries.

        Args:
            mapping: Optional mapping of string keys to child PyTree values.
        """
        ...

    def keys(self) -> Iterable[DictKey]:
        """Return an iterable over the mapping keys.

        Returns:
            An iterable of DictKey keys.
        """
        ...

    def __getitem__(self, key: DictKey) -> PyTree:
        """Retrieve a child PyTree value by key.

        Args:
            key: Key identifying the child PyTree entry.

        Returns:
            The child PyTree value associated with key.
        """
        ...

    def __len__(self) -> int:
        """Return the number of entries in the mapping."""
        ...

    def __iter__(self) -> Iterator[DictKey]:
        """Iterate over the keys in the mapping."""
        ...


@runtime_checkable
class ShapeOnlyProtocol(Protocol):
    """Protocol for lightweight array-like objects that expose at least a shape tuple."""

    @property
    def shape(self) -> tuple[int, ...] | TensorShape:
        """Return the shape tuple of the array-like object."""
        ...


@runtime_checkable
class DTypeAttrProtocol(Protocol):
    """Protocol for objects that expose a dtype attribute."""

    @property
    def dtype(self) -> DType | CompilerDType | DTypeLikeProtocol | str | type[Scalar]:
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
        shape: Sequence[int] | TensorShape | None = None,
        dtype: DType | DTypeLikeProtocol | str | type[Scalar] = DType.float32,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> None:
        """Initialize a ZeroArray instance.

        Args:
            data: Scalar or (optionally nested) sequence of scalar elements.
            shape: Explicit shape tuple or TensorShape. If None, inferred from data.
            dtype: Data type of the array elements.
            device: Device or sequence of devices where the array resides.
            sharding: Optional sharding descriptor for distributed arrays.

        Raises:
            ValueError: If explicit shape is incompatible with the number of provided elements.
        """
        self._flat, inferred_shape = _flatten_nested_data(data)

        if isinstance(shape, TensorShape):
            self._shape: tuple[int, ...] = shape.static_shape
        elif shape is not None:
            self._shape = tuple(shape)
        else:
            self._shape = inferred_shape
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
    def from_tensor_spec(
        cls,
        spec: TensorSpec,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> ZeroArray:
        """Create a ZeroArray matching an ml-switcheroo-ir TensorSpec.

        Args:
            spec: TensorSpec providing static shape and dtype.
            device: Optional device placement.
            sharding: Optional sharding descriptor.

        Returns:
            A zero-initialized ZeroArray matching spec.
        """
        return cls.zeros(spec.static_shape, dtype=spec.dtype, device=device, sharding=sharding)

    @classmethod
    def zeros(
        cls,
        shape: int | Sequence[int] | TensorShape,
        dtype: DType | DTypeLikeProtocol | str | type[Scalar] = DType.float32,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> ZeroArray:
        """Create a ZeroArray filled with zeros.

        Args:
            shape: Dimension integer, sequence of dimension integers, or TensorShape.
            dtype: Desired data type of the array.
            device: Optional device placement.
            sharding: Optional sharding configuration.

        Returns:
            A new ZeroArray filled with 0.0 or 0.
        """
        if isinstance(shape, TensorShape):
            shape_tuple = shape.static_shape
        elif isinstance(shape, int):
            shape_tuple = (shape,)
        else:
            shape_tuple = tuple(shape)
        fill: Scalar = 0 if dtype in (DType.int32, DType.int64, int, "int32", "int64") else 0.0
        total = math.prod(shape_tuple) if shape_tuple else 1
        return cls([fill] * total, shape=shape_tuple, dtype=dtype, device=device, sharding=sharding)

    @classmethod
    def ones(
        cls,
        shape: int | Sequence[int] | TensorShape,
        dtype: DType | DTypeLikeProtocol | str | type[Scalar] = DType.float32,
        device: DeviceProtocol | Sequence[DeviceProtocol] | None = None,
        sharding: ShardingProtocol | None = None,
    ) -> ZeroArray:
        """Create a ZeroArray filled with ones.

        Args:
            shape: Dimension integer, sequence of dimension integers, or TensorShape.
            dtype: Desired data type of the array.
            device: Optional device placement.
            sharding: Optional sharding configuration.

        Returns:
            A new ZeroArray filled with 1.0 or 1.
        """
        if isinstance(shape, TensorShape):
            shape_tuple = shape.static_shape
        elif isinstance(shape, int):
            shape_tuple = (shape,)
        else:
            shape_tuple = tuple(shape)
        fill: Scalar = 1 if dtype in (DType.int32, DType.int64, int, "int32", "int64") else 1.0
        total = math.prod(shape_tuple) if shape_tuple else 1
        return cls([fill] * total, shape=shape_tuple, dtype=dtype, device=device, sharding=sharding)

    def to_tensor_spec(self) -> TensorSpec:
        """Convert this ZeroArray's metadata into an ml-switcheroo-ir TensorSpec.

        Returns:
            TensorSpec with matching shape and normalized DType.
        """
        norm_dt = self._dtype if isinstance(self._dtype, DType) else DType.float32
        if isinstance(self._dtype, str):
            norm_dt = DType.from_str(self._dtype)
        elif self._dtype is int:
            norm_dt = DType.int32
        elif self._dtype is bool:
            norm_dt = DType.bool
        elif self._dtype is complex:
            norm_dt = DType.complex64
        return TensorSpec(shape=self._shape, dtype=norm_dt)

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
            return self._shape == other.shape and self._flat == _extract_flat_scalars(other)
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


DictKey = Union[str, int, tuple[Union[str, int], ...]]
CustomNodeType = Union[
    type[MappableDataclassProtocol],
    type[NamedTuplePyTreeProtocol],
    type[StateMappingProtocol],
]


class PyTreeDef:
    """Structural definition of a flattened PyTree hierarchy."""

    def __init__(
        self,
        node_type: str = "leaf",
        keys: tuple[DictKey, ...] = (),
        children: tuple[PyTreeDef, ...] = (),
        custom_cls: CustomNodeType | None = None,
    ) -> None:
        """Initialize a PyTreeDef node.

        Args:
            node_type: Identifier for the container kind ('leaf', 'list', 'tuple', 'dict', 'custom').
            keys: Mapping or field keys associated with children.
            children: Child PyTreeDef structures.
            custom_cls: Registered custom dataclass, NamedTuple, or State mapping type if applicable.
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
    def keys(self) -> tuple[DictKey, ...]:
        """Return the keys associated with child nodes.

        Returns:
            Tuple of string, integer, or tuple keys.
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
    def custom_cls(self) -> CustomNodeType | None:
        """Return the custom container class type if applicable.

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


Array = Union[
    ZeroArray,
    CompilerTensor,
    JaxNdarray,
    ArrayProtocol,
    ShapeOnlyProtocol,
    TensorSpec,
    LogicalNode,
]
ArrayBatched = Array
ArrayDevice = Array
ArrayNumpy = Array
ArraySharded = Union[ZeroArray, ShardingAttrProtocol, Array]
Numeric = Union[
    Array,
    FlaxVariable[Union[CompilerTensor, JaxNdarray, ZeroArray, Scalar]],
    VariableLeafProtocol,
    SharedMemoryArray,
    SharedMemoryArrayMetadata,
    Scalar,
    TaylorScalarProtocol,
    SymInt,
    SymNode,
]
Shape = Union[Sequence[int], TensorShape]
ArrayDType = Union[DType, CompilerDType, DTypeLikeProtocol, str, type[Scalar], type[str]]
PRNGKey = Union[tuple[int, int], Array]
LeafValue = Union[
    Numeric,
    str,
    None,
    DeviceProtocol,
    DevicesMethodProtocol,
    ShardingAttrProtocol,
    DTypeAttrProtocol,
    OptStateLeafProtocol,
    ToListProtocol,
]

ArrayTree = Union[
    LeafValue,
    Iterable["ArrayTree"],
    Mapping[DictKey, "ArrayTree"],
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    StateMappingProtocol,
]
ArrayDeviceTree = Union[
    LeafValue,
    Iterable["ArrayDeviceTree"],
    Mapping[DictKey, "ArrayDeviceTree"],
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    StateMappingProtocol,
]
ArrayNumpyTree = Union[
    LeafValue,
    Iterable["ArrayNumpyTree"],
    Mapping[DictKey, "ArrayNumpyTree"],
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    StateMappingProtocol,
]
PyTree = Union[
    LeafValue,
    Iterable["PyTree"],
    Mapping[DictKey, "PyTree"],
    MappableDataclassProtocol,
    NamedTuplePyTreeProtocol,
    StateMappingProtocol,
]

TDimMatcher = Union[int, set[int], tuple[int, ...], str, EllipsisType, None]
TShapeMatcher = Sequence[TDimMatcher]


def _extract_flat_scalars(leaf: PyTree) -> list[Scalar]:
    """Extract a flat list of scalar values from a leaf array, Variable, or scalar.

    Args:
        leaf: Array, Variable, or scalar leaf value.

    Returns:
        Flat list of numeric scalars.
    """
    if isinstance(leaf, VariableLeafProtocol):
        return _extract_flat_scalars(leaf.value)
    if isinstance(leaf, CompilerTensor):
        if leaf.ndim == 0:
            return [leaf.item()]
        out: list[Scalar] = []
        for sub_tensor in leaf:
            out.extend(_extract_flat_scalars(sub_tensor))
        return out
    if isinstance(leaf, JaxNdarray):
        raw_list = leaf.tolist()
        flat_items, _ = _flatten_nested_data(raw_list) if isinstance(raw_list, list) else ([raw_list], ())
        return flat_items
    if isinstance(leaf, (int, float, complex, bool)):
        return [leaf]
    if isinstance(leaf, ToListProtocol):
        return list(leaf.to_list())
    return []


def _has_valid_shape_tuple(val: ShapeOnlyProtocol) -> bool:
    """Check whether a ShapeOnlyProtocol object has a valid shape tuple or TensorShape.

    Args:
        val: Candidate object exposing `.shape`.

    Returns:
        True if `val.shape` is a TensorShape or a tuple of non-negative integers.
    """
    raw_shape = val.shape
    if isinstance(raw_shape, TensorShape):
        return True
    return all(not isinstance(d, bool) and d >= 0 for d in raw_shape)


def is_scalar(val: PyTree) -> bool:
    """Check whether a value is a Python scalar or 0-D array.

    Args:
        val: Value to inspect.

    Returns:
        True if val is a Python int, float, complex, bool, SymInt, or 0-D array.
    """
    if isinstance(val, VariableLeafProtocol):
        return is_scalar(val.value)
    if isinstance(val, (int, float, complex, bool, SymInt, SymNode)):
        return True
    if isinstance(val, TensorSpec):
        return len(val.shape) == 0
    if isinstance(val, ArrayProtocol):
        return val.shape == ()
    if isinstance(val, ShapeOnlyProtocol) and _has_valid_shape_tuple(val):
        raw_s = val.shape
        return (len(raw_s.dims) == 0) if isinstance(raw_s, TensorShape) else (raw_s == ())
    return False


def is_array(val: PyTree) -> bool:
    """Check whether a value satisfies the ArrayProtocol, VariableLeafProtocol, or TensorSpec interface.

    Args:
        val: Value to inspect.

    Returns:
        True if val implements ArrayProtocol, VariableLeafProtocol, SharedMemoryArray, or TensorSpec.
    """
    return isinstance(
        val,
        (
            ArrayProtocol,
            VariableLeafProtocol,
            SharedMemoryArray,
            SharedMemoryArrayMetadata,
            TensorSpec,
        ),
    )


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
    "CompilerDType",
    "CompilerTensor",
    "CustomNodeType",
    "DType",
    "DTypeAttrProtocol",
    "DTypeLikeProtocol",
    "Device",
    "DeviceProtocol",
    "DevicesMethodProtocol",
    "DictKey",
    "EllipsisType",
    "FlaxVariable",
    "JaxNdarray",
    "LeafValue",
    "MappableDataclassProtocol",
    "NamedTuplePyTreeProtocol",
    "NestedScalarSequence",
    "Numeric",
    "OptStateLeafProtocol",
    "PRNGKey",
    "PmapSharding",
    "PyTree",
    "PyTreeDef",
    "Scalar",
    "Shape",
    "ShapeOnlyProtocol",
    "SharedMemoryArray",
    "SharedMemoryArrayMetadata",
    "ShardingAttrProtocol",
    "ShardingProtocol",
    "StateMappingProtocol",
    "TDimMatcher",
    "TShapeMatcher",
    "TaylorScalarProtocol",
    "ToListProtocol",
    "VariableLeafProtocol",
    "ZeroArray",
    "_VALID_DTYPE_NAMES",
    "_extract_flat_scalars",
    "_flatten_nested_data",
    "_has_valid_shape_tuple",
    "is_array",
    "is_scalar",
]
