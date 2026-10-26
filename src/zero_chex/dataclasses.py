"""Dataclass decorators and PyTree registration utilities for zero_chex."""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, TypeVar, overload

from typing_extensions import TypeGuard, dataclass_transform

from zero_chex._src.tree_util import CustomDataclassType, register_custom_pytree_node
from zero_chex.types import DataclassInstanceProtocol, MappableDataclassProtocol, PyTree

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class _KwOnlyInitProtocol(Protocol[T_contra]):
    """Protocol for a keyword-only __init__ function in a dataclass namespace."""

    def __call__(self, self_inst: T_contra, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize dataclass instance enforcing keyword-only arguments.

        Args:
            self_inst: Instance being initialized.
            *args: Disallowed positional arguments.
            **kwargs: Keyword field values.
        """
        ...


def _get_field_names(instance: MappableDataclassProtocol | DataclassInstanceProtocol) -> tuple[str, ...]:
    """Return the declared dataclass field names for an instance.

    Args:
        instance: Dataclass instance to inspect.

    Returns:
        Tuple of declared field name strings.
    """
    assert isinstance(instance, DataclassInstanceProtocol)
    return tuple(f.name for f in dataclasses.fields(instance))


def _get_field_values(instance: MappableDataclassProtocol | DataclassInstanceProtocol) -> tuple[PyTree, ...]:
    """Extract declared dataclass field values shallowly in declaration order.

    Args:
        instance: Dataclass instance to inspect.

    Returns:
        Ordered tuple of field values corresponding to declared dataclass fields.
    """
    assert isinstance(instance, DataclassInstanceProtocol)
    inst_dict = vars(instance)
    values: list[PyTree] = []
    for f in dataclasses.fields(instance):
        if f.name in inst_dict:
            val: PyTree = inst_dict[f.name]
            values.append(val)
    return tuple(values)


def _extract_dataclass_fields(instance: MappableDataclassProtocol | DataclassInstanceProtocol) -> dict[str, PyTree]:
    """Extract an ordered mapping of field names to values from a dataclass instance.

    Args:
        instance: Dataclass instance to inspect.

    Returns:
        Dictionary mapping each declared dataclass field name to its current value.
    """
    return dict(zip(_get_field_names(instance), _get_field_values(instance)))


class MappableDataclassMixin(collections.abc.Mapping[str, PyTree]):
    """Mixin class providing dictionary-like Mapping behavior and MappableDataclassProtocol."""

    def __chex_mappable_fields__(self) -> tuple[str, ...]:
        """Return the ordered tuple of declared dataclass field names.

        Returns:
            Tuple of field name strings.
        """
        return _get_field_names(self)

    def __chex_field_values__(self) -> tuple[PyTree, ...]:
        """Return the ordered tuple of current dataclass field values.

        Returns:
            Tuple of field values.
        """
        return _get_field_values(self)

    def __getitem__(self, key: str) -> PyTree:
        """Retrieve a dataclass field value by field name.

        Args:
            key: The field name to look up.

        Returns:
            The value of the requested field.

        Raises:
            KeyError: If key is not a valid field name.
        """
        field_map = dict(zip(self.__chex_mappable_fields__(), self.__chex_field_values__()))
        if key not in field_map:
            raise KeyError(key)
        return field_map[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the field names of the dataclass.

        Returns:
            An iterator of field name strings.
        """
        return iter(self.__chex_mappable_fields__())

    def __len__(self) -> int:
        """Return the number of declared fields in the dataclass.

        Returns:
            Integer count of dataclass fields.
        """
        return len(self.__chex_mappable_fields__())

    def replace(self: T, **changes: PyTree) -> T:
        """Return a new dataclass instance with specified fields updated.

        Args:
            **changes: Field names and their replacement values.

        Returns:
            A new dataclass instance with updated values.

        Raises:
            TypeError: If an unknown field name is provided in changes.
        """
        cls_type: type[T] = type(self)
        assert isinstance(self, (MappableDataclassProtocol, DataclassInstanceProtocol))
        field_map = _extract_dataclass_fields(self)
        for k, v in changes.items():
            if k not in field_map:
                raise TypeError(f"{cls_type.__name__}.replace() got an unexpected keyword argument {k!r}")
            field_map[k] = v
        return cls_type(**field_map)

    def to_tuple(self) -> tuple[tuple[PyTree, ...], tuple[str, ...]]:
        """Convert the dataclass instance into a tuple of (field_values, field_names).

        Returns:
            Tuple of (field_values, field_names).
        """
        return self.__chex_field_values__(), self.__chex_mappable_fields__()

    @classmethod
    def from_tuple(cls: type[T], args: tuple[tuple[PyTree, ...], tuple[str, ...]]) -> T:
        """Reconstruct a dataclass instance from a (field_values, field_names) tuple.

        Args:
            args: Tuple of (field_values, field_names) produced by to_tuple().

        Returns:
            Reconstructed dataclass instance.
        """
        values, keys = args
        kwargs = dict(zip(keys, values))
        return cls(**kwargs)


def _make_kw_only_dataclass(cls: type[T]) -> type[T]:
    """Create a keyword-only subclass of a dataclass without mutating cls.__init__ in-place.

    Args:
        cls: Dataclass type whose __init__ should require keyword-only arguments.

    Returns:
        Subclass of cls whose __init__ rejects positional arguments.
    """
    orig_init = cls.__init__

    def _kw_only_init(self_inst: T, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize dataclass instance enforcing keyword-only arguments.

        Args:
            self_inst: Instance being initialized.
            *args: Disallowed positional arguments.
            **kwargs: Keyword field values.

        Raises:
            TypeError: If any positional arguments are provided.
        """
        if args:
            raise TypeError(f"{cls.__name__}.__init__() takes 1 positional argument but {len(args) + 1} were given")
        orig_init(self_inst, **kwargs)

    namespace = {
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
        "__init__": _kw_only_init,
    }
    derived: type[T] = type(cls.__name__, (cls,), namespace)
    return derived


def _is_registrable_dataclass_cls(cls: type[T]) -> TypeGuard[CustomDataclassType]:
    """Verify that cls is a dataclass or MappableDataclassProtocol class suitable for PyTree registration.

    Args:
        cls: Class to check.

    Returns:
        True if cls is a dataclass or MappableDataclassProtocol subclass.
    """
    return dataclasses.is_dataclass(cls) or issubclass(cls, MappableDataclassProtocol)


def _register_dataclass_pytree(dc_cls: type[T]) -> None:
    """Register a dataclass class with zero_chex and zero_jax PyTree registries.

    Args:
        dc_cls: The dataclass class to register.
    """

    def _flatten(
        obj: MappableDataclassProtocol | DataclassInstanceProtocol,
    ) -> tuple[tuple[PyTree, ...], tuple[str, ...]]:
        """Flatten a dataclass instance into a tuple of field values and field names.

        Args:
            obj: The dataclass instance to flatten.

        Returns:
            Tuple of (field_values, field_names).
        """
        field_map = _extract_dataclass_fields(obj)
        keys = tuple(field_map.keys())
        values = tuple(field_map.values())
        return values, keys

    def _unflatten(
        aux_data: tuple[str, ...],
        children: Sequence[PyTree],
    ) -> PyTree:
        """Reconstruct a dataclass instance from field names and flattened children.

        Args:
            aux_data: Tuple of field names.
            children: Sequence of reconstructed child values.

        Returns:
            New dataclass instance.
        """
        kwargs = dict(zip(aux_data, children))
        inst = functools.partial(dc_cls, **kwargs)()
        assert isinstance(inst, (MappableDataclassProtocol, DataclassInstanceProtocol))
        return inst

    assert _is_registrable_dataclass_cls(dc_cls)
    register_custom_pytree_node(dc_cls, _flatten, _unflatten)


def _apply_mappable_dataclass(
    cls: type[T],
    kw_only: bool = False,
    register_pytree: bool = False,
) -> type[T]:
    """Augment a dataclass with MappableDataclassMixin methods and optional PyTree registration.

    Args:
        cls: The dataclass type to augment.
        kw_only: Whether positional arguments should be disallowed in __init__.
        register_pytree: Whether to also register the resulting class in the PyTree registry.

    Returns:
        A dataclass type supporting collections.abc.Mapping and MappableDataclassProtocol.
    """
    if not dataclasses.is_dataclass(cls):
        cls = dataclasses.dataclass(cls)
    if issubclass(cls, MappableDataclassMixin):
        if kw_only:
            cls = _make_kw_only_dataclass(cls)
        if register_pytree:
            _register_dataclass_pytree(cls)
        return cls

    collections.abc.Mapping.register(cls)
    orig_init = cls.__init__

    def _kw_only_init(self_inst: T, *args: PyTree, **kwargs: PyTree) -> None:
        """Initialize dataclass instance enforcing keyword-only arguments.

        Args:
            self_inst: Instance being initialized.
            *args: Disallowed positional arguments.
            **kwargs: Keyword field values.

        Raises:
            TypeError: If any positional arguments are provided.
        """
        if args:
            raise TypeError(f"{cls.__name__}.__init__() takes 1 positional argument but {len(args) + 1} were given")
        orig_init(self_inst, **kwargs)

    namespace: dict[str, str | _KwOnlyInitProtocol[T] | None] = {
        "__module__": cls.__module__,
        "__qualname__": cls.__qualname__,
        "__doc__": cls.__doc__,
    }
    if kw_only:
        namespace["__init__"] = _kw_only_init
    bases: tuple[type[T] | type[MappableDataclassMixin], ...] = (cls, MappableDataclassMixin)
    derived: type[T] = type(cls.__name__, bases, namespace)
    if register_pytree:
        _register_dataclass_pytree(derived)
    return derived


def mappable_dataclass(cls: type[T]) -> type[T]:
    """Augment a dataclass with dictionary-like mapping behavior (__getitem__, __iter__, __len__).

    Args:
        cls: The dataclass type to decorate.

    Returns:
        The augmented dataclass supporting mapping methods and MappableDataclassProtocol.
    """
    return _apply_mappable_dataclass(cls)


def register_dataclass_type_with_jax_tree_util(data_class: type[T]) -> type[T]:
    """Register a dataclass type with the zero_chex and zero_jax PyTree registries.

    Args:
        data_class: The dataclass type to register as a PyTree container node.

    Returns:
        The registered dataclass type.
    """
    if not dataclasses.is_dataclass(data_class):
        data_class = dataclasses.dataclass(data_class)
    _register_dataclass_pytree(data_class)
    return data_class


@overload
def dataclass(
    cls: type[T],
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    kw_only: bool = False,
    mappable_dataclass: bool = True,
    registered_with_pytree: bool = True,
) -> type[T]: ...


@overload
def dataclass(
    cls: None = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    kw_only: bool = False,
    mappable_dataclass: bool = True,
    registered_with_pytree: bool = True,
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform()
def dataclass(
    cls: type[T] | None = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    kw_only: bool = False,
    mappable_dataclass: bool = True,
    registered_with_pytree: bool = True,
) -> type[T] | Callable[[type[T]], type[T]]:
    """Decorate a class as a Chex dataclass with optional mapping and PyTree support.

    Args:
        cls: The class to transform into a dataclass, or None when called with kwargs.
        init: Whether to generate an __init__ method.
        repr: Whether to generate a __repr__ method.
        eq: Whether to generate an __eq__ method.
        order: Whether to generate rich comparison methods.
        unsafe_hash: Whether to force generation of a __hash__ method.
        frozen: Whether field assignments should raise FrozenInstanceError.
        kw_only: Whether fields should be keyword-only in __init__.
        mappable_dataclass: Whether to attach dictionary-style mapping methods.
        registered_with_pytree: Whether to register the dataclass in the PyTree registry.

    Returns:
        The decorated dataclass type or a decorator function if cls is None.
    """
    use_mappable = mappable_dataclass

    def _decorate(target_cls: type[T]) -> type[T]:
        """Apply dataclass transformation, mappable mixin, and PyTree registration.

        Args:
            target_cls: The target class to decorate.

        Returns:
            The transformed class.
        """
        dc_cls: type[T] = dataclasses.dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )(target_cls)
        if use_mappable:
            return _apply_mappable_dataclass(
                dc_cls,
                kw_only=kw_only,
                register_pytree=registered_with_pytree,
            )
        if kw_only:
            dc_cls = _make_kw_only_dataclass(dc_cls)
        if registered_with_pytree:
            return register_dataclass_type_with_jax_tree_util(dc_cls)
        return dc_cls

    if cls is None:
        return _decorate
    return _decorate(cls)


__all__ = [
    "DataclassInstanceProtocol",
    "MappableDataclassMixin",
    "dataclass",
    "mappable_dataclass",
    "register_dataclass_type_with_jax_tree_util",
]
