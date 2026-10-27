"""Dataclass decorator module re-exporting dataclass utilities from zero_chex.dataclasses."""

from __future__ import annotations

from zero_chex.dataclasses import (
    DataclassInstanceProtocol,
    MappableDataclassMixin,
    dataclass,
    mappable_dataclass,
    register_dataclass_type_with_jax_tree_util,
)

__all__ = [
    "DataclassInstanceProtocol",
    "MappableDataclassMixin",
    "dataclass",
    "mappable_dataclass",
    "register_dataclass_type_with_jax_tree_util",
]
