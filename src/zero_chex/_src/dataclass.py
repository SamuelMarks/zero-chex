"""Internal dataclass module re-exporting dataclass utilities from zero_chex.dataclasses."""

from __future__ import annotations

from zero_chex.dataclasses import (
    dataclass,
    mappable_dataclass,
    register_dataclass_type_with_jax_tree_util,
)

__all__ = [
    "dataclass",
    "mappable_dataclass",
    "register_dataclass_type_with_jax_tree_util",
]
