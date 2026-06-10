# ruff: noqa: F821, F403, F405, E402, E731, F841, E721
"""Tests for type aliases."""

import zero_jax as jax
from ml_switcheroo.core.tensor import Tensor


from zero_chex import (
    ArrayBatched,
    ArrayDevice,
    ArrayNumpy,
    ArraySharded,
    Device,
    PRNGKey,
    PyTreeDef,
)


def test_array_batched() -> None:
    """Test ArrayBatched."""
    assert ArrayBatched is jax.Array


def test_array_device() -> None:
    """Test ArrayDevice."""
    assert ArrayDevice is jax.Array


def test_array_numpy() -> None:
    """Test ArrayNumpy."""
    assert ArrayNumpy is Tensor


def test_array_sharded() -> None:
    """Test ArraySharded."""
    assert ArraySharded is jax.Array


def test_device() -> None:
    """Test Device."""
    from typing import Any

    assert Device is Any


def test_prng_key() -> None:
    """Test PRNGKey."""
    from typing import Any

    assert PRNGKey is Any


def test_py_tree_def() -> None:
    """Test PyTreeDef."""
    assert PyTreeDef is jax.tree_util.PyTreeDef
