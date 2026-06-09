# ruff: noqa: F821, F403, F405, E402, E731, F841, E721

"""Module docstring."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    from typing import *

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        """Docstring."""

        user = None


"""Typing aliases for zero-chex."""

from typing import Any, Sequence, Set, Union

import zero_jax as jax
import numpy as np

# Array base class for JAX
ArrayBatched = jax.Array

# Array base class for JAX
ArrayDevice = jax.Array

# ArrayNumpy type alias
ArrayNumpy = np.ndarray

# Array base class for JAX
ArraySharded = jax.Array

# A descriptor of an available device.
Device = Any

# Array base class for JAX
PRNGKey = Any  # JAX doesn't strictly export PRNGKeyArray without random

# No docstring available.
PyTreeDef = jax.tree_util.PyTreeDef

# Internal aliases for typing matching
Array = Union[jax.Array, np.ndarray, np.bool_, np.number]
Scalar = Union[float, int]
ArrayDType = Union[str, type[Any], np.dtype, Any]

TDimMatcher = Union[int, Set[int], type(Ellipsis), type(None)]
TShapeMatcher = Sequence[TDimMatcher]
