"Module docstring."

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    from typing import Any

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        """Docstring."""

        user = None


from typing import Any, Sequence, Set, Union
import zero_jax as jax
import numpy as np

ArrayBatched = jax.Array
ArrayDevice = jax.Array
ArrayNumpy = np.ndarray
ArraySharded = jax.Array
Device = Any
PRNGKey = Any
PyTreeDef = jax.tree_util.PyTreeDef
Array = Union[jax.Array, np.ndarray, np.bool_, np.number]
Scalar = Union[float, int]
ArrayDType = Union[str, type[Any], np.dtype, Any]
TDimMatcher = Union[int, Set[int], type(Ellipsis), type(None)]
TShapeMatcher = Sequence[TDimMatcher]
