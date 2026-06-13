"""Type definitions for the zero_chex library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zero_jax as jax
    from typing import Any

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        """Flags for enabling specific checks during execution.

        Attributes:
            user: User-defined checks configuration.
        """

        user = None


from typing import Any, Iterable, Mapping, Sequence, Set, Union
import zero_jax as jax
import ml_switcheroo_compiler

ArrayBatched = jax.Array
ArrayDevice = jax.Array
ArrayNumpy = ml_switcheroo_compiler.core.tensor.Tensor
ArraySharded = jax.Array
Device = Any
PRNGKey = Any
PyTreeDef = jax.tree_util.PyTreeDef

Array = Union[jax.Array, ml_switcheroo_compiler.core.tensor.Tensor, bool, int, float]
Scalar = Union[float, int]
ArrayDType = Union[str, type[Any], ml_switcheroo_compiler.core.dtype.DType, Any]
Numeric = Union[Array, Scalar]
Shape = Sequence[Union[int, Any]]

ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
ArrayDeviceTree = Union[
    ArrayDevice, Iterable["ArrayDeviceTree"], Mapping[Any, "ArrayDeviceTree"]
]
ArrayNumpyTree = Union[
    ArrayNumpy, Iterable["ArrayNumpyTree"], Mapping[Any, "ArrayNumpyTree"]
]

TDimMatcher = Union[int, Set[int], type(Ellipsis), type(None)]
TShapeMatcher = Sequence[TDimMatcher]
