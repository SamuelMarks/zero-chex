# ruff: noqa: F821, F403, F405, E402, E731, F841, E721

"""Module docstring."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import *

    NoneType = type(None)
    JaxException = Exception

    class ChexifyChecks:
        """Docstring."""

        user = None


"""Zero-chex variants and classes."""

import math
import re
from typing import Any, Collection, Dict, Optional, Sized, Tuple


class ChexVariantType:
    """An enumeration of available Chex variants.

    Use ``self.variant.type`` to get type of the current test variant.
    See the docstring of ``chex.variants`` for more information.
    """

    WITH_JIT = type("CV", (), {"name": "WITH_JIT", "__str__": lambda s: "_with_jit"})()
    WITHOUT_JIT = type(
        "CV", (), {"name": "WITHOUT_JIT", "__str__": lambda s: "_without_jit"}
    )()
    WITH_DEVICE = type(
        "CV", (), {"name": "WITH_DEVICE", "__str__": lambda s: "_with_device"}
    )()
    WITHOUT_DEVICE = type(
        "CV", (), {"name": "WITHOUT_DEVICE", "__str__": lambda s: "_without_device"}
    )()
    WITH_PMAP = type(
        "CV", (), {"name": "WITH_PMAP", "__str__": lambda s: "_with_pmap"}
    )()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Docstring."""
        pass  # pragma: no cover

    def __str__(self) -> str:
        """Docstring."""
        return "_" + self.name.lower()  # pragma: no cover


def _optional_int(size: Optional[int]) -> Optional[int]:
    """Docstring."""
    if size is None:
        return None
    return int(size)


class Dimensions:
    """A lightweight utility that maps strings to shape tuples."""

    def __init__(self, **dim_sizes) -> None:
        """Docstring."""
        for dim, size in dim_sizes.items():
            self._setdim(dim, size)

    def size(self, key: str) -> int:
        """Returns the flat size of a given named shape, i.e. prod(shape)."""
        shape = self[key]
        if any(size is None or size <= 0 for size in shape):
            raise ValueError(
                f"cannot take product of shape '{key}' = {shape}, "
                "because it contains non-positive sized dimensions"
            )
        return math.prod(shape)  # type: ignore

    def __getitem__(self, key: str) -> Tuple[Optional[int], ...]:
        """Docstring."""
        self._validate_key(key)
        shape = []
        open_parentheses = False
        dims_to_flatten = ""
        for dim in key:
            if dim == "(":
                if open_parentheses:
                    raise ValueError(
                        f"nested parentheses are unsupported; got: '{key}'"
                    )
                open_parentheses = True
            elif dim == ")":
                if not open_parentheses:
                    raise ValueError(f"unmatched parentheses in named shape: '{key}'")
                if not dims_to_flatten:
                    raise ValueError(f"found empty parentheses in named shape: '{key}'")
                shape.append(self.size(dims_to_flatten))
                open_parentheses = False
                dims_to_flatten = ""
            elif open_parentheses:
                dims_to_flatten += dim
            else:
                shape.append(self._getdim(dim))

        if open_parentheses:
            raise ValueError(f"unmatched parentheses in named shape: '{key}'")
        return tuple(shape)

    def __setitem__(self, key: str, value: Collection[Optional[int]]) -> None:
        """Docstring."""
        self._validate_key(key)
        self._validate_value(value)
        if len(key) != len(value):
            raise ValueError(
                f"key string {repr(key)} and shape {tuple(value)} "
                "have different lengths"
            )
        for dim, size in zip(key, value):
            self._setdim(dim, size)

    def __delitem__(self, key: str) -> None:
        """Docstring."""
        self._validate_key(key)
        for dim in key:
            self._deldim(dim)

    def __repr__(self) -> str:
        """Docstring."""
        args = ", ".join(f"{k}={v}" for k, v in sorted(self._asdict().items()))
        return f"{type(self).__name__}({args})"

    def _asdict(self) -> Dict[str, Optional[int]]:
        """Docstring."""
        return {k: v for k, v in self.__dict__.items() if re.fullmatch(r"[a-zA-Z]", k)}

    def _getdim(self, dim: str) -> Optional[int]:
        """Docstring."""
        if dim == "*":
            return None
        if re.fullmatch(r"[0-9]", dim):
            return int(dim)
        try:
            return getattr(self, dim)
        except AttributeError as e:
            raise KeyError(dim) from e

    def _setdim(self, dim: str, size: Optional[int]) -> None:
        """Docstring."""
        if dim == "_":  # Skip.
            return
        self._validate_dim(dim)
        setattr(self, dim, _optional_int(size))

    def _deldim(self, dim: str) -> None:
        """Docstring."""
        if dim == "_":  # Skip.
            return
        self._validate_dim(dim)
        try:
            delattr(self, dim)
        except AttributeError as e:
            raise KeyError(dim) from e

    def _validate_key(self, key: Any) -> None:
        """Docstring."""
        if not isinstance(key, str):
            raise TypeError(f"key must be a string; got: {type(key).__name__}")

    def _validate_value(self, value: Any) -> None:
        """Docstring."""
        if not isinstance(value, Sized):
            raise TypeError(
                "value must be sized, i.e. an object with a well-defined len(value); "
                f"got object of type: {type(value).__name__}"
            )

    def _validate_dim(self, dim: Any) -> None:
        """Docstring."""
        if not isinstance(dim, str):
            raise TypeError(
                f"dimension name must be a string; got: {type(dim).__name__}"
            )
        if not re.fullmatch(r"[a-zA-Z]", dim):
            raise KeyError(
                "dimension names may only be contain letters (or '_' to skip); "
                f"got dimension name: {repr(dim)}"
            )


# Dummy test case to mimic parameterized.TestCase without importing absl
class TestCase:
    """A class for Chex tests that use variants."""

    def variant(self, *args: Any, **kwargs: Any) -> Any:
        """Raises a RuntimeError if not overriden or redefined."""
        raise RuntimeError(
            "self.variant is not defined: forgot to wrap a test in @chex.variants?"
        )
