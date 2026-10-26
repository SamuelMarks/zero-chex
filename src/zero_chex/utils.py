"""Dimensions mapping class, parameter cartesian products, and deprecation helpers."""

from __future__ import annotations

import collections.abc
import functools
import inspect
import itertools
import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Optional, TypeVar, Union, overload

from typing_extensions import ParamSpec

from zero_chex._src.asserts_internal import get_err_regex
from zero_chex.types import PyTree

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

DimSizeValue = Union[int, None]
DimInputSpec = Union[int, None, Sequence[Union[int, None]]]


class _DimLetterDescriptor:
    """Descriptor providing attribute-style access to a single-character dimension on Dimensions."""

    def __init__(self, letter: str) -> None:
        """Initialize descriptor for a single dimension letter.

        Args:
            letter: Single-character dimension name.
        """
        self._letter = letter

    @overload
    def __get__(self, instance: None, owner: type[Dimensions] | None = None) -> _DimLetterDescriptor: ...

    @overload
    def __get__(self, instance: Dimensions, owner: type[Dimensions] | None = None) -> int | None: ...

    def __get__(
        self,
        instance: Dimensions | None,
        owner: type[Dimensions] | None = None,
    ) -> _DimLetterDescriptor | int | None:
        """Retrieve the dimension size from the Dimensions instance.

        Args:
            instance: The Dimensions instance, or None when accessed on the class.
            owner: The Dimensions class.

        Returns:
            The stored dimension integer/None, or self when accessed on the class.

        Raises:
            AttributeError: If the dimension letter is not currently set on instance.
        """
        _ = owner
        if instance is None:
            return self
        if self._letter in instance._dims:
            return instance._dims[self._letter]
        raise AttributeError(self._letter)

    def __set__(self, instance: Dimensions, value: DimInputSpec) -> None:
        """Assign a dimension size on the Dimensions instance.

        Args:
            instance: The Dimensions instance.
            value: The integer, None, or sequence size specification.
        """
        instance._set_dim_item(self._letter, value)

    def __delete__(self, instance: Dimensions) -> None:
        """Delete a dimension letter from the Dimensions instance.

        Args:
            instance: The Dimensions instance.

        Raises:
            AttributeError: If the dimension letter is not currently set on instance.
        """
        if self._letter not in instance._dims:
            raise AttributeError(self._letter)
        del instance[self._letter]


def _tokenize_dim_pattern(key: str) -> list[str]:
    """Tokenize a left-hand dimension assignment pattern into single chars, '*', or multi-digit literals.

    Args:
        key: Pattern string such as 'BTN', 'X*2', or 'B12'.

    Returns:
        Ordered list of tokens where consecutive digits are grouped into a single token.
    """
    tokens: list[str] = []
    i = 0
    while i < len(key):
        if key[i].isdigit():
            j = i
            while j < len(key) and key[j].isdigit():
                j += 1
            tokens.append(key[i:j])
            i = j
        else:
            tokens.append(key[i])
            i += 1
    return tokens


class Dimensions(collections.abc.Mapping[str, tuple[Optional[int], ...]]):
    """Symbolic dimension size dictionary supporting compact string shape expressions.

    A Dimensions mapping stores single-character dimension sizes (e.g., B=2, T=4, N=8)
    and expands multi-character lookup keys such as dims["BTN"] into shape tuples (2, 4, 8),
    parenthesized product expressions such as dims["(BT)N"] into (8, 8), numeric digit
    literals such as dims["B3"] into (2, 3), and wildcard '*' into None.
    """

    _HAS_DYNAMIC_ATTRIBUTES: bool = True

    A = _DimLetterDescriptor("A")
    B = _DimLetterDescriptor("B")
    C = _DimLetterDescriptor("C")
    D = _DimLetterDescriptor("D")
    E = _DimLetterDescriptor("E")
    F = _DimLetterDescriptor("F")
    G = _DimLetterDescriptor("G")
    H = _DimLetterDescriptor("H")
    I = _DimLetterDescriptor("I")  # noqa: E741
    J = _DimLetterDescriptor("J")
    K = _DimLetterDescriptor("K")
    L = _DimLetterDescriptor("L")
    M = _DimLetterDescriptor("M")
    N = _DimLetterDescriptor("N")
    O = _DimLetterDescriptor("O")  # noqa: E741
    P = _DimLetterDescriptor("P")
    Q = _DimLetterDescriptor("Q")
    R = _DimLetterDescriptor("R")
    S = _DimLetterDescriptor("S")
    T = _DimLetterDescriptor("T")
    U = _DimLetterDescriptor("U")
    V = _DimLetterDescriptor("V")
    W = _DimLetterDescriptor("W")
    X = _DimLetterDescriptor("X")
    Y = _DimLetterDescriptor("Y")
    Z = _DimLetterDescriptor("Z")
    a = _DimLetterDescriptor("a")
    b = _DimLetterDescriptor("b")
    c = _DimLetterDescriptor("c")
    d = _DimLetterDescriptor("d")
    e = _DimLetterDescriptor("e")
    f = _DimLetterDescriptor("f")
    g = _DimLetterDescriptor("g")
    h = _DimLetterDescriptor("h")
    i = _DimLetterDescriptor("i")
    j = _DimLetterDescriptor("j")
    k = _DimLetterDescriptor("k")
    l = _DimLetterDescriptor("l")  # noqa: E741
    m = _DimLetterDescriptor("m")
    n = _DimLetterDescriptor("n")
    o = _DimLetterDescriptor("o")
    p = _DimLetterDescriptor("p")
    q = _DimLetterDescriptor("q")
    r = _DimLetterDescriptor("r")
    s = _DimLetterDescriptor("s")
    t = _DimLetterDescriptor("t")
    u = _DimLetterDescriptor("u")
    v = _DimLetterDescriptor("v")
    w = _DimLetterDescriptor("w")
    x = _DimLetterDescriptor("x")
    y = _DimLetterDescriptor("y")
    z = _DimLetterDescriptor("z")

    def __init__(
        self,
        dim_sizes: Mapping[str, int | None] | None = None,
        **dim_args: int | None,
    ) -> None:
        """Initialize a Dimensions mapping with initial single-character dimension sizes.

        Args:
            dim_sizes: Optional mapping of single-character dimension names to sizes.
            **dim_args: Single-character dimension names mapped to positive integers or None.

        Raises:
            TypeError: If any dimension key is not a single character.
            ValueError: If any integer dimension size is <= 0.
        """
        self._dims: dict[str, int | None] = {}
        combined: dict[str, int | None] = {}
        if dim_sizes is not None:
            combined.update(dim_sizes)
        combined.update(dim_args)
        for k, v in combined.items():
            self._validate_key(k)
            self._validate_dim(v)
            self._dims[k] = v

    def _validate_dim(self, val: DimInputSpec) -> None:
        """Validate that a dimension value is a positive integer or None.

        Args:
            val: Dimension size to validate.

        Raises:
            TypeError: If val is neither an int nor None.
            ValueError: If val is an integer <= 0.
        """
        if val is not None and (not isinstance(val, int) or isinstance(val, bool)):
            raise TypeError(f"Dimension must be int or None, got {type(val).__name__}")
        if isinstance(val, int) and val <= 0:
            raise ValueError(f"Dimension must be > 0, got {val}")

    def _validate_key(self, k: str | int) -> None:
        """Validate that a dimension key is a single-character string.

        Args:
            k: Key to validate.

        Raises:
            TypeError: If k is not a string of length 1.
        """
        if not isinstance(k, str) or len(k) != 1:
            raise TypeError(f"Dimension key must be a single-char string, got {k!r}")

    def __getitem__(self, key: str | int) -> tuple[int | None, ...]:
        """Expand a symbolic shape expression string into a tuple of dimension sizes.

        Args:
            key: Symbolic shape string (e.g., 'NBT', '(BT)N', 'B12', '*').

        Returns:
            Tuple of resolved integer or None dimension sizes.

        Raises:
            TypeError: If key is not a string.
            ValueError: If parentheses are unbalanced, nested, empty, or multiply by None.
            KeyError: If an unknown dimension character is referenced.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key).__name__}")
        res: list[int | None] = []
        i = 0
        while i < len(key):
            if key[i] == "(":
                j = key.find(")", i)
                if j == -1:
                    raise ValueError(f"Unmatched '(' in key {key!r}")
                sub = key[i + 1 : j]
                if "(" in sub:
                    raise ValueError(f"Nested '(' in key {key!r}")
                if not sub:
                    raise ValueError(f"Empty '()' in key {key!r}")
                sub_dims = self[sub]
                if any(d is None for d in sub_dims):
                    raise ValueError("Cannot multiply None dimensions.")
                int_dims = [d for d in sub_dims if d is not None]
                res.append(math.prod(int_dims))
                i = j + 1
            elif key[i] == ")":
                raise ValueError(f"Unmatched ')' in key {key!r}")
            elif key[i].isdigit():
                j = i
                while j < len(key) and key[j].isdigit():
                    j += 1
                res.append(int(key[i:j]))
                i = j
            elif key[i] == "*":
                res.append(None)
                i += 1
            elif key[i] in self._dims:
                res.append(self._dims[key[i]])
                i += 1
            else:
                raise KeyError(key[i])
        return tuple(res)

    def _set_dim_item(
        self,
        key: str | int,
        value: DimInputSpec,
    ) -> None:
        """Assign one or more dimension sizes from an integer, None, or sequence.

        Args:
            key: Single-character key, multi-character pattern, or '__'-prefixed name.
            value: Integer, None, or sequence of dimension sizes matching key.

        Raises:
            TypeError: If key is not a string or value has an invalid type.
            ValueError: If key and value lengths mismatch or a dimension value conflicts.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key).__name__}")
        if key.startswith("__"):
            stripped_key = key[2:]
            self._validate_key(stripped_key)
            if isinstance(value, (tuple, list)):
                if any(v is None or not isinstance(v, int) or isinstance(v, bool) for v in value):
                    raise ValueError("Cannot multiply None or non-int dimensions in '__' assignment.")
                prod_val = math.prod(v for v in value if isinstance(v, int))
                self._validate_dim(prod_val)
                self._dims[stripped_key] = prod_val
                return
            key = stripped_key

        tokens = _tokenize_dim_pattern(key)
        if isinstance(value, (int, type(None))) and not isinstance(value, bool):
            value_seq: tuple[int | None, ...] = (value,)
        elif isinstance(value, Sequence):
            value_seq = tuple(value)
        else:
            raise TypeError(f"Invalid dimension value type: {type(value).__name__}")
        if len(tokens) != len(value_seq):
            raise ValueError(f"Key {key!r} and value {value_seq!r} must have the same length.")
        for tok, v in zip(tokens, value_seq):
            if tok == "*":
                continue
            if tok.isdigit():
                if int(tok) != v:
                    raise ValueError(f"Cannot assign {v} to digit literal {tok!r}.")
                continue
            self._validate_key(tok)
            self._validate_dim(v)
            if tok in self._dims and self._dims[tok] != v and self._dims[tok] is not None:
                if v is not None:
                    raise ValueError(f"Conflicting assignment for dimension {tok!r}: {self._dims[tok]} vs {v}.")
                self._dims[tok] = None
            else:
                self._dims[tok] = v

    def __setitem__(
        self,
        key: str | int,
        value: DimInputSpec,
    ) -> None:
        """Assign one or more dimension sizes from an integer, None, or sequence.

        Args:
            key: Single-character key, multi-character pattern, or '__'-prefixed name.
            value: Integer, None, or sequence of dimension sizes matching key.
        """
        self._set_dim_item(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete a single-character dimension from the mapping.

        Args:
            key: Single-character dimension name to remove.

        Raises:
            KeyError: If key does not exist in the mapping.
        """
        if key not in self._dims:
            raise KeyError(key)
        del self._dims[key]

    def clear(self) -> None:
        """Remove all stored dimensions from the mapping."""
        self._dims.clear()

    def __iter__(self) -> Iterator[str]:
        """Iterate over stored single-character dimension keys.

        Returns:
            Iterator over dimension key strings.
        """
        return iter(self._dims)

    def __len__(self) -> int:
        """Return the number of stored single-character dimensions.

        Returns:
            Integer count of stored dimensions.
        """
        return len(self._dims)

    def size(self, key: str) -> int:
        """Compute the total product of dimensions for a symbolic shape expression.

        Args:
            key: Symbolic shape string (e.g., 'NBT').

        Returns:
            Product of all dimensions in self[key].

        Raises:
            ValueError: If any dimension in self[key] is None or <= 0.
        """
        dims = self[key]
        if any(d is None or d <= 0 for d in dims):
            raise ValueError(f"Invalid size for key {key!r} with dimensions {dims!r}.")
        int_dims = [d for d in dims if d is not None]
        return math.prod(int_dims)


collections.abc.MutableMapping.register(Dimensions)


def params_product(
    *params_lists: Sequence[PyTree] | Mapping[str, PyTree],
    named: bool = False,
) -> list[tuple[PyTree, ...]]:
    """Generate the cartesian product of multiple parameter sequences for parameterized tests.

    Args:
        *params_lists: Parameter sequences or named parameter mappings.
        named: If True, prefixes each generated tuple with a joined test case name string.

    Returns:
        List of parameter tuples representing the cartesian product.
    """
    if named:
        normalized_lists: list[list[tuple[str, PyTree]]] = []
        for p_list in params_lists:
            if isinstance(p_list, Mapping):
                normalized_lists.append([(str(k), v) for k, v in p_list.items()])
            else:
                sub_items: list[tuple[str, PyTree]] = []
                for item in p_list:
                    if isinstance(item, (tuple, list)) and len(item) == 2:
                        sub_items.append((str(item[0]), item[1]))
                    else:
                        sub_items.append((str(item), item))
                normalized_lists.append(sub_items)
        named_res: list[tuple[PyTree, ...]] = []
        for named_prod in itertools.product(*normalized_lists):
            names = [str(pair[0]) for pair in named_prod]
            vals: list[PyTree] = []
            for pair in named_prod:
                v = pair[1]
                if isinstance(v, (list, tuple)):
                    vals.extend(v)
                else:
                    vals.append(v)
            named_res.append(("_".join(names), *vals))
        return named_res

    unnamed_res: list[tuple[PyTree, ...]] = []
    seq_lists: list[Sequence[PyTree]] = [list(p.values()) if isinstance(p, Mapping) else p for p in params_lists]
    for unnamed_prod in itertools.product(*seq_lists):
        flat_vals: list[PyTree] = []
        for v in unnamed_prod:
            if isinstance(v, (list, tuple)):
                flat_vals.extend(v)
            else:
                flat_vals.append(v)
        unnamed_res.append(tuple(flat_vals))
    return unnamed_res


def if_args_not_none(
    fn: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R | None:
    """Invoke fn(*args, **kwargs) only if none of the positional arguments are None.

    Args:
        fn: The callable to invoke conditionally.
        *args: Positional arguments checked for None and passed to fn.
        **kwargs: Keyword arguments passed to fn.

    Returns:
        The return value of fn(*args, **kwargs), or None if any positional arg is None.
    """
    if any(a is None for a in args):
        return None
    return fn(*args, **kwargs)


def create_deprecated_function_alias(
    fun: Callable[P, R],
    new_name: str,
    deprecated_alias: str = "",
    *args: str,
    **kwargs: str,
) -> Callable[P, R]:
    """Create a wrapper function that emits a DeprecationWarning pointing to new_name.

    Args:
        fun: The target function to alias.
        new_name: Name of the replacement function to mention in the warning.
        deprecated_alias: Optional name of the deprecated alias.
        *args: Additional positional metadata strings.
        **kwargs: Additional keyword metadata strings.

    Returns:
        Wrapped callable that issues a DeprecationWarning when called.
    """
    _ = (args, kwargs)
    prefix = f"Function {deprecated_alias} is deprecated" if deprecated_alias else "This function is deprecated"

    @functools.wraps(fun)
    def _wrapper(*f_args: P.args, **f_kwargs: P.kwargs) -> R:
        """Emit DeprecationWarning and forward call to fun.

        Args:
            *f_args: Positional arguments forwarded to fun.
            **f_kwargs: Keyword arguments forwarded to fun.

        Returns:
            Result of fun(*f_args, **f_kwargs).
        """
        warnings.warn(
            f"{prefix}; please use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return fun(*f_args, **f_kwargs)

    return _wrapper


@overload
def warn_deprecated_function(
    fun: Callable[P, R],
    replacement: str | None = None,
) -> Callable[P, R]: ...


@overload
def warn_deprecated_function(
    fun: None = None,
    replacement: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def warn_deprecated_function(
    fun: Callable[P, R] | None = None,
    replacement: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to emit a DeprecationWarning whenever it is called.

    Args:
        fun: The deprecated function, or None when called with keyword arguments.
        replacement: Optional name of the recommended replacement function.

    Returns:
        Decorated callable or a decorator function.
    """
    if fun is None:

        def _decorator(f: Callable[P, R]) -> Callable[P, R]:
            """Apply deprecation warning wrapper to f.

            Args:
                f: Function to wrap.

            Returns:
                Wrapped function.
            """
            return create_deprecated_function_alias(f, replacement or "")

        return _decorator
    return create_deprecated_function_alias(fun, replacement or "")


@overload
def warn_only_n_pos_args_in_future(
    fun: Callable[P, R],
    n: int = 1,
) -> Callable[P, R]: ...


@overload
def warn_only_n_pos_args_in_future(
    fun: None = None,
    n: int = 1,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def warn_only_n_pos_args_in_future(
    fun: Callable[P, R] | None = None,
    n: int = 1,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to emit a FutureWarning if more than n positional args are passed.

    Args:
        fun: The function to wrap, or None when parameterized with n.
        n: Maximum number of positional arguments that will be allowed in the future.

    Returns:
        Wrapped callable or decorator function.
    """
    if fun is None:

        def _decorator(f: Callable[P, R]) -> Callable[P, R]:
            """Apply positional argument count warning wrapper to f.

            Args:
                f: Function to wrap.

            Returns:
                Wrapped function.
            """
            return warn_only_n_pos_args_in_future(f, n=n)

        return _decorator

    sig = inspect.signature(fun)
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    max_pos_params = sum(
        1
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )

    @functools.wraps(fun)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Check positional argument count and forward call to fun.

        Args:
            *args: Positional arguments passed by the caller.
            **kwargs: Keyword arguments passed by the caller.

        Returns:
            Result of fun(*args, **kwargs).

        Raises:
            TypeError: If more positional arguments are given than fun accepts.
        """
        if len(args) > n:
            if not has_varargs and len(args) > max_pos_params:
                raise TypeError(f"{fun.__name__}() takes at most {max_pos_params} arguments ({len(args)} given)")
            warnings.warn(
                f"Only {n} positional arguments will be allowed in the future.",
                FutureWarning,
                stacklevel=2,
            )
        return fun(*args, **kwargs)

    return _wrapper


@overload
def warn_keyword_args_only_in_future(
    fun: Callable[P, R],
    *,
    n: int = 0,
) -> Callable[P, R]: ...


@overload
def warn_keyword_args_only_in_future(
    fun: None = None,
    *,
    n: int = 0,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def warn_keyword_args_only_in_future(
    fun: Callable[P, R] | None = None,
    *,
    n: int = 0,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function to emit a FutureWarning if any positional arguments are passed.

    Args:
        fun: The function to wrap, or None when called with parentheses.
        n: Maximum number of positional arguments allowed (defaults to 0).

    Returns:
        Wrapped callable or decorator function.
    """
    return warn_only_n_pos_args_in_future(fun, n=n)


__all__ = [
    "DimInputSpec",
    "DimSizeValue",
    "Dimensions",
    "create_deprecated_function_alias",
    "get_err_regex",
    "if_args_not_none",
    "params_product",
    "warn_deprecated_function",
    "warn_keyword_args_only_in_future",
    "warn_only_n_pos_args_in_future",
]
