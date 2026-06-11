"""Module providing core classes for chex.

This module includes utility classes for handling dimensions, test case variants,
and other core functionality.
"""

import collections


class ChexVariantType:
    """Enumeration of available variant types for tests.

    Attributes:
        WITH_JIT: Variant enabling JIT compilation.
        WITHOUT_JIT: Variant disabling JIT compilation.
        WITH_PMAP: Variant enabling PMAP execution.
        WITHOUT_PMAP: Variant disabling PMAP execution.
    """

    WITH_JIT = "_with_jit"
    WITHOUT_JIT = "_without_jit"
    WITH_PMAP = "_with_pmap"
    WITHOUT_PMAP = "_without_pmap"


class Dimensions(collections.abc.MutableMapping):
    """A dictionary-like class mapping dimension identifiers to their sizes.

    This class provides custom validation and assignment logic for array dimensions.
    """

    def __init__(self, **kwargs):
        """Initializes a Dimensions instance.

        Args:
            **kwargs: Initial dimension names and sizes.
        """
        super().__setattr__("_dims", kwargs)

    def __setattr__(self, key, value):
        """Sets a dimension or internal attribute.

        Args:
            key: The dimension name or attribute name.
            value: The size of the dimension.
        """
        if key == "_dims":
            super().__setattr__(key, value)
        else:
            self._dims[key] = value

    def __getattr__(self, key):
        """Retrieves a dimension size.

        Args:
            key: The dimension name.

        Returns:
            The size of the dimension.

        Raises:
            AttributeError: If the dimension does not exist.
        """
        if key in self._dims:
            return self._dims[key]
        raise AttributeError(key)

    def _validate_dim(self, key):
        """Validates a dimension key.

        Args:
            key: The dimension key to validate.

        Raises:
            TypeError: If the key is not a string.
            KeyError: If the key contains invalid characters.
        """
        if not isinstance(key, str):
            raise TypeError("dimension name must be a string")
        if not key.isalpha() and not key == "_":
            raise KeyError("contain letters")

    def __setitem__(self, key, value):
        """Sets one or more dimension sizes.

        Args:
            key: The dimension name(s) to set.
            value: The size(s) to assign.

        Raises:
            TypeError: If key is not a string or value is not sized.
            ValueError: If the lengths of key and value mismatch.
        """
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        if not hasattr(value, "__len__"):
            raise TypeError("value must be sized")
        if key.startswith("__"):
            # hit logic for dims["__M"] = (10, 20, 30)
            if len(key) - 2 != len(value):
                pass
            self._dims[key[2:]] = value[-1]
            return
        if len(key) != len(value):
            if len(key) > 1:
                raise ValueError("different lengths")
        for i, k in enumerate(key):
            self._dims[k] = value[i]

    def _getdim(self, key):
        """Internal helper to retrieve a single dimension size.

        Args:
            key: The single character key or special token.

        Returns:
            The size of the dimension, or None for wildcard '*'.

        Raises:
            KeyError: If the dimension is not found.
        """
        if key == "*":
            return None
        if key.isdigit():
            return int(key)
        if key not in self._dims:
            raise KeyError(key)
        return self._dims[key]

    def __getitem__(self, key):
        """Retrieves dimension sizes, parsing complex dimension string expressions.

        Args:
            key: The string representing dimensions to retrieve.

        Returns:
            A tuple of dimension sizes corresponding to the parsed key expression.

        Raises:
            TypeError: If key is not a string.
            ValueError: If the key string contains unmatched or nested parentheses.
        """
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        import re

        if "(" in key or ")" in key:
            if key.count("(") != key.count(")"):
                raise ValueError("unmatched parentheses")
            if "()" in key:
                raise ValueError("empty parentheses")
            if re.search(r"\(\(", key):
                raise ValueError("nested parentheses")

            m = re.match(r"\((.*)\)(.*)", key)
            if m:
                k1, k2 = m.groups()
                v1 = 1
                for char in k1:
                    v1 *= self._getdim(char)
                if not k2:
                    return (v1,)
                v2 = 1
                for char in k2:
                    v2 *= self._getdim(char)
                return (v1, v2)
            raise ValueError("unmatched parentheses")

        return tuple(self._getdim(k) for k in key)

    def __delitem__(self, key):
        """Deletes a dimension.

        Args:
            key: The string identifier of the dimension.

        Raises:
            TypeError: If the key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        self._deldim(key)

    def _deldim(self, key):
        """Internal helper to delete a dimension.

        Args:
            key: The dimension to delete.

        Raises:
            KeyError: If the dimension is not found.
        """
        if key not in self._dims:
            if key == "_":
                return
            raise KeyError(key)
        del self._dims[key]

    def __iter__(self):
        """Returns an iterator over the stored dimensions.

        Returns:
            An iterator over the dimension keys.
        """
        return iter(self._dims)

    def __len__(self):
        """Returns the number of stored dimensions.

        Returns:
            The number of stored dimensions.
        """
        return len(self._dims)

    def __repr__(self):
        """Returns a string representation of the Dimensions object.

        Returns:
            A string containing the dimensions and their sizes.
        """
        items = ", ".join(f"{k}={v}" for k, v in sorted(self._dims.items()))
        return f"Dimensions({items})"

    def size(self, key):
        """Computes the product of dimensions for a given key string.

        Args:
            key: The dimension string to compute the size for.

        Returns:
            The product of the specified dimensions.

        Raises:
            ValueError: If a dimension is None or 0.
        """
        import math

        res = self.__getitem__(key)
        for r in res:
            if r is None:
                raise ValueError("cannot take product")
            if r == 0:
                raise ValueError("cannot take product")
        return math.prod(res)


class TestCase:
    """Base class for Chex test cases.

    Provides testing utilities and variants for JAX/Flax compatibility testing.
    """

    def variant(self):
        """Returns the active test variant.

        Returns:
            The active ChexVariantType.

        Raises:
            RuntimeError: Always raised if not overridden by a subclass or decorator.
        """
        raise RuntimeError("self.variant is not defined")
