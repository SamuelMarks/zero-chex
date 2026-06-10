import collections


class ChexVariantType:
    WITH_JIT = "_with_jit"
    WITHOUT_JIT = "_without_jit"
    WITH_PMAP = "_with_pmap"
    WITHOUT_PMAP = "_without_pmap"


class Dimensions(collections.abc.MutableMapping):
    def __init__(self, **kwargs):
        super().__setattr__("_dims", kwargs)

    def __setattr__(self, key, value):
        if key == "_dims":
            super().__setattr__(key, value)
        else:
            self._dims[key] = value

    def __getattr__(self, key):
        if key in self._dims:
            return self._dims[key]
        raise AttributeError(key)

    def _validate_dim(self, key):
        if not isinstance(key, str):
            raise TypeError("dimension name must be a string")
        if not key.isalpha() and not key == "_":
            raise KeyError("contain letters")

    def __setitem__(self, key, value):
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
        if key == "*":
            return None
        if key.isdigit():
            return int(key)
        if key not in self._dims:
            raise KeyError(key)
        return self._dims[key]

    def __getitem__(self, key):
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
        if not isinstance(key, str):
            raise TypeError("key must be a string")
        self._deldim(key)

    def _deldim(self, key):
        if key not in self._dims:
            if key == "_":
                return
            raise KeyError(key)
        del self._dims[key]

    def __iter__(self):
        return iter(self._dims)

    def __len__(self):
        return len(self._dims)

    def __repr__(self):
        items = ", ".join(f"{k}={v}" for k, v in sorted(self._dims.items()))
        return f"Dimensions({items})"

    def size(self, key):
        import math

        res = self.__getitem__(key)
        for r in res:
            if r is None:
                raise ValueError("cannot take product")
            if r == 0:
                raise ValueError("cannot take product")
        return math.prod(res)


class TestCase:
    def variant(self):
        raise RuntimeError("self.variant is not defined")
