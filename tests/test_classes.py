"""Tests for Dimensions, ChexVariantType, TestCase, @variants, and @all_variants."""

from __future__ import annotations

import pytest

from zero_chex import (
    ChexVariantType,
    Dimensions,
    TestCase,
    all_variants,
    restrict_backends,
    variants,
)
from zero_chex.utils import _DimLetterDescriptor
from zero_chex.variants import VariantDecorator, _select_variant_types


def test_chex_variant_type_enum() -> None:
    """Test ChexVariantType enum values, __str__, and _select_variant_types."""
    assert str(ChexVariantType.WITH_JIT) == "_with_jit"
    assert str(ChexVariantType.WITHOUT_JIT) == "_without_jit"
    assert str(ChexVariantType.WITH_DEVICE) == "_with_device"
    assert str(ChexVariantType.WITHOUT_DEVICE) == "_without_device"
    assert str(ChexVariantType.WITH_PMAP) == "_with_pmap"
    assert str(ChexVariantType.WITHOUT_PMAP) == "_without_pmap"

    selected = _select_variant_types(
        with_jit=False,
        without_jit=False,
        with_device=False,
        without_device=False,
        with_pmap=False,
        without_pmap=True,
    )
    assert selected == (ChexVariantType.WITHOUT_PMAP,)


def test_dimensions_comprehensive() -> None:
    """Test Dimensions initialization, indexing, assignment, deletion, descriptors, and error paths."""
    assert isinstance(Dimensions.B, _DimLetterDescriptor)
    dims = Dimensions({"A": 3}, B=2, T=4, N=8)
    assert dims["ABTN"] == (3, 2, 4, 8)
    assert dims["BTN"] == (2, 4, 8)
    assert dims["(BT)N"] == (8, 8)
    assert dims["B12"] == (2, 12)
    assert dims["B*N"] == (2, None, 8)
    assert dims.B == 2
    assert len(dims) == 4
    assert list(dims) == ["A", "B", "T", "N"]
    assert dims.size("BTN") == 64

    dims["__M"] = (2, 3, 4)
    assert dims.M == 24
    dims["__P"] = 7
    assert dims.P == 7
    with pytest.raises(ValueError, match="Cannot multiply None"):
        dims["__Q"] = (2, None)

    dims["X*12"] = (5, 99, 12)
    assert dims.X == 5
    dims.X = 5
    dims.X = None

    dims.K = None
    with pytest.raises(ValueError, match="Cannot multiply None"):
        _ = dims["(BK)"]
    with pytest.raises(ValueError, match="Invalid size"):
        dims.size("BK")

    del dims["N"]
    del dims.T
    with pytest.raises(KeyError):
        del dims["UNKNOWN"]
    with pytest.raises(AttributeError):
        del dims.Z
    with pytest.raises(AttributeError):
        _ = dims.Z

    with pytest.raises(TypeError, match="Dimension must be int or None"):
        dims._validate_dim(True)
    with pytest.raises(ValueError, match="Dimension must be > 0"):
        dims._validate_dim(0)
    with pytest.raises(TypeError, match="single-char string"):
        dims._validate_key("AB")
    with pytest.raises(TypeError, match="Key must be a string"):
        _ = dims[123]
    with pytest.raises(TypeError, match="Key must be a string"):
        dims[123] = 4
    with pytest.raises(ValueError, match=r"Unmatched '\('"):
        _ = dims["(B"]
    with pytest.raises(ValueError, match=r"Nested '\('"):
        _ = dims["((B))"]
    with pytest.raises(ValueError, match=r"Empty '\(\)'"):
        _ = dims["()"]
    with pytest.raises(ValueError, match=r"Unmatched '\)'"):
        _ = dims["B)"]
    with pytest.raises(KeyError):
        _ = dims["Z"]
    with pytest.raises(TypeError, match="Invalid dimension value type"):
        dims["Q"] = True
    with pytest.raises(ValueError, match="same length"):
        dims["AB"] = (1, 2, 3)
    with pytest.raises(ValueError, match="digit literal"):
        dims["12"] = 3
    with pytest.raises(ValueError, match="Conflicting assignment"):
        dims["B"] = 99


def test_variants_and_testcase() -> None:
    """Test TestCase, @variants, and @all_variants across single and multi-variant executions."""

    def _inc1(x: int) -> int:
        """Increment x by 1.

        Args:
            x: Input integer.

        Returns:
            x + 1.
        """
        return x + 1

    def _inc2(x: int) -> int:
        """Increment x by 2.

        Args:
            x: Input integer.

        Returns:
            x + 2.
        """
        return x + 2

    base_case = TestCase()
    with pytest.raises(RuntimeError, match="self.variant is not defined"):
        base_case.variant(_inc1)

    class SampleTest(TestCase):
        """Sample TestCase subclass exercising variant decorators."""

        def __init__(self) -> None:
            """Initialize SampleTest with a log of executed variant types."""
            super().__init__()
            self.seen_variants: list[ChexVariantType] = []

        @variants(with_jit=True)
        def run_jit(self) -> int:
            """Run with_jit variant."""
            fn = self.variant(_inc1)
            fn_kw = self.variant(static_argnums=0)(_inc2)
            assert self._active_variant is not None
            assert self._active_variant.type == ChexVariantType.WITH_JIT
            res = fn(1) + fn_kw(1)
            assert self._active_variant.call_count == 2
            return res

        @variants(without_jit=True)
        def run_without_jit(self) -> int:
            """Run without_jit variant."""
            assert self._active_variant is not None
            assert self._active_variant.type == ChexVariantType.WITHOUT_JIT
            fn = self.variant(_inc1)
            return fn(0)

        @variants(with_device=True)
        def run_with_device(self) -> int:
            """Run with_device variant."""
            assert self._active_variant is not None
            assert self._active_variant.type == ChexVariantType.WITH_DEVICE
            fn = self.variant(_inc1)
            return fn(1)

        @variants(without_device=True)
        def run_without_device(self) -> int:
            """Run without_device variant."""
            assert self._active_variant is not None
            assert self._active_variant.type == ChexVariantType.WITHOUT_DEVICE
            return 3

        @variants(with_pmap=True)
        def run_with_pmap(self) -> int:
            """Run with_pmap variant."""
            assert self._active_variant is not None
            assert self._active_variant.type == ChexVariantType.WITH_PMAP
            fn = self.variant(_inc1)
            return fn(3)

        @variants()
        def run_default(self) -> int:
            """Run default variant when all flags are False."""
            assert self._active_variant is not None
            assert self._active_variant.type == ChexVariantType.WITHOUT_PMAP
            return 5

        @variants
        def run_bare_variants(self) -> int:
            """Run bare @variants decorator."""
            return 6

        @all_variants
        def run_bare_all_variants(self) -> int:
            """Run bare @all_variants decorator and record each executed variant."""
            assert self._active_variant is not None
            self.seen_variants.append(self._active_variant.type)
            return len(self.seen_variants)

        @all_variants(with_pmap=False)
        def run_kw_all_variants(self) -> int:
            """Run parameterized @all_variants decorator."""
            assert self._active_variant is not None
            self.seen_variants.append(self._active_variant.type)
            return 8

        @variants(with_jit=True, without_jit=True)
        def run_parameterized(self, a: int, b: int = 10) -> int:
            """Run parameterized method across variants.

            Args:
                a: First integer parameter.
                b: Second integer parameter.

            Returns:
                a + b.
            """
            fn = self.variant(_inc1)
            return fn(a) + b

    inst = SampleTest()
    assert inst.run_jit() == 5
    assert inst.run_without_jit() == 1
    assert inst.run_with_device() == 2
    assert inst.run_without_device() == 3
    assert inst.run_with_pmap() == 4
    assert inst.run_default() == 5
    assert inst.run_bare_variants() == 6

    inst.seen_variants.clear()
    assert inst.run_bare_all_variants() == 5
    assert inst.seen_variants == [
        ChexVariantType.WITH_JIT,
        ChexVariantType.WITHOUT_JIT,
        ChexVariantType.WITH_DEVICE,
        ChexVariantType.WITHOUT_DEVICE,
        ChexVariantType.WITH_PMAP,
    ]
    assert SampleTest.run_bare_all_variants.variant_types == (
        ChexVariantType.WITH_JIT,
        ChexVariantType.WITHOUT_JIT,
        ChexVariantType.WITH_DEVICE,
        ChexVariantType.WITHOUT_DEVICE,
        ChexVariantType.WITH_PMAP,
    )

    inst.seen_variants.clear()
    assert inst.run_kw_all_variants() == 8
    assert len(inst.seen_variants) == 4
    assert inst.run_parameterized(5, b=20) == 26
    assert SampleTest.run_jit(inst) == 5

    with restrict_backends(forbidden=["cpu"]):
        with pytest.raises(AssertionError, match="forbidden by restrict_backends"):
            inst.run_with_device()

    with restrict_backends(allowed=["gpu"]):
        with pytest.raises(AssertionError, match="not in allowed backends"):
            inst.run_with_pmap()

    class PlainClass:
        """Plain non-TestCase class implementing VariantAttributeHolderProtocol."""

        def __init__(self) -> None:
            """Initialize plain class."""
            self.variant: VariantDecorator | None = None

        @variants(with_jit=True, without_jit=True)
        def run(self) -> int:
            """Execute method across two variants with existing self.variant attribute."""
            assert self.variant is not None
            return 10

    class PlainClassNoAttr:
        """Plain class without initial variant attribute."""

        @variants(with_jit=True, without_jit=True)
        def run(self) -> int:
            """Execute method without existing self.variant attribute."""
            return 11

    p1 = PlainClass()
    sentinel = VariantDecorator(ChexVariantType.WITHOUT_PMAP)
    p1.variant = sentinel
    assert p1.run() == 10
    assert PlainClass.run(p1) == 10
    assert p1.variant is sentinel

    p2 = PlainClassNoAttr()
    assert p2.run() == 11
    assert PlainClassNoAttr.run(p2) == 11

    @variants(with_jit=True, without_jit=True)
    def standalone() -> int:
        """Standalone zero-arg function wrapped with @variants."""
        return 12

    assert standalone() == 12
    assert standalone.executed_variants == [
        ChexVariantType.WITH_JIT,
        ChexVariantType.WITHOUT_JIT,
    ]
