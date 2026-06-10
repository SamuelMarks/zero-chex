import pytest
import zero_chex as chex


def test_dimensions():
    dims = chex.Dimensions()
    dims["x"] = (2,)
    assert dims.x == 2

    with pytest.raises(TypeError):
        dims._validate_dim(123)
    with pytest.raises(KeyError):
        dims._validate_dim("123")

    with pytest.raises(TypeError):
        dims[123] = (2,)
    with pytest.raises(TypeError):
        dims["y"] = 2

    with pytest.raises(ValueError):
        dims["ab"] = (1, 2, 3)

    with pytest.raises(TypeError):
        _ = dims[123]
    with pytest.raises(ValueError):
        _ = dims["(x"]
    with pytest.raises(ValueError):
        _ = dims["()"]
    with pytest.raises(ValueError):
        _ = dims["((x))"]

    dims["a"] = (2,)
    dims["b"] = (3,)
    dims["c"] = (4,)
    assert dims["(a)b"] == (2, 3)

    with pytest.raises(ValueError):
        _ = dims["(a)b)"]

    with pytest.raises(TypeError):
        del dims[123]
    with pytest.raises(KeyError):
        del dims["z"]


def test_testcase():
    from zero_chex._src.classes import TestCase

    tc = TestCase()
    with pytest.raises(RuntimeError):
        tc.variant()


def test_dimensions_missing():
    dims = chex.Dimensions()
    dims["x"] = (2,)

    # getattr AttributeError
    with pytest.raises(AttributeError):
        _ = dims.y

    # repr
    assert repr(dims) == "Dimensions(x=2)"
