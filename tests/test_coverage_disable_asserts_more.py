import ml_switcheroo.ops as np
import ml_switcheroo.core.dtype as dt

import zero_chex as chex


def test_everything_disabled_more():
    chex.disable_asserts()

    def fail(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            print(f"Failed {f.__name__}: {e}")

    x = np.ones((2, 3))
    w = np.ones(2)

    # 10-11
    fail(chex.assert_shape, 1, ())

    # tensors.py
    # 161: assert_rank expected is list of lists
    fail(chex.assert_rank, [x], [[1]])

    # 180-184: x is list, expected is single int, one tensor is wrong rank
    fail(chex.assert_rank, [w, x], 1)

    # 201: assert_equal_shape expected is list, diff shape
    fail(chex.assert_equal_shape, [w, x])

    # 208-211: dims provided
    fail(chex.assert_equal_shape, [w, x], dims=(0,))

    # 241: assert_equal_shape_prefix, diff prefix
    fail(chex.assert_equal_shape_prefix, [np.ones((4, 3)), np.ones((5, 3))], 1)

    # 285: assert_equal_shape_suffix, dims tuple
    fail(chex.assert_equal_shape_suffix, [np.ones((2, 4)), np.ones((2, 5))], 1)

    # 293-297: assert_equal_size diff size
    fail(chex.assert_equal_size, [w, np.ones(3)])

    # 307: assert_shape list of arrays
    fail(chex.assert_shape, [w], (3,))

    # 318-322: assert_shape list of shapes
    fail(chex.assert_shape, [w, x], [(2,)])

    # 332: Ellipsis
    fail(chex.assert_shape, x, (Ellipsis, 4))

    # 346: tuple vs set
    fail(chex.assert_shape, x, (2, {4}))

    # 378: assert_size list of arrays
    fail(chex.assert_size, [w], 6)

    # 396-399: assert_type diff type
    fail(chex.assert_type, x, dt.DType.Int32)
    fail(chex.assert_type, [x], dt.DType.Int32)
    fail(chex.assert_type, [x, np.ones((2, 3), dtype=dt.DType.Int32)], dt.DType.Float64)

    chex.enable_asserts()
