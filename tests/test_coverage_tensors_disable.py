import ml_switcheroo.ops as np
import ml_switcheroo.core.dtype as dt

import zero_chex as chex


def test_tensors_disable_asserts():
    chex.disable_asserts()

    def fail(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            print(f"Failed {f.__name__}: {e}")

    x = np.ones((2, 3))

    # assert_rank
    fail(chex.assert_rank, x, np.ones(1))
    fail(chex.assert_rank, x, [[1]])
    fail(chex.assert_rank, [x], [1, 2])
    fail(chex.assert_rank, x, [1, 2])
    fail(chex.assert_rank, x, {3})
    fail(chex.assert_rank, x, 3)

    # assert_shape
    fail(chex.assert_shape, [np.ones((2,))], [(2,), (3,)])
    fail(chex.assert_shape, np.ones((2,)), ())
    fail(chex.assert_shape, np.ones((2,)), (Ellipsis, Ellipsis))
    fail(chex.assert_shape, np.ones((2,)), (2, Ellipsis, 3, 4))
    fail(chex.assert_shape, np.ones((2, 3)), ({3}, Ellipsis))
    fail(chex.assert_shape, np.ones((2, 3)), (3, Ellipsis))
    fail(chex.assert_shape, np.ones((2, 3)), (Ellipsis, {4}))
    fail(chex.assert_shape, np.ones((2, 3)), (Ellipsis, 4))
    fail(chex.assert_shape, np.ones((2, 3)), (2, 3, 4))
    fail(chex.assert_shape, np.ones((2, 3)), (2, {4}))
    fail(chex.assert_shape, np.ones((2, 3)), (2, 4))

    # assert_size
    fail(chex.assert_size, [np.ones((2,))], [1, 2])
    fail(chex.assert_size, np.ones((2,)), [1, 2])
    fail(chex.assert_size, np.ones((2,)), {3})
    fail(chex.assert_size, np.ones((2,)), 3)

    # assert_type
    fail(chex.assert_type, [np.ones((2,))], [dt.DType.Int32, dt.DType.Float32])
    fail(chex.assert_type, np.ones((2,)), [dt.DType.Int32, dt.DType.Float32])
    fail(
        chex.assert_type,
        np.ones((2,), dtype=dt.DType.Int32),
        (dt.DType.Float32, dt.DType.Float64),
    )
    fail(chex.assert_type, np.ones((2,), dtype=dt.DType.Int32), dt.DType.Float32)

    chex.enable_asserts()
