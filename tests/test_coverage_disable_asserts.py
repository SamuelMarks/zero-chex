import ml_switcheroo.ops as np
import ml_switcheroo.core.dtype as dt

import zero_chex as chex


def test_everything_disabled():
    chex.disable_asserts()

    # Call all assertion functions with clearly failing inputs
    def fail(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception:
            pass

    x = np.ones((2, 3))
    y = np.ones((4, 5))
    z = np.ones((2, 4))
    w = np.ones(2)

    # basic
    fail(chex.assert_equal, 1, 2)
    fail(chex.assert_exactly_one_is_none, 1, 2)
    fail(chex.assert_not_both_none, None, None)
    fail(chex.assert_devices_available, 999, "cpu")
    fail(chex.assert_gpu_available)
    fail(chex.assert_tpu_available)
    fail(chex.assert_is_broadcastable, (2,), (3,))
    fail(chex.assert_is_divisible, 10, 3)

    # scalar
    fail(chex.assert_scalar, [1])
    fail(chex.assert_scalar_in, 0, 1, 2)
    fail(chex.assert_scalar_negative, 1)
    fail(chex.assert_scalar_non_negative, -1)
    fail(chex.assert_scalar_positive, 0)

    # tensors
    # To hit all branches of assert_rank
    fail(chex.assert_rank, x, 3)
    fail(chex.assert_rank, [x], [(1,)])
    fail(chex.assert_rank, [x, x], [1, 2])
    fail(chex.assert_rank, [x, w], 1)

    # assert_equal_shape
    fail(chex.assert_equal_shape, [x, y])
    fail(chex.assert_equal_shape, [x, z])
    fail(chex.assert_equal_shape, [x, w], dims=[1])

    # assert_equal_shape_prefix
    fail(chex.assert_equal_shape_prefix, [x, y], 1)
    fail(chex.assert_equal_shape_prefix, [x, y], 2)

    # assert_equal_shape_suffix
    fail(chex.assert_equal_shape_suffix, [x, y], 1)
    fail(chex.assert_equal_shape_suffix, [x, y], 2)

    # assert_shape
    fail(chex.assert_shape, x, (2, 4))
    fail(chex.assert_shape, x, (3, 3))
    fail(chex.assert_shape, x, ())
    fail(chex.assert_shape, x, (Ellipsis, Ellipsis))
    fail(chex.assert_shape, x, ((3,), {4}))
    fail(chex.assert_shape, x, (2, {4}))

    # assert_size
    fail(chex.assert_size, x, 5)
    fail(chex.assert_size, x, 10)

    # assert_type
    fail(chex.assert_type, x, dt.DType.Int32)
    fail(chex.assert_type, [x, w], dt.DType.Int32)

    # trees
    t1 = {"a": 1}
    t2 = {"a": 2}
    t3 = {"b": 1}

    fail(chex.assert_tree_is_on_device, t1, "gpu")
    fail(chex.assert_tree_is_sharded, t1)
    fail(chex.assert_trees_all_close, t1, t2)
    fail(chex.assert_trees_all_close_ulp, t1, t2)
    fail(chex.assert_trees_all_equal, t1, t2)
    fail(chex.assert_tree_all_finite, {"a": float("inf")})
    fail(chex.assert_tree_has_only_ndarrays, t1)
    fail(chex.assert_tree_is_on_host, t1)
    fail(chex.assert_tree_no_nones, {"a": None})
    fail(chex.assert_tree_shape_prefix, x, (99,))
    fail(chex.assert_tree_shape_suffix, x, (99,))
    fail(chex.assert_trees_all_equal_structs, t1, t3)
    fail(chex.assert_trees_all_equal_structs, t1, [1])
    fail(
        chex.assert_trees_all_equal_comparator,
        lambda a, b: False,
        lambda a, b: "",
        t1,
        t2,
    )
    fail(chex.assert_trees_all_equal_dtypes, x, np.ones((2, 3), dtype=dt.DType.Int32))
    fail(chex.assert_trees_all_equal_shapes, x, y)
    fail(chex.assert_trees_all_equal_sizes, x, w)
    fail(chex.assert_trees_all_equal_shapes_and_dtypes, x, y)

    chex.enable_asserts()
