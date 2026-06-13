import ml_switcheroo_compiler.ops as np
import ml_switcheroo_compiler.core.dtype as dt

import zero_chex as chex


def test_trees_disable_asserts():
    chex.disable_asserts()

    def fail(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
        except Exception as e:
            print(f"Failed {f.__name__}: {e}")

    x = np.ones((2, 3))
    t1 = {"a": x}
    t2 = {"a": np.ones((4, 5))}
    t3 = {"b": x}
    w = np.ones(2)

    # 12
    fail(chex.assert_tree_is_on_device, t1, "tpu")

    # 37-42
    fail(chex.assert_tree_is_sharded, {"a": "not_an_array"})

    # 76-81
    fail(chex.assert_trees_all_close, t1, t3)
    fail(chex.assert_trees_all_close, t1, {"a": np.add(x, np.ones((2, 3)))})

    # 88-96
    fail(chex.assert_trees_all_close_ulp, t1, t3)
    fail(chex.assert_trees_all_close_ulp, t1, {"a": np.add(x, np.ones((2, 3)))})

    # 111-114
    fail(chex.assert_trees_all_equal, t1, t3)
    fail(chex.assert_trees_all_equal, t1, {"a": np.add(x, np.ones((2, 3)))})

    # 146
    fail(chex.assert_tree_all_finite, {"a": float("inf")})

    # 180, 184
    fail(chex.assert_tree_has_only_ndarrays, {"a": "string"})

    # 214
    fail(chex.assert_tree_is_on_host, {"a": "string"})

    # 253, 257
    fail(chex.assert_tree_no_nones, {"a": None})

    # 277-282
    fail(chex.assert_tree_shape_prefix, t1, (5,))

    # 291-299
    fail(chex.assert_tree_shape_suffix, t1, (5,))

    # 304, 312
    fail(chex.assert_trees_all_equal_structs, t1, t3)
    fail(chex.assert_trees_all_equal_structs, t1, [1])

    # 338-343
    fail(
        chex.assert_trees_all_equal_comparator,
        lambda a, b: False,
        lambda a, b: "",
        t1,
        {"a": x},
    )

    # 367-372
    fail(
        chex.assert_trees_all_equal_dtypes,
        t1,
        {"a": np.ones((2, 3), dtype=dt.DType.Int32)},
    )
    fail(chex.assert_trees_all_equal_dtypes, t1, t3)

    # 423-428
    fail(chex.assert_trees_all_equal_shapes, t1, t2)
    fail(chex.assert_trees_all_equal_shapes, t1, t3)

    # 459-464
    fail(chex.assert_trees_all_equal_sizes, t1, {"a": w})
    fail(chex.assert_trees_all_equal_sizes, t1, t3)

    # 500-505
    fail(chex.assert_trees_all_equal_shapes_and_dtypes, t1, t2)

    # 540-545

    chex.enable_asserts()
