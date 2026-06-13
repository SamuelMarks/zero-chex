# Zero-Chex Local Execution Plan

This local plan tracks the tasks required to refactor `zero-chex` to integrate with the newly refactored `ml-switcheroo-compiler` (which is now a pure backend) and `zero-jax`.

## Architectural Mandates
1. **Frontend Shim:** `zero-chex` acts as a compiler-frontend shim. It must be 100% API compatible with `chex`.
2. **Zero Dependencies:** `zero-chex` must **never** depend on `numpy`, `jax`, `flax`, or `chex` outside of tests.
3. **Backend Separation:** `ml-switcheroo-compiler` is now a pure backend. `zero-chex` must generate IR or use `zero-jax` (the new frontend counterpart) to construct assertions rather than relying on legacy `zero_jax_compiler` frontend ops.

## Refactoring Action Items

### 1. Eliminate Legacy Imports
-[x] Replace `from zero_jax_compiler.tree_util import tree_flatten` with the equivalent from `zero_jax` (e.g., `zero_jax.tree_util.tree_flatten`).
-[x] Replace `from zero_jax_compiler.core.tensor import Tensor` with the equivalent `Array`/`Tensor` class from `zero_jax` or `ml-switcheroo-compiler`.
-[x] Refactor numerical and logical operations in tree asserts (`assert_trees_all_close`, `assert_trees_all_close_ulp`, `assert_trees_all_equal`, `assert_tree_all_finite`) to use `zero_jax.numpy` instead of `zero_jax_compiler.ops` (e.g., `abs`, `add`, `multiply`, `subtract`, `less_equal`, `equal`, `all`, `isfinite`).

### 2. IR Generation and Backend Integration
-[x] Refactor scalar and axis dimension assertions (`assert_axis_dimension`, `assert_axis_dimension_gt`, etc.) to lower to `ml-switcheroo-compiler` assertion IR nodes (e.g., `AssertEq`, `Greater`, `Less`) during tracing instead of eagerly evaluating Python `int` vs `Tensor`.
-[x] Ensure all dynamic shape assertions correctly map to `ml-switcheroo-compiler` shape constraint IR nodes.
-[x] Refactor `assert_devices_available`, `assert_gpu_available`, `assert_tpu_available` to query the backend hardware topology from `ml-switcheroo-compiler`.
-[x] Refactor device placement assertions (`assert_tree_is_on_device`, `assert_tree_is_on_host`, `assert_tree_is_sharded`) to use the new `ml-switcheroo-compiler` sharding/device APIs.
-[x] Refactor `chexify` to lower runtime assertions to the backend IR and validate that it correctly flushes D2H assertion status queues.
-[x] Update `block_until_chexify_assertions_complete` and `set_n_cpu_devices` for backend synchronization.

### 3. Validation and Compliance
-[x] Run `pytest tests/` ensuring no `ModuleNotFoundError` is raised for legacy `zero_jax_compiler`.
-[x] Audit `src/zero_chex/` to guarantee zero usages of `numpy`, `jax`, `flax`, and `chex`.
-[x] Verify 100% semantic and syntactic parity of the test suite against the official `chex` behavior.
