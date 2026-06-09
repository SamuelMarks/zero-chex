# Zero-Chex Semantic Implementation Plan

## What Does "Implemented" Mean Currently?
Thus far, **"Implemented" only meant Structural Compliance (API Parity).** 
By passing the `ml-framework-snapshots` check, we guaranteed that `zero-chex` exposes the exact same modules, classes, functions, arguments, default values, and type hints as the real `chex` library. This ensures that any downstream library importing `chex` (via our `zero_chex` alias) will not crash due to `AttributeError` or `TypeError` during import or standard invocation. 

**However, we have not achieved Semantic or Mathematical Parity.** 
Currently, the internal logic of these functions uses standard `numpy`, `unittest`, or mock objects to pass structural coverage requirements. 

## The Semantic Goal
To achieve **Semantic Compliance**, `zero-chex` must correctly integrate with `zero-jax` (and the broader `ml-switcheroo` ecosystem). It must correctly handle JAX tracers, abstract shapes, JIT compilation, parallel mapping (PMAP), device placement, and PyTree structures, ensuring that assertions and utilities behave exactly as they do in the real JAX/XLA ecosystem, ultimately emitting correct Intermediate Representation (IR) graphs.

---

## Phase 1: Zero-JAX Core Integration & Types
*Objective: Ensure foundational types strictly map to `zero-jax` internals, enabling proper tracer and PyTree resolution.*

- [x] **Type Mapping**
  - [x] Map `ArrayBatched` semantically to `zero_jax.Array` with batched tracer semantics.
  - [x] Map `ArrayDevice` to `zero_jax` physical device array structures.
  - [x] Map `ArraySharded` to `zero_jax.sharding` primitives.
  - [x] Map `Device` strictly to `zero_jax.Device`.
  - [x] Map `PRNGKey` to `zero_jax.random.PRNGKeyArray`.
  - [x] Map `PyTreeDef` to `zero_jax.tree_util.PyTreeDef`.
- [x] **Tree Utilities (`zero_chex._src.tree_util`)**
  - [x] Replace internal naive `tree_map` and `tree_flatten` with direct bindings to `zero_jax.tree_util`.
  - [x] Ensure `tree_flatten_with_path` matches JAX path resolution semantics.

## Phase 2: PyTree & Device Assertions
*Objective: Assertions must operate on `zero-jax` PyTrees and correctly inspect `zero-jax` device topologies.*

- [x] **Tree Assertions (`asserts/trees.py`)**
  - [x] `assert_tree_all_finite`: Implement using `zero_jax.numpy.isfinite` and `zero_jax.tree_util.tree_all`, ensuring compatibility inside `zero_jax.jit`.
  - [x] `assert_tree_has_only_ndarrays`: Verify leaves against `zero_jax.Array`.
  - [x] `assert_tree_is_on_device`: Hook into `zero_jax.Array.devices()` and validate against `zero_jax` platform specifications.
  - [x] `assert_tree_is_on_host`: Validate against `zero_jax` CPU device memory backing.
  - [x] `assert_tree_is_sharded`: Parse and validate `zero_jax.sharding.Sharding` attributes on arrays.
  - [x] `assert_trees_all_close`: Implement using `zero_jax.numpy.allclose`.
  - [x] `assert_trees_all_equal`: Implement using `zero_jax.numpy.array_equal`.
- [x] **Device & Basic Assertions (`asserts/basic.py`)**
  - [x] `_num_devices_available`: Integrate with `zero_jax.local_devices` / `zero_jax.devices`.
  - [x] `assert_devices_available`: Test against `zero-jax` mock device topologies.
  - [x] `assert_gpu_available` & `assert_tpu_available`: Query `zero_jax` backend configurations.

## Phase 3: Tensor, Shape, and Scalar Assertions
*Objective: Assertions must support `zero-jax` tracers. Static assertions must resolve at trace time, while runtime assertions must emit the correct `ml-switcheroo-ir` nodes.*

- [x] **Tensor Assertions (`asserts/tensors.py`)**
  - [x] `assert_axis_dimension*`: Handle `zero_jax.core.Tracer` shapes (including dynamic shapes if supported by `zero-jax`).
  - [x] `assert_equal_shape`: Validate shapes across evaluated arrays and JAX tracers.
  - [x] `assert_equal_rank`: Ensure rank resolution works transparently on `zero_jax.Array`.
  - [x] `assert_type`: Align strictly with `zero_jax.numpy.dtype` and `zero_jax._src.typing.SupportsDType`.
- [x] **Scalar Assertions (`asserts/scalar.py`)**
  - [x] Ensure `assert_scalar*` properly handles JAX 0-D arrays (which JAX frequently treats as scalars).
- [x] **Basic Logic (`asserts/basic.py`)**
  - [x] `assert_is_broadcastable`: Implement using `zero_jax.numpy.broadcast_shapes` for identical broadcasting semantics.

## Phase 4: Context Managers & Decorators
*Objective: System manipulation must perfectly mimic JAX's transformation boundaries.*

- [x] **Patching & Fake Contexts (`misc.py`)**
  - [x] `fake_jit`: Semantically disable JIT compilation using `zero_jax.disable_jit` context.
  - [x] `fake_pmap`: Semantically patch `zero_jax.pmap` to dynamically map to `zero_jax.vmap`, handling axis mappings correctly.
  - [x] `fake_pmap_and_jit`: Test the combination of disabling JIT and swapping PMAP.
- [x] **JAX Ecosystem Specifics**
  - [x] `dataclass`: Ensure `chex.dataclass` correctly wraps classes and calls `zero_jax.tree_util.register_dataclass`.
  - [x] `assert_max_traces`: Hook deeply into `zero_jax.core.trace_eval` or compiler internals to count exact JAX retriggers.
  - [x] `assert_numerical_grads`: Implement using `zero_jax.test_util.check_grads` or explicit `zero_jax.grad` analysis.
  - [x] `restrict_backends`: Tie into `zero_jax` config or `XLA_FLAGS` to actively block backend execution.
  - [x] `set_n_cpu_devices`: Configure `zero_jax` initialization logic correctly.

## Phase 5: Checkify & IR Compiler Compliance
*Objective: Ensure `chexify` aligns with the `ml-switcheroo` IR and `zero_jax.experimental.checkify`.*

- [x] **Checkify Assertions (`misc.py`)**
  - [x] Implement `chexify` to correctly delegate to `zero_jax.experimental.checkify.checkify`.
  - [x] Ensure that assertions injected inside `@chexify` functions emit the proper `LogicalNode` validation ops into the `ml-switcheroo-ir`.
  - [x] Implement asynchronous error evaluation (`async_check=True`) mapping to `zero_jax.device_get`.

## Phase 6: Switcheroo Cross-Framework Validation
*Objective: Prove that `zero-chex` operates identically to `chex` within the `ml-switcheroo` ecosystem.*

- [x] **Compiler/IR Validation**
  - [x] Run `ml-switcheroo-compiler` over a graph containing `zero-chex` assertions and ensure the resulting IR matches the expected ONNX/Switcheroo dialect structure without failing compliance.
  - [x] Create a comprehensive end-to-end test suite matching `chex` test cases against `zero-chex` + `zero-jax`.
