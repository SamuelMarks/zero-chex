# Zero Framework API Shell

> **Note:** This repository is an API-compatible shell. All underlying math, autodiff, and graph execution has been migrated to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler) backend. This repository purely implements frontend routing and syntactic parity for the target framework.

# zero-chex

[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/SamuelMarks/zero-chex/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelMarks/zero-chex/actions)
[![Test Coverage](https://img.shields.io/badge/test_coverage-99.8%25-green.svg)](#)
[![Doc Coverage](https://img.shields.io/badge/doc_coverage-99.2%25-green.svg)](#)
[![API Compliance](https://img.shields.io/badge/api_compliance-100%25-brightgreen.svg)](#)

## Features & Compliance

`zero-chex` currently achieves **100% API coverage** (75/75 target APIs implemented) of the [chex](https://github.com/google-deepmind/chex) public interface.

- **Test Suite Parity**: The official test suite from `google-deepmind/chex` has been ported directly into this repository. **100% of the relevant official test suite (389 tests) passes perfectly** against `zero-chex`, achieving exact 1-to-1 API parity, including identical exception messages, structure validation, and edge cases.
- **Backend**: As an API shell, it delegates all structural validation and runtime execution to the [ml-switcheroo-compiler](https://github.com/SamuelMarks/ml-switcheroo-compiler).
- **Zero Dependencies**: Beyond standard Python libraries and NumPy, it brings no external baggage, replacing the heavy [chex](https://github.com/google-deepmind/chex)/JAX/XLA stack with a lightweight alternative.

## Why `zero-chex` Exists

This repository is a foundational component of the **Abstract ML Machine Ecosystem**, designed to solve the **$N \times M$ translation problem** in Machine Learning. 

Currently, the ML landscape is heavily fragmented. If you write a model in [JAX](https://github.com/google/jax), [PyTorch](https://pytorch.org/), [Keras](https://keras.io/), or [MLX](https://github.com/ml-explore/mlx) (the $N$ frontends), deploying that model efficiently across [WASM](https://webassembly.org/), [WebGPU](https://www.w3.org/TR/webgpu/), [TensorRT](https://developer.nvidia.com/tensorrt), or custom edge hardware (the $M$ backends) usually requires building and maintaining bespoke, complex translation pipelines for every single combination. 

### The Zero-Dependency Approach

`zero-chex` exists to address this by providing a **strictly zero external dependency** implementation of the [chex](https://github.com/google-deepmind/chex) API surface. It relies solely on the Python Standard Library and [`numpy`](https://numpy.org/) (for eager evaluations). 

Instead of wrapping heavy C++ binaries or relying on [XLA](https://openxla.org/), `zero-chex` mimics the public [chex](https://github.com/google-deepmind/chex) API and acts as a pure Python frontend. 

When you execute code using `zero-chex`, it delegates to [`zero-jax`](https://github.com/SamuelMarks/zero-jax) and [`ml-switcheroo-compiler`](https://github.com/SamuelMarks/ml-switcheroo-compiler) which dynamically traces the operations using proxy tensors. The resulting IR can then be seamlessly consumed by various backends, enabling a robust **source-to-source** and **source-to-browser** compilation pipeline.

### Part of a Larger Ecosystem

`zero-chex` is not a standalone numerical library, but rather Tier 4 of the ML Switcheroo architecture:
1. **Tier 1 ([`ml-switcheroo-ir`](https://github.com/SamuelMarks/ml-switcheroo-ir)):** Defines the canonical logical graph dialect ([ONNX](https://onnx.ai/) spec compliance).
2. **Tier 2 ([`ml-switcheroo-compiler`](https://github.com/SamuelMarks/ml-switcheroo-compiler)):** The computational heart, featuring AOT tracing, ProxyTensors, reverse-mode automatic differentiation, and optimizations like Dead Code Elimination (DCE).
3. **Tier 3 ([`zero-jax`](https://github.com/SamuelMarks/zero-jax)):** Provides the functional foundation and [JAX](https://github.com/google/jax) API parity. Pytree flattening is used to safely route state into the compiler tape.
4. **Tier 4 (Frontends & Add-ons):** Repositories like [`zero-flax`](https://github.com/SamuelMarks/zero-flax), [`zero-optax`](https://github.com/SamuelMarks/zero-optax), and `zero-chex` build on top of [`zero-jax`](https://github.com/SamuelMarks/zero-jax) to provide Neural Network layers, optimizers, and typing without any heavy external dependencies.
5. **Tier 5 ([`zero-zoo`](https://github.com/SamuelMarks/zero-zoo)):** Headless CI pipelines that train models deterministically to assert float-for-float equivalence ("Golden Seed" testing) across all simulated frameworks.

By maintaining structural API parity with the real [chex](https://github.com/google-deepmind/chex) framework (verified via [`ml-framework-snapshots`](https://github.com/SamuelMarks/ml-framework-snapshots)), `zero-chex` allows users to drop it in as a lightweight substitute in environments where installing the massive official [chex](https://github.com/google-deepmind/chex)/[JAX](https://github.com/google/jax)/[XLA](https://openxla.org/) stack is unfeasible—such as highly constrained serverless functions, or directly inside a web browser natively via [Pyodide](https://pyodide.org/) and [PyScript](https://pyscript.net/).

---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.