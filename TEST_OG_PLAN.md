# Official Chex Test Suite Porting Plan

## Phase 1: Discovery & Test Setup
- [x] Research the official `google-deepmind/chex` repository on GitHub to locate their test files and directory structure.
- [x] Identify all test dependencies required by the upstream test suite (e.g., `jax`, `jaxlib`, `absl-py`, `flax`, `numpy`, `pytest`, `dm-tree`).
- [x] Create or update `requirements-test.txt` to include these identified test dependencies.
- [x] Install the test dependencies locally to prepare the environment for running the upstream tests.

## Phase 2: Porting the Test Suite
- [x] Create a dedicated directory in this repository for the upstream tests (e.g., `tests/official/`) to keep them isolated from custom tests.
- [x] Fetch the official test suite scripts (e.g., `asserts_test.py`, `dataclass_test.py`, `variants_test.py`, `type_test.py`, etc.) using a web fetch tool.
- [x] Write the downloaded test scripts into the `tests/official/` directory.
- [x] Refactor imports within the downloaded tests:
  - [x] Replace `import chex` with `import zero_chex as chex`.
  - [x] Replace `from chex import ...` with `from zero_chex import ...`.
  - [x] Adjust any internal import paths (e.g., `chex._src...` to `zero_chex._src...`) if accessed directly.

## Phase 3: Execution and 1-to-1 Parity Validation
- [x] Run the official test suite via `pytest tests/official/`.
- [x] Iteratively fix errors and API inconsistencies by addressing test failures module by module:
  - [x] Fix `zero_chex.asserts` (e.g., shape, rank, type, dtype assertions).
  - [x] Fix `zero_chex.classes` (dataclasses, mapping extensions).
  - [x] Fix `zero_chex.types` (ArrayTree, Scalar, Numeric type aliases and arrays).
  - [x] Fix `zero_chex.tree_util` (tree mappings, reductions).
  - [x] Fix `zero_chex.misc` (fake representations, clear_trace, etc.).
- [x] Ensure numerical evaluations and tensor checks match expected outputs exactly (1-to-1) or fall within `allclose` tolerance.
- [x] For mock/zero-dependency specific behavior in `zero-chex` (if any checks are stubbed out), add explicit patches or bypass logic to the tests without compromising the test integrity of the parts that must match.

## Phase 4: Finalization and CI Integration
- [x] Ensure 100% of the relevant official test suite passes.
- [x] Add `requirements-test.txt` installation step to `.github/workflows/ci.yml`.
- [x] Ensure the CI workflow runs `pytest tests/official/` on every push and PR.
- [x] Update `COMPLIANCE.md` and/or `README.md` to indicate 1-to-1 parity with the upstream `chex` test suite.