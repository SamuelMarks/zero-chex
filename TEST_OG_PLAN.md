# Official Chex Test Suite Porting Plan

## Phase 1: Discovery & Test Setup
- [ ] Research the official `google-deepmind/chex` repository on GitHub to locate their test files and directory structure.
- [ ] Identify all test dependencies required by the upstream test suite (e.g., `jax`, `jaxlib`, `absl-py`, `flax`, `numpy`, `pytest`, `dm-tree`).
- [ ] Create or update `requirements-test.txt` to include these identified test dependencies.
- [ ] Install the test dependencies locally to prepare the environment for running the upstream tests.

## Phase 2: Porting the Test Suite
- [ ] Create a dedicated directory in this repository for the upstream tests (e.g., `tests/official/`) to keep them isolated from custom tests.
- [ ] Fetch the official test suite scripts (e.g., `asserts_test.py`, `dataclass_test.py`, `variants_test.py`, `type_test.py`, etc.) using a web fetch tool.
- [ ] Write the downloaded test scripts into the `tests/official/` directory.
- [ ] Refactor imports within the downloaded tests:
  - [ ] Replace `import chex` with `import zero_chex as chex`.
  - [ ] Replace `from chex import ...` with `from zero_chex import ...`.
  - [ ] Adjust any internal import paths (e.g., `chex._src...` to `zero_chex._src...`) if accessed directly.

## Phase 3: Execution and 1-to-1 Parity Validation
- [ ] Run the official test suite via `pytest tests/official/`.
- [ ] Iteratively fix errors and API inconsistencies by addressing test failures module by module:
  - [ ] Fix `zero_chex.asserts` (e.g., shape, rank, type, dtype assertions).
  - [ ] Fix `zero_chex.classes` (dataclasses, mapping extensions).
  - [ ] Fix `zero_chex.types` (ArrayTree, Scalar, Numeric type aliases and arrays).
  - [ ] Fix `zero_chex.tree_util` (tree mappings, reductions).
  - [ ] Fix `zero_chex.misc` (fake representations, clear_trace, etc.).
- [ ] Ensure numerical evaluations and tensor checks match expected outputs exactly (1-to-1) or fall within `allclose` tolerance.
- [ ] For mock/zero-dependency specific behavior in `zero-chex` (if any checks are stubbed out), add explicit patches or bypass logic to the tests without compromising the test integrity of the parts that must match.

## Phase 4: Finalization and CI Integration
- [ ] Ensure 100% of the relevant official test suite passes.
- [ ] Add `requirements-test.txt` installation step to `.github/workflows/ci.yml`.
- [ ] Ensure the CI workflow runs `pytest tests/official/` on every push and PR.
- [ ] Update `COMPLIANCE.md` and/or `README.md` to indicate 1-to-1 parity with the upstream `chex` test suite.