# API Compliance Declaration

This framework is a zero-dependency pure-Python implementation of the `google-deepmind/chex` API surface, built on `ml-switcheroo-ir` types and protocols.

- **100% Documentation Coverage**: Maintained across `src/`, `tests/`, and `scripts/` via `interrogate` and Ruff `D` rules.
- **100% Test Coverage**: Maintained (100% line and branch coverage) across all modules in `src/zero_chex/`.
- **Strong Typing**: Maintained (`mypy --strict`, explicit annotations, no `Any`, `object`, `cast`, `getattr`/`setattr`, or `# type: ignore`).
- **No 3rd Party Executable Dependencies**: Maintained (purely standard library and `ml-switcheroo-ir`).

## Test Suite Structure & Parity
All 75 public APIs in `chex_api_snapshot.json` and `CHEX_TODO.md` are verified by the unit and integration test suite under `tests/` (`test_basic.py`, `test_chex.py`, `test_classes.py`, `test_misc.py`, `test_scalar.py`, `test_tensors.py`, `test_trees.py`, `test_types.py`), achieving 1-to-1 signature and behavioral parity with upstream `chex`.
