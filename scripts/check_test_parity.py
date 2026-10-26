"""Verify that all zero_chex public symbols and snapshot APIs are imported and exercised in tests/."""

from __future__ import annotations

import ast
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from ml_ecosystem_snapshots import extract_snapshot
from ml_ecosystem_snapshots.frameworks.chex import CANONICAL_CHEX_OPS

import zero_chex


def get_tested_zero_chex_symbols(tree: ast.Module) -> set[str]:
    """Return zero_chex symbols that are both imported from zero_chex and used inside test_* functions.

    Args:
        tree: Parsed AST module of a test file.

    Returns:
        Set of canonical zero_chex symbol names verified in this test module.
    """
    alias_to_symbol: dict[str, str] = {}
    module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "zero_chex" or node.module.startswith("zero_chex."))
        ):
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                alias_to_symbol[local_name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zero_chex":
                    local_mod = alias.asname if alias.asname else alias.name
                    module_aliases.add(local_mod)

    tested: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in alias_to_symbol:
                    tested.add(alias_to_symbol[child.id])
                elif (
                    isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id in module_aliases
                ):
                    tested.add(child.attr)
    return tested


def main() -> None:
    """Check that every public symbol in zero_chex is imported from zero_chex and tested in tests/."""
    with open("chex_api_snapshot.json", encoding="utf-8") as file_handle:
        api: dict[str, dict[str, str]] = json.load(file_handle)

    snap_envelope = extract_snapshot("chex")
    snapshot_items = snap_envelope["categories"].get("array", [])
    snapshot_names = {item["api_path"].split(".", 1)[1] for item in snapshot_items if "." in item["api_path"]}
    canonical_names = {spec.name for spec in CANONICAL_CHEX_OPS}
    exported_names = set(zero_chex.__all__)

    all_symbols = set(api.keys()) | snapshot_names | canonical_names | exported_names

    tests_dir = "tests"
    covered: set[str] = set()

    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py") and file != "test_chex.py":
                path = os.path.join(root, file)
                with open(path, encoding="utf-8") as file_handle:
                    tree = ast.parse(file_handle.read(), filename=path)
                    covered.update(get_tested_zero_chex_symbols(tree).intersection(all_symbols))

    missing = all_symbols - covered

    if missing:
        print("Missing tests (not imported from zero_chex and used in test_* functions) for:")
        for item in sorted(missing):
            print(" -", item)
        sys.exit(1)
    else:
        print(f"Test Parity Check Passed! ({len(all_symbols)} symbols imported and verified in test_* functions)")
        sys.exit(0)


if __name__ == "__main__":
    main()
