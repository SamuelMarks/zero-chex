"""Verify that all symbols in chex_api_snapshot.json are exercised in enabled-assertion tests/."""

from __future__ import annotations

import ast
import json
import os
import sys


def get_referenced_names(node: ast.AST) -> set[str]:
    """Collect all function, class, type alias, and decorator names referenced within an AST subtree.

    Args:
        node: The root AST node to walk.

    Returns:
        A set of referenced identifier names.
    """
    referenced: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            referenced.add(child.id)
        elif isinstance(child, ast.Attribute):
            referenced.add(child.attr)
    return referenced


def main() -> None:
    """Check that every symbol in chex_api_snapshot.json is exercised in tests/ (excluding test_chex.py)."""
    with open("chex_api_snapshot.json", encoding="utf-8") as file_handle:
        api: dict[str, dict[str, str]] = json.load(file_handle)

    all_symbols = set(api.keys())

    tests_dir = "tests"
    covered: set[str] = set()

    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py") and file != "test_chex.py":
                path = os.path.join(root, file)
                with open(path, encoding="utf-8") as file_handle:
                    try:
                        tree = ast.parse(file_handle.read(), filename=path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                                refs = get_referenced_names(node)
                                covered.update(refs.intersection(all_symbols))
                            elif isinstance(node, ast.Module):
                                covered.update(get_referenced_names(node).intersection(all_symbols))
                    except SyntaxError as exc:
                        print(f"Failed to parse {path}: {exc}")

    missing = all_symbols - covered

    if missing:
        print("Missing tests for:")
        for item in sorted(missing):
            print(" -", item)
        sys.exit(1)
    else:
        print("Test Parity Check Passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
