"""Verify that all functions in chex_api_snapshot.json are exercised in tests/."""

from __future__ import annotations

import ast
import json
import os
import sys


def get_called_names(node: ast.AST) -> set[str]:
    """Collect all function and decorator names referenced within an AST subtree.

    Args:
        node: The root AST node to walk.

    Returns:
        A set of called function and decorator identifier names.
    """
    called: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                called.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                called.add(child.func.attr)
        elif isinstance(child, (ast.FunctionDef, ast.ClassDef)):
            for dec in child.decorator_list:
                if isinstance(dec, ast.Name):
                    called.add(dec.id)
                elif isinstance(dec, ast.Attribute):
                    called.add(dec.attr)
                elif isinstance(dec, ast.Call):
                    if isinstance(dec.func, ast.Name):
                        called.add(dec.func.id)
                    elif isinstance(dec.func, ast.Attribute):
                        called.add(dec.func.attr)
    return called


def main() -> None:
    """Check that every function in chex_api_snapshot.json is called in tests/."""
    with open("chex_api_snapshot.json", encoding="utf-8") as file_handle:
        api: dict[str, dict[str, str]] = json.load(file_handle)

    funcs = {k for k, v in api.items() if v.get("kind") == "function"}

    tests_dir = "tests"
    covered: set[str] = set()

    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, encoding="utf-8") as file_handle:
                    try:
                        tree = ast.parse(file_handle.read(), filename=path)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                                called = get_called_names(node)
                                covered.update(called.intersection(funcs))
                            elif isinstance(node, ast.Module):
                                covered.update(get_called_names(node).intersection(funcs))
                    except SyntaxError as exc:
                        print(f"Failed to parse {path}: {exc}")

    missing = funcs - covered

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
