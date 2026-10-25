"""Verify that all symbols in ml-ecosystem-snapshots v0.0.5 and chex_api_snapshot.json are exercised in tests/."""

from __future__ import annotations

import ast
import json
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml-ecosystem-snapshots", "src"))
)

from ml_ecosystem_snapshots import extract_snapshot
from ml_ecosystem_snapshots.frameworks.chex import CANONICAL_CHEX_OPS


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
    """Check that every symbol in ml-ecosystem-snapshots v0.0.5 and chex_api_snapshot.json is tested in tests/."""
    with open("chex_api_snapshot.json", encoding="utf-8") as file_handle:
        api: dict[str, dict[str, str]] = json.load(file_handle)

    snap_envelope = extract_snapshot("chex")
    snapshot_items = snap_envelope["categories"].get("array", [])
    snapshot_names = {item["api_path"].split(".", 1)[1] for item in snapshot_items if "." in item["api_path"]}
    canonical_names = {spec.name for spec in CANONICAL_CHEX_OPS}

    all_symbols = set(api.keys()) | snapshot_names | canonical_names

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
        print(f"Test Parity Check Passed! ({len(all_symbols)} symbols verified)")
        sys.exit(0)


if __name__ == "__main__":
    main()
