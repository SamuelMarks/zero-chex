"""Pre-commit hook to check for disallowed 3rd-party dependencies."""

from __future__ import annotations

import argparse
import ast
import os
import sys
import sysconfig


def get_stdlib_modules() -> set[str]:
    """Retrieve a set of all Python standard library module names.

    Returns:
        A set of standard library top-level module names.
    """
    stdlib_dir = sysconfig.get_path("stdlib")
    modules: set[str] = set(sys.builtin_module_names)
    if hasattr(sys, "stdlib_module_names"):
        modules.update(sys.stdlib_module_names)

    if stdlib_dir and os.path.exists(stdlib_dir):
        for entry in os.listdir(stdlib_dir):
            if entry.endswith(".py"):
                modules.add(entry[:-3])
            elif os.path.isdir(os.path.join(stdlib_dir, entry)) and entry != "site-packages" and entry != "__pycache__":
                modules.add(entry)

        lib_dynload = os.path.join(stdlib_dir, "lib-dynload")
        if os.path.exists(lib_dynload):
            for entry in os.listdir(lib_dynload):
                modules.add(entry.split(".")[0])

    return modules


ALLOWED_3RD_PARTY: frozenset[str] = frozenset(
    {
        "pydantic",
        "cdd",
        "ml_switcheroo_ir",
        "ml_ecosystem_snapshots",
        "typing_extensions",
        "zero_chex",
        "zero_jax",
    }
)
STDLIB_MODULES: set[str] = get_stdlib_modules()


def check_file(filepath: str) -> bool:
    """Check a single Python file for disallowed 3rd-party imports.

    Args:
        filepath: Path to the Python file to inspect.

    Returns:
        True if all imports are allowed, False otherwise.
    """
    with open(filepath, encoding="utf-8") as file_handle:
        source_text = file_handle.read()
        try:
            tree = ast.parse(source_text, filename=filepath)
        except SyntaxError:
            print(f"Syntax error in {filepath}")
            return False

    for line_no, line in enumerate(source_text.splitlines(), start=1):
        if "# type: ignore" in line:
            print(f"{filepath}:{line_no} contains disallowed '# type: ignore' suppression.")
            return False

    disallowed_found: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module not in STDLIB_MODULES and base_module not in ALLOWED_3RD_PARTY:
                    disallowed_found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module not in STDLIB_MODULES and base_module not in ALLOWED_3RD_PARTY:
                    disallowed_found.append(node.module)

    if disallowed_found:
        print(f"{filepath} contains disallowed 3rd-party imports:")
        for imp in sorted(set(disallowed_found)):
            print(f"  - {imp}")
        return False
    return True


def main() -> None:
    """Parse CLI arguments and run 3rd-party dependency checks across source files."""
    parser = argparse.ArgumentParser(description="Check for disallowed 3rd-party imports.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    args = parser.parse_args()

    success = True
    filenames: list[str] = list(args.filenames)

    if filenames:
        files_to_check = [
            f
            for f in filenames
            if f.endswith(".py")
            and os.path.commonpath([os.path.abspath(f), os.path.abspath("src")]) == os.path.abspath("src")
        ]
    else:
        files_to_check = []
        for root, _, files in os.walk("src"):
            for file in files:
                if file.endswith(".py"):
                    files_to_check.append(os.path.join(root, file))

    for filepath in files_to_check:
        if not check_file(filepath):
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
