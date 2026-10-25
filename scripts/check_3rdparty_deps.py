"""Pre-commit hook to check for disallowed 3rd-party dependencies and banned typing/dunder constructs."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import sysconfig


def get_stdlib_modules() -> set[str]:
    """Retrieve a set of all Python standard library module names.

    Returns:
        A set of standard library top-level module names.
    """
    stdlib_dir = sysconfig.get_path("stdlib")
    modules: set[str] = set(sys.builtin_module_names)
    if sys.version_info >= (3, 10):
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


ALLOWED_3RD_PARTY_SRC: frozenset[str] = frozenset(
    {
        "pydantic",
        "cdd",
        "ml_switcheroo_ir",
        "ml_switcheroo_compiler",
        "ml_ecosystem_snapshots",
        "typing_extensions",
        "zero_chex",
        "zero_jax",
        "zero_flax",
        "zero_grain",
        "zero_optax",
        "zero_orbax",
    }
)
ALLOWED_3RD_PARTY_TESTS: frozenset[str] = ALLOWED_3RD_PARTY_SRC | frozenset({"pytest"})
BANNED_CALL_NAMES: frozenset[str] = frozenset({"cast", "getattr", "setattr", "hasattr", "delattr"})
BANNED_DUNDER_ATTRS: frozenset[str] = frozenset({"__dict__", "__class__", "__bases__", "__mro__", "__subclasses__"})
BANNED_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object"})
STDLIB_MODULES: set[str] = get_stdlib_modules()


def _is_overload_node(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function AST node is decorated with @overload.

    Args:
        node: Function or async function AST node.

    Returns:
        True if decorated with @overload.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "overload":
            return True
    return False


def check_file(filepath: str, is_test_or_script: bool = False) -> bool:
    """Check a single Python file for disallowed 3rd-party imports, banned typing constructs, and docstrings.

    Args:
        filepath: Path to the Python file to inspect.
        is_test_or_script: Whether the file is in tests/ or scripts/ (allowing pytest).

    Returns:
        True if all imports, AST nodes, and docstrings are valid, False otherwise.
    """
    with open(filepath, encoding="utf-8") as file_handle:
        source_text = file_handle.read()
        try:
            tree = ast.parse(source_text, filename=filepath)
        except SyntaxError:
            print(f"Syntax error in {filepath}")
            return False

    ignore_tag = "# " + "type: ignore"
    for line_no, line in enumerate(source_text.splitlines(), start=1):
        if ignore_tag in line:
            print(f"{filepath}:{line_no} contains disallowed '{ignore_tag}' suppression.")
            return False

    allowed_3rd = ALLOWED_3RD_PARTY_TESTS if is_test_or_script else ALLOWED_3RD_PARTY_SRC
    disallowed_found: list[str] = []
    banned_constructs: list[str] = []

    if not ast.get_docstring(tree):
        banned_constructs.append("Line 1: missing module-level docstring")

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_overload_node(node)):
                doc = ast.get_docstring(node)
                if not doc:
                    banned_constructs.append(f"Line {node.lineno}: '{node.name}' missing docstring")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params: list[str] = []
                    for arg_node in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                        if arg_node.arg not in ("self", "cls"):
                            params.append(arg_node.arg)
                    if node.args.vararg is not None:
                        params.append(node.args.vararg.arg)
                    if node.args.kwarg is not None:
                        params.append(node.args.kwarg.arg)
                    for param_name in params:
                        pattern = rf"(?:^|\n)\s*\*{{0,2}}{re.escape(param_name)}\s*(?:\([^)]*\))?\s*:"
                        if not re.search(pattern, doc):
                            banned_constructs.append(
                                f"Line {node.lineno}: '{node.name}' missing docstring for param '{param_name}'"
                            )

        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module not in STDLIB_MODULES and base_module not in allowed_3rd:
                    disallowed_found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module not in STDLIB_MODULES and base_module not in allowed_3rd:
                    disallowed_found.append(node.module)
            for alias in node.names:
                if alias.name in BANNED_TYPE_NAMES or alias.name == "cast":
                    banned_constructs.append(f"Line {node.lineno}: import of banned symbol '{alias.name}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                banned_constructs.append(f"Line {node.lineno}: call to banned function '{node.func.id}()'")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_CALL_NAMES:
                banned_constructs.append(f"Line {node.lineno}: call to banned method '.{node.func.attr}()'")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_DUNDER_ATTRS:
            banned_constructs.append(f"Line {node.lineno}: access to banned dunder attribute '.{node.attr}'")
        elif isinstance(node, (ast.arg, ast.AnnAssign)):
            ann = node.annotation
            if ann is not None:
                for sub in ast.walk(ann):
                    if isinstance(sub, ast.Name) and sub.id in BANNED_TYPE_NAMES:
                        banned_constructs.append(f"Line {node.lineno}: banned type annotation '{sub.id}'")

    if disallowed_found:
        print(f"{filepath} contains disallowed 3rd-party imports:")
        for imp in sorted(set(disallowed_found)):
            print(f"  - {imp}")
        return False
    if banned_constructs:
        print(f"{filepath} contains banned constructs:")
        for msg in banned_constructs:
            print(f"  - {msg}")
        return False
    return True


def main() -> None:
    """Parse CLI arguments and run 3rd-party dependency and typing checks across files."""
    parser = argparse.ArgumentParser(description="Check for disallowed 3rd-party imports and constructs.")
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    args = parser.parse_args()

    success = True
    filenames: list[str] = list(args.filenames)

    if filenames:
        files_to_check = [f for f in filenames if f.endswith(".py") and os.path.exists(f)]
    else:
        files_to_check = []
        for scan_dir in ("src", "tests", "scripts"):
            if not os.path.exists(scan_dir):
                continue
            for root, _, files in os.walk(scan_dir):
                for file in files:
                    if file.endswith(".py"):
                        files_to_check.append(os.path.join(root, file))

    abs_src = os.path.abspath("src")
    for filepath in files_to_check:
        abs_fp = os.path.abspath(filepath)
        in_src = os.path.commonpath([abs_fp, abs_src]) == abs_src
        if not check_file(filepath, is_test_or_script=not in_src):
            success = False

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
