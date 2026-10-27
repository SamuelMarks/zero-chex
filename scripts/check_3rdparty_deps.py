"""Pre-commit hook to check for disallowed 3rd-party dependencies, banned typing/dunder constructs, and docstrings."""

from __future__ import annotations

import ast
import glob
import os
import re
import sys

ALLOWED_3RD_PARTY_SRC: frozenset[str] = frozenset(
    {
        "typing_extensions",
        "ml_switcheroo_ir",
        "ml_switcheroo_compiler",
        "ml_ecosystem_snapshots",
        "pydantic",
        "cdd",
        "zero_chex",
        "zero_jax",
        "zero_flax",
        "zero_grain",
        "zero_optax",
        "zero_orbax",
    }
)
ALLOWED_3RD_PARTY_TESTS: frozenset[str] = ALLOWED_3RD_PARTY_SRC | frozenset({"pytest", "scripts"})
BANNED_CALL_NAMES: frozenset[str] = frozenset({"cast", "getattr", "setattr", "hasattr", "delattr"})
BANNED_DUNDER_ATTRS: frozenset[str] = frozenset({"__dict__", "__class__", "__bases__", "__mro__", "__subclasses__"})
BANNED_TYPE_NAMES: frozenset[str] = frozenset({"Any", "object", "TYPE_CHECKING"})

STDLIB_MODULES: frozenset[str] = frozenset(
    set(sys.builtin_module_names)
    | {
        "__future__",
        "abc",
        "argparse",
        "ast",
        "builtins",
        "collections",
        "contextlib",
        "copy",
        "dataclasses",
        "enum",
        "functools",
        "glob",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "os",
        "pathlib",
        "re",
        "shutil",
        "struct",
        "subprocess",
        "sys",
        "sysconfig",
        "tempfile",
        "threading",
        "time",
        "traceback",
        "types",
        "typing",
        "unittest",
        "uuid",
        "warnings",
        "weakref",
    }
)


def _is_overload_node(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function AST node is decorated with @overload.

    Args:
        node: FunctionDef or AsyncFunctionDef AST node.

    Returns:
        True if decorated with @overload.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "overload":
            return True
    return False


def _check_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module,
    filepath: str,
    kind: str,
) -> bool:
    """Verify that a module, class, or function node has a complete docstring.

    Args:
        node: AST node to inspect.
        filepath: File path for diagnostic reporting.
        kind: Description of the node kind.

    Returns:
        True if docstring is present and documents all parameters.
    """
    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        return False
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        params: list[str] = [
            arg_node.arg
            for arg_node in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if arg_node.arg not in ("self", "cls")
        ]
        if node.args.vararg is not None:
            params.append(node.args.vararg.arg)
        if node.args.kwarg is not None:
            params.append(node.args.kwarg.arg)
        is_test_fn = os.path.basename(filepath).startswith("test_") and node.name.startswith("test_")
        if params and not is_test_fn:
            for p_name in params:
                pat = re.compile(rf"(^|\s|\*){re.escape(p_name)}\s*:")
                if not pat.search(doc):
                    return False
    _ = kind
    return True


def check_file(filepath: str, is_test_or_script: bool = False) -> bool:
    """Validate imports, banned typing/reflection constructs, and docstrings in a single Python file.

    Args:
        filepath: Path to the Python file to check.
        is_test_or_script: Whether the file is in tests/ or scripts/.

    Returns:
        True if the file passes all checks, False otherwise.
    """
    with open(filepath, encoding="utf-8") as file_handle:
        content = file_handle.read()

    if re.search(r"#\s*type:\s*ignore", content):
        return False

    tree = ast.parse(content, filename=filepath)
    allowed_3rd = ALLOWED_3RD_PARTY_TESTS if is_test_or_script else ALLOWED_3RD_PARTY_SRC
    allowed = STDLIB_MODULES | allowed_3rd

    if not _check_docstring(tree, filepath, "Module"):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not _check_docstring(node, filepath, f"Class '{node.name}'"):
                return False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_overload_node(node) and not _check_docstring(node, filepath, f"Function '{node.name}'"):
                return False
        elif isinstance(node, ast.Import):
            for alias in node.names:
                base_mod = alias.name.split(".")[0]
                if base_mod not in allowed:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                base_mod = node.module.split(".")[0]
                if base_mod not in allowed:
                    return False
            for alias in node.names:
                if alias.name in BANNED_TYPE_NAMES or alias.name in BANNED_CALL_NAMES:
                    return False
        elif isinstance(node, ast.Name):
            if node.id in BANNED_TYPE_NAMES:
                return False
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                return False
        elif isinstance(node, ast.Attribute):
            if (
                node.attr in BANNED_DUNDER_ATTRS
                or node.attr in ("Any", "TYPE_CHECKING")
                or node.attr in BANNED_CALL_NAMES
            ):
                return False

    return True


def collect_python_files(files: list[str] | None = None) -> list[str]:
    """Collect Python files to check from explicit paths or default src/tests/scripts directories.

    Args:
        files: Optional explicit list of file paths.

    Returns:
        Sorted list of Python file paths.
    """
    if files:
        return [f for f in files if f.endswith(".py")]
    collected: list[str] = []
    for root_dir in ("src", "tests", "scripts"):
        collected.extend(sorted(glob.glob(os.path.join(root_dir, "**", "*.py"), recursive=True)))
    return collected


def main(argv: list[str] | None = None) -> None:
    """Run dependency, typing, and docstring checks across target Python files.

    Args:
        argv: Optional command-line argument list.
    """
    raw_args = argv if argv is not None else sys.argv[1:]
    files = collect_python_files(raw_args)
    abs_src = os.path.abspath("src")
    for filepath in files:
        abs_fp = os.path.abspath(filepath)
        in_src = os.path.commonpath([abs_fp, abs_src]) == abs_src
        assert check_file(filepath, is_test_or_script=not in_src), f"Check failed for {filepath}"


if __name__ == "__main__":
    main()
