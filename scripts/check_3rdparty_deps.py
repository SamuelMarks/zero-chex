"""Pre-commit hook to check for disallowed 3rd-party dependencies."""

import ast
import os
import sys
import distutils.sysconfig as ds
import argparse


def get_stdlib_modules():
    """Get a set of all standard library module names."""
    stdlib_dir = ds.get_python_lib(standard_lib=True)
    modules = set(sys.builtin_module_names)

    for f in os.listdir(stdlib_dir):
        if f.endswith(".py"):
            modules.add(f[:-3])
        elif (
            os.path.isdir(os.path.join(stdlib_dir, f))
            and f != "site-packages"
            and f != "__pycache__"
        ):
            modules.add(f)

    lib_dynload = os.path.join(stdlib_dir, "lib-dynload")
    if os.path.exists(lib_dynload):
        for f in os.listdir(lib_dynload):
            modules.add(f.split(".")[0])

    return modules


ALLOWED_3RD_PARTY = {
    "pydantic",
    "cdd",
    "ml_switcheroo_compiler",
    "ml_switcheroo_ir",
    "zero_jax",
    "zero_chex",
}
STDLIB_MODULES = get_stdlib_modules()


def check_file(filepath):
    """Check a single file for disallowed 3rd-party imports."""
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            print(f"Syntax error in {filepath}")
            return False

    disallowed_found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if (
                    base_module not in STDLIB_MODULES
                    and base_module not in ALLOWED_3RD_PARTY
                ):
                    disallowed_found.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                continue
            if node.module:
                base_module = node.module.split(".")[0]
                if (
                    base_module not in STDLIB_MODULES
                    and base_module not in ALLOWED_3RD_PARTY
                ):
                    disallowed_found.append(node.module)

    if disallowed_found:
        print(f"{filepath} contains disallowed 3rd-party imports:")
        for imp in set(disallowed_found):
            print(f"  - {imp}")
        return False
    return True


def main():
    """Main function to parse arguments and run checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", help="Filenames to check.")
    args = parser.parse_args()

    success = True

    if args.filenames:
        files_to_check = [
            f
            for f in args.filenames
            if f.endswith(".py")
            and os.path.commonpath([os.path.abspath(f), os.path.abspath("src")])
            == os.path.abspath("src")
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
