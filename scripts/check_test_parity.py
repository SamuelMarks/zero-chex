import ast
import os
import json
import sys


def get_called_names(node):
    called = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                called.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                called.add(child.func.attr)
        elif isinstance(child, ast.FunctionDef) or isinstance(child, ast.ClassDef):
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


def main():
    with open("chex_api_snapshot.json", "r") as f:
        api = json.load(f)

    funcs = {k for k, v in api.items() if v["kind"] == "function"}

    # functions that we don't strictly require testing if they are weird aliases or so.
    # but let's see what fails first.

    tests_dir = "tests"
    covered = set()

    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    try:
                        tree = ast.parse(f.read(), filename=path)
                        for node in ast.walk(tree):
                            if isinstance(
                                node, ast.FunctionDef
                            ) and node.name.startswith("test_"):
                                called = get_called_names(node)
                                covered.update(called.intersection(funcs))
                            elif isinstance(
                                node, ast.ClassDef
                            ) and node.name.startswith("Test"):
                                # some might just call inside class? Tests are usually functions.
                                pass
                            elif isinstance(node, ast.Module):
                                # actually, maybe we should just look at all calls in the file
                                # wait, the prompt says "a test inside tests/ that calls it"
                                # Let's just collect all calls inside test files.
                                covered.update(
                                    get_called_names(node).intersection(funcs)
                                )
                    except Exception as e:
                        print(f"Failed to parse {path}: {e}")

    missing = funcs - covered

    if missing:
        print("Missing tests for:")
        for m in sorted(missing):
            print(" -", m)
        sys.exit(1)
    else:
        print("Test Parity Check Passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
