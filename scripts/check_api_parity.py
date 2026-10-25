"""Verify API parity of zero_chex against ml-ecosystem-snapshots v0.0.5 and chex_api_snapshot.json."""

from __future__ import annotations

import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml-ecosystem-snapshots", "src"))
)

from ml_ecosystem_snapshots import extract_snapshot
from ml_ecosystem_snapshots.frameworks.chex import CANONICAL_CHEX_OPS

import zero_chex

REQUIRED_SUBMODULES: tuple[str, ...] = (
    "assertions",
    "dataclasses",
    "devices",
    "mocks",
    "scalars",
    "trees",
    "types",
    "utils",
    "variants",
    "_src",
    "_src.asserts",
    "_src.asserts.basic",
    "_src.asserts.scalar",
    "_src.asserts.tensors",
    "_src.asserts.trees",
    "_src.asserts_chexify",
    "_src.asserts_internal",
    "_src.classes",
    "_src.dataclass",
    "_src.dimensions",
    "_src.fake",
    "_src.misc",
    "_src.pytypes",
    "_src.restrict_backends",
    "_src.tree_util",
    "_src.types",
    "_src.variants",
    "_src.warnings",
)

REQUIRED_CLASS_FIELDS: dict[str, tuple[str, ...]] = {
    "ChexVariantType": ("WITH_JIT", "WITHOUT_JIT", "WITH_DEVICE", "WITHOUT_DEVICE", "WITH_PMAP"),
    "ChexifyChecks": ("user", "nan", "index", "div", "float", "automatic", "all"),
    "Device": ("platform", "id"),
    "Dimensions": ("__getitem__", "__setitem__", "__delitem__"),
    "PyTreeDef": ("node_type", "keys", "children", "custom_cls", "num_leaves"),
    "TestCase": ("variant",),
}


def extract_snapshot_param_names(signature_str: str) -> list[str]:
    """Extract parameter names from a snapshot signature string.

    Args:
        signature_str: The raw signature string from chex_api_snapshot.json.

    Returns:
        Ordered list of parameter names.
    """
    inner = signature_str.strip()
    if inner.startswith("("):
        inner = inner[1:]
    if ") ->" in inner:
        inner = inner.rsplit(") ->", 1)[0]
    elif inner.endswith(")"):
        inner = inner[:-1]
    if not inner.strip():
        return []

    params: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in inner:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                params.append(_clean_param_name(token))
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        params.append(_clean_param_name(tail))
    return [p for p in params if p and p != "/" and p != "*"]


def _clean_param_name(token: str) -> str:
    """Clean a single parameter declaration token into its bare identifier.

    Args:
        token: A single parameter segment from a signature string.

    Returns:
        The bare parameter name without stars, type annotations, or default values.
    """
    base = token.split(":", 1)[0].split("=", 1)[0].strip()
    return re.sub(r"^\*+", "", base)


def main() -> None:
    """Run API parity validation against ml-ecosystem-snapshots v0.0.5 and chex_api_snapshot.json."""
    with open("chex_api_snapshot.json", encoding="utf-8") as file_handle:
        chex_api: dict[str, dict[str, str]] = json.load(file_handle)

    zero_chex_members = vars(zero_chex)
    zero_chex_all = set(zero_chex.__all__)
    chex_all = set(chex_api.keys())

    errors: list[str] = []

    # 1. Verify module hierarchy exists and has module docstrings
    for submod_rel in REQUIRED_SUBMODULES:
        full_mod_name = f"zero_chex.{submod_rel}"
        mod = sys.modules.get(full_mod_name)
        if mod is None:
            errors.append(f"Missing required submodule: {full_mod_name}")
        elif not inspect.getdoc(mod):
            errors.append(f"Submodule {full_mod_name} is missing a module docstring.")

    # 2. Verify minimum API symbols from ml-ecosystem-snapshots v0.0.5 (CANONICAL_CHEX_OPS + extract_snapshot)
    canonical_names = {spec.name for spec in CANONICAL_CHEX_OPS}
    snap_envelope = extract_snapshot("chex")
    snapshot_items = snap_envelope["categories"].get("array", [])
    snapshot_names = {item["api_path"].split(".", 1)[1] for item in snapshot_items if "." in item["api_path"]}

    required_minimum_symbols = chex_all | canonical_names | snapshot_names
    missing_in_zero = required_minimum_symbols - zero_chex_all
    if missing_in_zero:
        errors.append(f"Missing minimum required symbols in zero_chex.__all__: {sorted(missing_in_zero)}")

    # 3. Verify classes and their required fields/methods
    for cls_name, field_names in REQUIRED_CLASS_FIELDS.items():
        cls_obj = zero_chex_members.get(cls_name)
        if cls_obj is None:
            errors.append(f"Missing required class: {cls_name}")
            continue
        cls_vars = dir(cls_obj)
        for field_name in field_names:
            if field_name not in cls_vars:
                errors.append(f"Class {cls_name} is missing required field/attribute '{field_name}'")

    # 4. Verify function signatures and docstrings against chex_api_snapshot.json
    for name in sorted(chex_all.intersection(zero_chex_all)):
        chex_info = chex_api[name]
        obj = zero_chex_members.get(name)
        if obj is None:
            errors.append(f"Symbol {name} declared in __all__ but missing from zero_chex module")
            continue
        if chex_info.get("kind") == "function" and callable(obj):
            if not inspect.getdoc(obj):
                errors.append(f"Function {name} is missing a docstring")
            try:
                sig_zero = inspect.signature(obj)
                zero_params = list(sig_zero.parameters.keys())
                snap_params = extract_snapshot_param_names(chex_info.get("signature", "()"))
                if snap_params and (
                    not all(p in zero_params for p in snap_params)
                    or zero_params[: len(snap_params) - (1 if snap_params[-1] == "check_kwargs" else 0)]
                    != snap_params[: len(snap_params) - (1 if snap_params[-1] == "check_kwargs" else 0)]
                ):
                    errors.append(f"{name} parameter mismatch: snapshot={snap_params}, zero={zero_params}")
            except (ValueError, TypeError) as exc:
                errors.append(f"{name} signature inspection failed: {exc}")

    if errors:
        print("API Parity Check Failed:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    else:
        print(
            f"API Parity Check Passed! ({len(required_minimum_symbols)} minimum symbols verified across "
            f"{len(REQUIRED_SUBMODULES)} submodules)"
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
