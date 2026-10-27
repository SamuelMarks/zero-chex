"""Verify API parity of zero_chex against ml-ecosystem-snapshots v0.0.5 and chex_api_snapshot.json."""

from __future__ import annotations

import inspect
import json
import os
import re
import sys

sys.path.extend(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
    for rel_path in (
        "src",
        "../zero-jax/src",
        "../zero-flax/src",
        "../zero-grain/src",
        "../zero-optax/src",
        "../zero-orbax/src",
        "../ml-switcheroo-ir/src",
        "../ml-switcheroo-compiler/src",
    )
)

from ml_ecosystem_snapshots import extract_snapshot
from ml_ecosystem_snapshots.frameworks.chex import CANONICAL_CHEX_OPS

import zero_chex

REQUIRED_SUBMODULES: tuple[str, ...] = (
    "assertions",
    "dataclass",
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
            params.append(_clean_param_name("".join(current).strip()))
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


def check_api_parity(snapshot_path: str = "chex_api_snapshot.json") -> tuple[int, list[str]]:
    """Run API parity validation against ml-ecosystem-snapshots v0.0.5 and chex_api_snapshot.json.

    Args:
        snapshot_path: Path to chex_api_snapshot.json.

    Returns:
        Tuple of (verified_minimum_symbol_count, error_messages_list).
    """
    import importlib

    with open(snapshot_path, encoding="utf-8") as file_handle:
        chex_api: dict[str, dict[str, str]] = json.load(file_handle)

    zero_chex_members = vars(zero_chex)
    zero_chex_all = set(zero_chex.__all__)
    chex_all = set(chex_api.keys())

    for submod_rel in REQUIRED_SUBMODULES:
        full_mod_name = f"zero_chex.{submod_rel}"
        mod = importlib.import_module(full_mod_name)
        assert inspect.getdoc(mod), f"Submodule {full_mod_name} is missing a module docstring."

    canonical_names = {spec.name for spec in CANONICAL_CHEX_OPS}
    snap_envelope = extract_snapshot("chex")
    snapshot_items = snap_envelope["categories"].get("array", [])
    snapshot_names = {item["api_path"].split(".", 1)[1] for item in snapshot_items if "." in item["api_path"]}

    required_minimum_symbols = chex_all | canonical_names | snapshot_names
    missing_in_zero = required_minimum_symbols - zero_chex_all
    assert not missing_in_zero, f"Missing minimum required symbols in zero_chex.__all__: {sorted(missing_in_zero)}"

    for cls_name, field_names in REQUIRED_CLASS_FIELDS.items():
        cls_obj = zero_chex_members[cls_name]
        cls_vars = dir(cls_obj)
        for field_name in field_names:
            assert field_name in cls_vars, f"Class {cls_name} is missing required field/attribute '{field_name}'"

    for name in sorted(chex_all.intersection(zero_chex_all)):
        chex_info = chex_api[name]
        obj = zero_chex_members[name]
        if chex_info.get("kind") == "function" and callable(obj):
            assert inspect.getdoc(obj), f"Function {name} is missing a docstring"
            sig_zero = inspect.signature(obj)
            zero_params = list(sig_zero.parameters.keys())
            snap_params = extract_snapshot_param_names(chex_info.get("signature", "()"))
            prefix_len = len(snap_params) - (1 if snap_params and snap_params[-1] == "check_kwargs" else 0)
            assert not snap_params or zero_params[:prefix_len] == snap_params[:prefix_len], (
                f"{name} parameter mismatch: snapshot={snap_params}, zero={zero_params}"
            )

    return len(required_minimum_symbols), []


def main() -> None:
    """Execute API parity validation and exit with status 0 on success."""
    count, errors = check_api_parity()
    assert not errors, f"API Parity Check Failed: {errors}"
    print(f"API Parity Check Passed! ({count} minimum symbols verified across {len(REQUIRED_SUBMODULES)} submodules)")


if __name__ == "__main__":
    main()
