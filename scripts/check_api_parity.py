"""Verify API parity of zero_chex against chex_api_snapshot.json."""

from __future__ import annotations

import inspect
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import zero_chex

# Internal Chex helper utilities intentionally re-exported at top-level for test suite compatibility
_ALLOWED_INTERNAL_EXPORTS: frozenset[str] = frozenset({"get_err_regex"})


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
    """Run API parity validation against chex_api_snapshot.json."""
    with open("chex_api_snapshot.json", encoding="utf-8") as file_handle:
        chex_api: dict[str, dict[str, str]] = json.load(file_handle)

    zero_chex_all = set(zero_chex.__all__) - _ALLOWED_INTERNAL_EXPORTS
    chex_all = set(chex_api.keys())

    missing_in_zero = chex_all - zero_chex_all
    extra_in_zero = zero_chex_all - chex_all

    errors: list[str] = []
    if missing_in_zero:
        errors.append(f"Missing in zero_chex: {sorted(missing_in_zero)}")
    if extra_in_zero:
        errors.append(f"Extra in zero_chex: {sorted(extra_in_zero)}")

    for name in sorted(chex_all.intersection(zero_chex_all)):
        chex_info = chex_api[name]
        if name not in zero_chex.__all__:
            continue
        obj = zero_chex.__dict__.get(name)
        if chex_info.get("kind") == "function" and callable(obj):
            try:
                sig_zero = inspect.signature(obj)
                zero_params = list(sig_zero.parameters.keys())
                snap_params = extract_snapshot_param_names(chex_info.get("signature", "()"))
                if snap_params and zero_params[: len(snap_params)] != snap_params:
                    errors.append(f"{name} parameter mismatch: snapshot={snap_params}, zero={zero_params}")
            except (ValueError, TypeError):
                pass

    if errors:
        print("API Parity Check Failed:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    else:
        print("API Parity Check Passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
