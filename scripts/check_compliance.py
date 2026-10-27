"""Check API compliance against ml-ecosystem-snapshots v0.0.5, CHEX_TODO.md, SEMANTIC_PLAN.md, and TODO_PLAN.md."""

from __future__ import annotations

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


def _parse_chex_todo(path: str = "CHEX_TODO.md") -> list[tuple[str, bool]]:
    """Parse API compliance table rows from CHEX_TODO.md.

    Args:
        path: Path to CHEX_TODO.md.

    Returns:
        List of (symbol_name, is_checked) pairs from CHEX_TODO.md.
    """
    with open(path, encoding="utf-8") as file_handle:
        content = file_handle.read()

    items: list[tuple[str, bool]] = []
    row_pattern = re.compile(r"^\|\s*\[([xX ])\]\s*\|\s*chex\s*\|\s*chex\s*\|\s*([a-zA-Z0-9_]+)\s*\|")
    for line in content.splitlines():
        match = row_pattern.match(line.strip())
        if match:
            checked = match.group(1).lower() == "x"
            symbol_name = match.group(2)
            items.append((symbol_name, checked))
    return items


def _parse_semantic_plan(path: str = "SEMANTIC_PLAN.md") -> list[tuple[str, bool]]:
    """Parse backticked API symbol targets from SEMANTIC_PLAN.md.

    Args:
        path: Path to SEMANTIC_PLAN.md.

    Returns:
        List of (symbol_name, is_checked) pairs from SEMANTIC_PLAN.md.
    """
    with open(path, encoding="utf-8") as file_handle:
        content = file_handle.read()

    exported_names = set(zero_chex.__all__)
    found: list[tuple[str, bool]] = []
    bullet_pattern = re.compile(r"^\s*-\s*\[([xX ])\]\s*(.+)$")
    for line in content.splitlines():
        match = bullet_pattern.match(line)
        if not match:
            continue
        checked = match.group(1).lower() == "x"
        body = match.group(2)
        for token in re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", body):
            if token in exported_names:
                found.append((token, checked))
    return found


def check_compliance(
    todo_path: str = "CHEX_TODO.md",
    plan_path: str = "SEMANTIC_PLAN.md",
) -> tuple[int, int, list[str]]:
    """Validate that all API targets in ml-ecosystem-snapshots v0.0.5, CHEX_TODO.md, and SEMANTIC_PLAN.md exist.

    Args:
        todo_path: Path to CHEX_TODO.md.
        plan_path: Path to SEMANTIC_PLAN.md.

    Returns:
        Tuple of (implemented_count, total_count, missing_entries).
    """
    todo_items = _parse_chex_todo(todo_path)
    semantic_items = _parse_semantic_plan(plan_path)

    exported_set = set(zero_chex.__all__)
    zero_members = vars(zero_chex)
    missing: list[str] = []

    snap_envelope = extract_snapshot("chex")
    snapshot_items = snap_envelope["categories"].get("array", [])
    snapshot_names = {item["api_path"].split(".", 1)[1] for item in snapshot_items if "." in item["api_path"]}
    canonical_names = {spec.name for spec in CANONICAL_CHEX_OPS}

    minimum_snapshot_apis = sorted(snapshot_names | canonical_names)
    total = len(minimum_snapshot_apis)
    implemented = 0

    for symbol_name in minimum_snapshot_apis:
        assert symbol_name in exported_set and symbol_name in zero_members, (
            f"Missing zero_chex.{symbol_name} (ml-ecosystem-snapshots v0.0.5)"
        )
        implemented += 1

    for symbol_name, checked in todo_items:
        assert checked and symbol_name in exported_set, f"Missing or unchecked zero_chex.{symbol_name} (CHEX_TODO.md)"

    for symbol_name, checked in semantic_items:
        assert checked and symbol_name in exported_set, (
            f"Missing or unchecked zero_chex.{symbol_name} (SEMANTIC_PLAN.md)"
        )

    return implemented, total, missing


def main() -> None:
    """Execute API compliance check and exit with status 0 on success."""
    implemented, total, missing = check_compliance()
    assert not missing, f"COMPLIANCE FAILURE: {missing}"
    print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} Snapshot APIs).")


if __name__ == "__main__":
    main()
