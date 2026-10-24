"""Check API compliance against CHEX_TODO.md and SEMANTIC_PLAN.md."""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import zero_chex


def _parse_chex_todo() -> list[tuple[str, bool]]:
    """Parse API compliance table rows from CHEX_TODO.md.

    Returns:
        List of (symbol_name, is_checked) pairs from CHEX_TODO.md.
    """
    if not os.path.exists("CHEX_TODO.md"):
        return []
    with open("CHEX_TODO.md", encoding="utf-8") as file_handle:
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


def _parse_semantic_plan() -> list[tuple[str, bool]]:
    """Parse backticked API symbol targets from SEMANTIC_PLAN.md.

    Returns:
        List of (symbol_name, is_checked) pairs from SEMANTIC_PLAN.md.
    """
    if not os.path.exists("SEMANTIC_PLAN.md"):
        return []
    with open("SEMANTIC_PLAN.md", encoding="utf-8") as file_handle:
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


def main() -> None:
    """Validate that all 75 API targets in CHEX_TODO.md and SEMANTIC_PLAN.md are exported by zero_chex."""
    todo_items = _parse_chex_todo()
    semantic_items = _parse_semantic_plan()

    if not todo_items and not semantic_items:
        print("COMPLIANCE FAILURE: No API targets found in CHEX_TODO.md or SEMANTIC_PLAN.md.")
        sys.exit(1)

    exported_set = set(zero_chex.__all__)
    missing: list[str] = []
    total = len(todo_items)
    implemented = 0

    for symbol_name, checked in todo_items:
        if checked and symbol_name in exported_set:
            implemented += 1
        else:
            missing.append(f"zero_chex.{symbol_name}")

    for symbol_name, checked in semantic_items:
        if not checked or symbol_name not in exported_set:
            entry = f"zero_chex.{symbol_name} (SEMANTIC_PLAN.md)"
            if entry not in missing:
                missing.append(entry)

    if missing:
        print("COMPLIANCE FAILURE: The following expected APIs are missing or unchecked:")
        for item in missing:
            print(f"  - {item}")
        pct = (implemented / total) * 100.0 if total > 0 else 0.0
        print(f"Overall Compliance: {pct:.1f}% ({implemented}/{total})")
        sys.exit(1)

    print(f"Compliance Check Passed! 100% compliant ({implemented}/{total} APIs).")
    sys.exit(0)


if __name__ == "__main__":
    main()
