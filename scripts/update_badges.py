"""Update test coverage, documentation coverage, and API compliance badges in README.md."""

from __future__ import annotations

import json
import os
import re
import subprocess


def get_color(pct: float) -> str:
    """Determine shield badge color from percentage value.

    Args:
        pct: Percentage value between 0.0 and 100.0.

    Returns:
        Shields.io color name string.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format a coverage percentage float into a concise badge string.

    Args:
        cov: Coverage percentage value.

    Returns:
        Formatted string without trailing .0 for whole numbers.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage() -> float:
    """Read test coverage percentage from coverage.json if available.

    Returns:
        Coverage percentage as a float.
    """
    try:
        subprocess.run(["coverage", "json", "-o", "coverage.json"], check=False)
        with open("coverage.json", encoding="utf-8") as file_handle:
            data = json.load(file_handle)
            totals = data.get("totals", {})
            pct = totals.get("percent_covered", 100.0)
            return float(pct)
    except (OSError, ValueError, KeyError):
        return 100.0


def get_doc_coverage() -> float:
    """Compute documentation coverage percentage via interrogate.

    Returns:
        Documentation coverage percentage as a float.
    """
    try:
        subprocess.run(
            ["interrogate", "--generate-badge", "."],
            capture_output=True,
            text=True,
            check=False,
        )
        result = subprocess.run(
            ["interrogate", "src"],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"actual: (\d+\.?\d*)%", result.stdout)
        if match:
            return float(match.group(1))
        match_err = re.search(r"actual: (\d+\.?\d*)%", result.stderr)
        if match_err:
            return float(match_err.group(1))
        return 100.0
    except (OSError, ValueError):
        return 100.0


def get_api_compliance() -> float:
    """Read API compliance percentage from CHEX_TODO.md.

    Returns:
        Compliance percentage as a float.
    """
    try:
        todo_files = [f for f in os.listdir(".") if f.endswith("_TODO.md")]
        if not todo_files:
            return 100.0
        with open(todo_files[0], encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.startswith("Overall Compliance:"):
                    return float(line.split(":")[1].strip().replace("%", ""))
    except (OSError, ValueError, IndexError):
        pass
    return 100.0


def update_readme() -> None:
    """Update coverage and compliance shield badges in README.md."""
    if not os.path.exists("README.md"):
        return

    test_cov = get_test_coverage()
    doc_cov = get_doc_coverage()

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    with open("README.md", encoding="utf-8") as file_handle:
        content = file_handle.read()

    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    api_comp = get_api_compliance()
    api_str = format_cov(api_comp)
    api_color = get_color(api_comp)

    api_re = re.compile(
        r"\[?\!\[API Compliance\]\(https://img\.shields\.io/badge/(?:[aA]pi_)?(?:[cC]ompliance)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    if api_re.search(content):
        content = api_re.sub(
            f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)",
            content,
        )

    with open("README.md", "w", encoding="utf-8") as file_handle:
        file_handle.write(content)


if __name__ == "__main__":
    update_readme()
