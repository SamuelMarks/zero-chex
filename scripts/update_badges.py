"""Update test coverage, documentation coverage, and API compliance badges in README.md."""

from __future__ import annotations

import os
import re


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


def get_api_compliance(todo_path: str = "CHEX_TODO.md") -> float:
    """Read API compliance percentage from CHEX_TODO.md.

    Args:
        todo_path: Path to CHEX_TODO.md.

    Returns:
        Compliance percentage as a float.
    """
    if os.path.exists(todo_path):
        with open(todo_path, encoding="utf-8") as file_handle:
            for line in file_handle:
                if line.startswith("Overall Compliance:"):
                    return float(line.split(":")[1].strip().replace("%", ""))
    return 100.0


def update_readme(
    readme_path: str = "README.md",
    test_cov: float = 100.0,
    doc_cov: float = 100.0,
    todo_path: str = "CHEX_TODO.md",
) -> None:
    """Update coverage and compliance shield badges in README.md.

    Args:
        readme_path: Path to README.md.
        test_cov: Test coverage percentage.
        doc_cov: Documentation coverage percentage.
        todo_path: Path to CHEX_TODO.md for API compliance percentage.
    """
    if not os.path.exists(readme_path):
        return

    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)

    with open(readme_path, encoding="utf-8") as file_handle:
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

    api_comp = get_api_compliance(todo_path)
    api_str = format_cov(api_comp)
    api_color = get_color(api_comp)

    api_re = re.compile(
        r"\[?\!\[API Compliance\]\(https://img\.shields\.io/badge/(?:[aA]pi_)?(?:[cC]ompliance)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = api_re.sub(
        f"[![API Compliance](https://img.shields.io/badge/api_compliance-{api_str}%25-{api_color}.svg)](#)",
        content,
    )

    with open(readme_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(content)


if __name__ == "__main__":
    update_readme()
