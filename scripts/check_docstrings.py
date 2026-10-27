"""Verify 100% Google-style docstring coverage across src/zero_chex, scripts, and tests."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.check_3rdparty_deps import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
