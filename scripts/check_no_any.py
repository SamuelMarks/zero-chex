"""Verify that zero Any, object, cast, getattr/setattr/hasattr/delattr, __dict__, or TYPE_CHECKING hacks exist."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.check_3rdparty_deps import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
