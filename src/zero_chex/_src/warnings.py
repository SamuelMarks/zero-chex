"""Internal warnings module re-exporting deprecation helpers from zero_chex.utils."""

from __future__ import annotations

from zero_chex.utils import (
    create_deprecated_function_alias,
    warn_deprecated_function,
    warn_keyword_args_only_in_future,
    warn_only_n_pos_args_in_future,
)

__all__ = [
    "create_deprecated_function_alias",
    "warn_deprecated_function",
    "warn_keyword_args_only_in_future",
    "warn_only_n_pos_args_in_future",
]
