"""Helpers for optional runtime dependencies."""

from __future__ import annotations

from typing import Any


def require_matplotlib_pyplot() -> Any:
    """Import matplotlib.pyplot or raise a clear optional-dependency error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'insufficient-effort[plot]'"
        ) from exc

    return plt
