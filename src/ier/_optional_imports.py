"""Helpers for optional runtime dependencies."""

from __future__ import annotations

from typing import Any


def scipy_install_hint(feature: str) -> str:
    """Return a consistent install hint for SciPy-backed features."""
    return f"scipy is required for {feature}. Install with: pip install 'insufficient-effort[full]'"


def require_scipy(feature: str) -> None:
    """Raise RuntimeError if SciPy is not installed."""
    try:
        import scipy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(scipy_install_hint(feature)) from exc


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
