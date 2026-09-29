"""Shared summary statistics utilities for careless detection functions."""

from typing import Any

import numpy as np


def calculate_summary_stats(
    values: np.ndarray,
    suffix: str = "",
) -> dict[str, Any]:
    """
    Calculate common summary statistics for an array of values.

    Parameters:
    - values: Array of values (may contain NaN)
    - suffix: Optional suffix for dictionary keys (e.g., "_score" -> "mean_score")

    Returns:
    - Dictionary with mean, std, min, max, median statistics. Empty or entirely
      missing inputs have unavailable (NaN) statistics.
    """
    available = ~np.isnan(values)
    observed = values if np.all(available) else values[available]
    if observed.size == 0:
        return {f"{name}{suffix}": float("nan") for name in ("mean", "std", "min", "max", "median")}
    return {
        f"mean{suffix}": float(np.mean(observed)),
        f"std{suffix}": float(np.std(observed)),
        f"min{suffix}": float(np.min(observed)),
        f"max{suffix}": float(np.max(observed)),
        f"median{suffix}": float(np.median(observed)),
    }
