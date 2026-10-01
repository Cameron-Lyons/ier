"""Shared summary statistics utilities for careless detection functions."""

from typing import Any

import numpy as np

from ier._row_statistics import row_mean_std, row_median


def observed_summary_stats(values: np.ndarray, *, include_median: bool = False) -> dict[str, float]:
    """Summarize already-selected observations with stable shared reductions."""
    names = (
        ("mean", "std", "min", "max", "median") if include_median else ("mean", "std", "min", "max")
    )
    if values.size == 0:
        return dict.fromkeys(names, float("nan"))
    vector = values.reshape(1, -1)
    means, deviations = row_mean_std(vector, ignore_nan=False)
    stats = {
        "mean": float(means[0]),
        "std": float(deviations[0]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    if include_median:
        stats["median"] = float(row_median(vector, ignore_nan=False)[0])
    return stats


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
    stats = observed_summary_stats(observed, include_median=True)
    return {f"{name}{suffix}": value for name, value in stats.items()}
