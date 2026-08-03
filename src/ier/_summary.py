"""Shared summary statistics utilities for careless detection functions."""

import numpy as np


def calculate_summary_stats(
    values: np.ndarray,
    suffix: str = "",
) -> dict[str, float]:
    """
    Calculate common summary statistics for an array of values.

    Parameters:
    - values: Array of values (may contain NaN)
    - suffix: Optional suffix for dictionary keys (e.g., "_score" -> "mean_score")

    Returns:
    - Dictionary with mean, std, min, max, median statistics
    """
    return {
        f"mean{suffix}": float(np.nanmean(values)),
        f"std{suffix}": float(np.nanstd(values)),
        f"min{suffix}": float(np.nanmin(values)),
        f"max{suffix}": float(np.nanmax(values)),
        f"median{suffix}": float(np.nanmedian(values)),
    }
