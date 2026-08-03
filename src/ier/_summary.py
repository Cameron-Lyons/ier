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
    missing = np.isnan(values)
    valid_values = values[~missing] if np.any(missing) else values

    if valid_values.size == 0:
        mean = std = minimum = maximum = median = float("nan")
    else:
        mean = float(np.mean(valid_values))
        std = float(np.std(valid_values))
        minimum = float(np.min(valid_values))
        maximum = float(np.max(valid_values))
        median = float(np.median(valid_values))

    return {
        f"mean{suffix}": mean,
        f"std{suffix}": std,
        f"min{suffix}": minimum,
        f"max{suffix}": maximum,
        f"median{suffix}": median,
    }
