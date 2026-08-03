"""Bounded reductions for predefined item pairs."""

import warnings

import numpy as np

from ier._row_statistics import row_mean, row_slices


def paired_mean_absolute_difference(
    x: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    right_reflection: float | None,
    ignore_nan: bool,
) -> np.ndarray:
    """Reduce absolute differences for aligned column pairs in row batches."""
    n_pairs = min(len(left_indices), len(right_indices))
    left_indices = left_indices[:n_pairs]
    right_indices = right_indices[:n_pairs]
    scores = np.empty(len(x))

    for start, stop in row_slices(len(x), n_pairs):
        left = np.asarray(x[start:stop, left_indices], dtype=float)
        right = np.asarray(x[start:stop, right_indices], dtype=float)
        if right_reflection is not None:
            np.subtract(right_reflection, right, out=right)
        np.subtract(left, right, out=left)
        np.abs(left, out=left)
        scores[start:stop] = row_mean(left, ignore_nan=ignore_nan)

    return scores


def resolve_scale_bounds(
    x: np.ndarray,
    *,
    scale_min: float | None,
    scale_max: float | None,
) -> tuple[float, float] | None:
    """Resolve observed scale endpoints without a flattened data copy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resolved_min = float(np.nanmin(x)) if scale_min is None else float(scale_min)
        resolved_max = float(np.nanmax(x)) if scale_max is None else float(scale_max)

    if np.isnan(resolved_min) or np.isnan(resolved_max):
        return None
    if resolved_max < resolved_min:
        raise ValueError("scale_max must be greater than or equal to scale_min")
    return resolved_min, resolved_max
