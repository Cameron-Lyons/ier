"""Bounded reductions for predefined item pairs."""

import warnings
from collections.abc import Sequence
from operator import index

import numpy as np

from ier._row_statistics import row_mean, row_slices


def validate_paired_item_indices(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    n_columns: int,
    *,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two ordered, equally sized item-index lists."""
    if len(left_indices) == 0 or len(right_indices) == 0:
        raise ValueError(f"{left_name} and {right_name} cannot be empty")
    if len(left_indices) != len(right_indices):
        raise ValueError(f"{left_name} and {right_name} must contain the same number of items")

    normalized: list[np.ndarray] = []
    for name, values in ((left_name, left_indices), (right_name, right_indices)):
        result = np.empty(len(values), dtype=np.intp)
        for position, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must contain integer column indices")
            try:
                item_index = index(value)
            except TypeError as error:
                raise ValueError(f"{name} must contain integer column indices") from error
            if item_index < 0 or item_index >= n_columns:
                raise ValueError(
                    f"item index {item_index} out of bounds for data with {n_columns} columns"
                )
            result[position] = item_index
        normalized.append(result)

    return normalized[0], normalized[1]


def paired_mean_absolute_difference(
    x: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    right_reflection: float | None,
    ignore_nan: bool,
) -> np.ndarray:
    """Reduce absolute differences for aligned column pairs in row batches."""
    if len(left_indices) != len(right_indices):
        raise ValueError("paired index arrays must contain the same number of items")
    n_pairs = len(left_indices)
    if n_pairs == 0:
        raise ValueError("paired index arrays cannot be empty")
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
