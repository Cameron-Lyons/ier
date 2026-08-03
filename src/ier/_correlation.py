"""Allocation-conscious row-wise correlation helpers."""

import numpy as np


def row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return paired row correlations using pairwise-complete observations."""
    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must have the same number of rows")

    n_rows = left.shape[0]
    n_columns = min(left.shape[1], right.shape[1])
    if n_columns < 2:
        return np.full(n_rows, np.nan)

    left_values = left[:, :n_columns]
    right_values = right[:, :n_columns]
    has_missing = bool(np.isnan(left_values).any() or np.isnan(right_values).any())

    enough_values: np.ndarray | None = None
    with np.errstate(invalid="ignore", divide="ignore"):
        if has_missing:
            valid = ~(np.isnan(left_values) | np.isnan(right_values))
            valid_counts = valid.sum(axis=1)
            nonempty = valid_counts > 0
            left_mean = np.divide(
                np.sum(left_values, axis=1, where=valid),
                valid_counts,
                out=np.zeros(n_rows),
                where=nonempty,
            )
            right_mean = np.divide(
                np.sum(right_values, axis=1, where=valid),
                valid_counts,
                out=np.zeros(n_rows),
                where=nonempty,
            )
            left_centered = np.zeros(left_values.shape, dtype=float)
            right_centered = np.zeros(right_values.shape, dtype=float)
            np.subtract(
                left_values,
                left_mean[:, np.newaxis],
                out=left_centered,
                where=valid,
            )
            np.subtract(
                right_values,
                right_mean[:, np.newaxis],
                out=right_centered,
                where=valid,
            )
            enough_values = valid_counts >= 2
        else:
            left_centered = left_values - np.mean(left_values, axis=1, keepdims=True)
            right_centered = right_values - np.mean(right_values, axis=1, keepdims=True)

        covariance = np.einsum("ij,ij->i", left_centered, right_centered)
        denominator = np.einsum("ij,ij->i", left_centered, left_centered)
        denominator *= np.einsum("ij,ij->i", right_centered, right_centered)
        np.sqrt(denominator, out=denominator)
        correlations: np.ndarray = np.divide(
            covariance,
            denominator,
            out=np.zeros(n_rows),
            where=denominator > 0,
        )

    np.clip(correlations, -1.0, 1.0, out=correlations)
    if enough_values is not None:
        correlations[~enough_values] = np.nan

    return correlations
