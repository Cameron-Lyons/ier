"""Allocation-conscious row-wise correlation helpers."""

import numpy as np


def row_correlations(
    left: np.ndarray,
    right: np.ndarray,
    *,
    zero_variance: float = 0.0,
) -> np.ndarray:
    """Return paired row correlations using pairwise-complete observations."""
    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must have the same number of rows")

    n_rows = left.shape[0]
    n_columns = min(left.shape[1], right.shape[1])
    if n_columns < 2:
        return np.full(n_rows, np.nan)

    left_values = left[:, :n_columns]
    right_values = right[:, :n_columns]
    if n_columns == 2 and not np.isinf(left_values).any() and not np.isinf(right_values).any():
        return _two_point_row_correlations(
            left_values,
            right_values,
            zero_variance=zero_variance,
        )

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
            out=np.full(n_rows, zero_variance),
            where=denominator > 0,
        )

    np.clip(correlations, -1.0, 1.0, out=correlations)
    if enough_values is not None:
        correlations[~enough_values] = np.nan

    return correlations


def _two_point_row_correlations(
    left: np.ndarray,
    right: np.ndarray,
    *,
    zero_variance: float,
) -> np.ndarray:
    """Correlate two paired observations from their difference signs."""
    left_delta = np.empty(len(left), dtype=float)
    right_delta = np.empty(len(right), dtype=float)
    with np.errstate(over="ignore"):
        np.subtract(left[:, 0], left[:, 1], out=left_delta, casting="unsafe")
        np.subtract(right[:, 0], right[:, 1], out=right_delta, casting="unsafe")

    zero_variance_rows: np.ndarray | None = None
    if zero_variance != 0.0:
        zero_variance_rows = (left_delta == 0.0) | (right_delta == 0.0)

    np.sign(left_delta, out=left_delta)
    np.sign(right_delta, out=right_delta)
    np.multiply(left_delta, right_delta, out=left_delta)
    if zero_variance_rows is not None:
        left_delta[zero_variance_rows] = zero_variance
    return left_delta
