"""Allocation-conscious row-wise correlation helpers."""

import numpy as np

from ier._row_statistics import row_slices


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
    use_two_point = n_columns == 2 and not _contains_inf(left_values, right_values)
    has_missing = False if use_two_point else _contains_nan(left_values, right_values)
    correlations = np.empty(n_rows)

    for start, stop in row_slices(n_rows, n_columns):
        left_block = left_values[start:stop]
        right_block = right_values[start:stop]
        if use_two_point:
            correlations[start:stop] = _two_point_row_correlations(
                left_block,
                right_block,
                zero_variance=zero_variance,
            )
        else:
            correlations[start:stop] = _row_correlations_block(
                left_block,
                right_block,
                has_missing=has_missing,
                zero_variance=zero_variance,
            )

    return correlations


def _contains_inf(left: np.ndarray, right: np.ndarray) -> bool:
    """Check paired matrices for infinities without a full Boolean workspace."""
    for start, stop in row_slices(len(left), left.shape[1]):
        if np.isinf(left[start:stop]).any() or np.isinf(right[start:stop]).any():
            return True
    return False


def _contains_nan(left: np.ndarray, right: np.ndarray) -> bool:
    """Check paired matrices for missing values without a full Boolean workspace."""
    for start, stop in row_slices(len(left), left.shape[1]):
        if np.isnan(left[start:stop]).any() or np.isnan(right[start:stop]).any():
            return True
    return False


def _row_correlations_block(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    has_missing: bool,
    zero_variance: float,
) -> np.ndarray:
    """Correlate one bounded block using the globally selected numerical path."""
    n_rows = len(left_values)
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
