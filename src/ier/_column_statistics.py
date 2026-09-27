"""Bounded column-wise statistical reductions."""

import numpy as np

from ier._row_statistics import row_slices


def column_mean(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate item means without a full missing-value mask or data copy."""
    if not ignore_nan:
        result: np.ndarray = np.mean(x, axis=0)
        return result

    sums = np.zeros(x.shape[1])
    counts = np.zeros(x.shape[1], dtype=np.intp)
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        valid = ~np.isnan(block)
        sums += np.sum(block, axis=0, dtype=float, where=valid)
        counts += np.sum(valid, axis=0, dtype=np.intp)

    means: np.ndarray = np.divide(
        sums,
        counts,
        out=np.full(x.shape[1], np.nan),
        where=counts > 0,
    )
    return means


def column_correlations(x: np.ndarray) -> np.ndarray:
    """Correlate items with bounded centering workspaces and NaN propagation.

    Missing observations invalidate their entire item, as in ``np.corrcoef``.
    Constant items and samples with fewer than two rows have undefined
    correlations. The item-by-item output is necessarily quadratic in width.
    """
    dtype = np.result_type(x.dtype, np.float64)
    if len(x) < 2:
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=dtype)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.mean(x, axis=0, dtype=dtype)
        available = np.flatnonzero(np.isfinite(means))
        if len(available) == 0:
            return np.full((x.shape[1], x.shape[1]), np.nan, dtype=dtype)
        all_available = len(available) == x.shape[1]
        means = means[available]
        correlations = np.zeros((len(available), len(available)), dtype=dtype)
        for start, stop in row_slices(len(x), x.shape[1]):
            if all_available:
                centered = np.array(x[start:stop], dtype=dtype, copy=True)
            else:
                centered = np.asarray(x[start:stop, available], dtype=dtype)
            centered -= means
            right = centered.conj() if np.iscomplexobj(centered) else centered
            correlations += centered.T @ right
            del centered, right
        correlations /= len(x) - 1
        deviations = np.sqrt(correlations.diagonal().real)
        correlations /= deviations[:, None]
        correlations /= deviations[None, :]

    np.clip(correlations.real, -1.0, 1.0, out=correlations.real)
    if np.iscomplexobj(correlations):
        np.clip(correlations.imag, -1.0, 1.0, out=correlations.imag)
    if not all_available:
        result = np.full((x.shape[1], x.shape[1]), np.nan, dtype=dtype)
        result[np.ix_(available, available)] = correlations
        return result
    return correlations
