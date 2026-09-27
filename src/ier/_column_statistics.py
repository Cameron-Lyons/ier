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
