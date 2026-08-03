"""Bounded row-wise mean and standard-deviation reductions."""

from collections.abc import Iterator

import numpy as np

_ROW_BATCH_ELEMENTS = 262_144


def row_slices(n_rows: int, n_columns: int) -> Iterator[tuple[int, int]]:
    """Yield row slices whose matrices stay near the shared element budget."""
    batch_rows = max(1, _ROW_BATCH_ELEMENTS // max(1, n_columns))
    for start in range(0, n_rows, batch_rows):
        yield start, min(start + batch_rows, n_rows)


def row_mean(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate row means without a complete missing-value workspace."""
    means = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        means[start:stop] = _row_mean_block(x[start:stop], ignore_nan=ignore_nan)
    return means


def row_std(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate population row standard deviations in bounded batches."""
    _, deviations = row_mean_std(x, ignore_nan=ignore_nan)
    return deviations


def row_mean_std(x: np.ndarray, *, ignore_nan: bool) -> tuple[np.ndarray, np.ndarray]:
    """Calculate row means and population standard deviations together."""
    means = np.empty(len(x))
    deviations = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        block_means, block_deviations = _row_mean_std_block(
            x[start:stop],
            ignore_nan=ignore_nan,
        )
        means[start:stop] = block_means
        deviations[start:stop] = block_deviations
    return means, deviations


def _row_mean_block(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Reduce one bounded block to its row means."""
    if not ignore_nan:
        result: np.ndarray = np.mean(x, axis=1)
        return result

    valid = ~np.isnan(x)
    counts = np.sum(valid, axis=1, dtype=np.intp)
    means: np.ndarray = np.divide(
        np.sum(x, axis=1, dtype=float, where=valid),
        counts,
        out=np.full(len(x), np.nan),
        where=counts > 0,
    )
    return means


def _row_mean_std_block(
    x: np.ndarray,
    *,
    ignore_nan: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce one bounded block to row means and population deviations."""
    if not ignore_nan:
        means: np.ndarray = np.mean(x, axis=1)
        deviations: np.ndarray = np.std(x, axis=1)
        return means, deviations

    valid = ~np.isnan(x)
    counts = np.sum(valid, axis=1, dtype=np.intp)
    means = np.divide(
        np.sum(x, axis=1, dtype=float, where=valid),
        counts,
        out=np.full(len(x), np.nan),
        where=counts > 0,
    )
    centered = np.zeros(x.shape, dtype=float)
    with np.errstate(invalid="ignore"):
        np.subtract(x, means[:, np.newaxis], out=centered, where=valid)
    squared_deviations = np.einsum("ij,ij->i", centered, centered)
    deviations = np.divide(
        squared_deviations,
        counts,
        out=np.full(len(x), np.nan),
        where=counts > 0,
    )
    np.sqrt(deviations, out=deviations)
    return means, deviations
