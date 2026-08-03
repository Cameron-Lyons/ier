"""Bounded row-wise mean, median, and standard-deviation reductions."""

from collections.abc import Iterator

import numpy as np

_ROW_BATCH_ELEMENTS = 262_144


def row_slices(n_rows: int, n_columns: int) -> Iterator[tuple[int, int]]:
    """Yield row slices whose matrices stay near the shared element budget."""
    batch_rows = max(1, _ROW_BATCH_ELEMENTS // max(1, n_columns))
    for start in range(0, n_rows, batch_rows):
        yield start, min(start + batch_rows, n_rows)


def compressed_row_groups(
    x: np.ndarray,
    *,
    min_columns: int,
    max_elements: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield row indices and NaN-free rows grouped by retained column count."""
    element_budget = _ROW_BATCH_ELEMENTS if max_elements is None else max_elements
    batch_rows = max(1, element_budget // max(1, x.shape[1]))
    valid_counts = np.empty(len(x), dtype=np.intp)

    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        block = x[start:stop]
        valid_counts[start:stop] = x.shape[1] - np.count_nonzero(np.isnan(block), axis=1)

    for count_value in np.unique(valid_counts[valid_counts >= min_columns]):
        count = int(count_value)
        row_indices = np.flatnonzero(valid_counts == count)
        for start in range(0, len(row_indices), batch_rows):
            rows = row_indices[start : start + batch_rows]
            selected = x[rows]
            compressed = selected[~np.isnan(selected)].reshape(len(rows), count)
            yield rows, compressed


def row_mean(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate row means without a complete missing-value workspace."""
    means = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        means[start:stop] = _row_mean_block(x[start:stop], ignore_nan=ignore_nan)
    return means


def row_median(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate row medians without a complete partition workspace."""
    medians = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        medians[start:stop] = _row_median_block(x[start:stop], ignore_nan=ignore_nan)
    return medians


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


def _row_median_block(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Reduce one bounded block to its row medians."""
    if not ignore_nan:
        result: np.ndarray = np.median(x, axis=1)
        return result

    available = np.any(~np.isnan(x), axis=1)
    medians = np.full(len(x), np.nan)
    if np.any(available):
        medians[available] = np.nanmedian(x[available], axis=1)
    return medians


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
