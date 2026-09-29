"""Bounded row-wise sum, mean, median, and standard-deviation reductions."""

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


def row_sum(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate floating-point row totals without a complete missing-value copy."""
    sums = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        valid = ~np.isnan(block) if ignore_nan else True
        sums[start:stop] = np.sum(block, axis=1, dtype=float, where=valid)
    return sums


def row_median(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate row medians without a complete partition workspace."""
    medians = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        medians[start:stop] = _row_median_block(x[start:stop], ignore_nan=ignore_nan)
    return medians


def row_std(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate population row standard deviations in bounded batches."""
    deviations = np.empty(len(x))
    for start, stop in row_slices(len(x), x.shape[1]):
        _, deviations[start:stop] = _row_mean_std_block(x[start:stop], ignore_nan=ignore_nan)
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
    means, _, _ = _row_mean_counts_block(x, ignore_nan=ignore_nan)
    return means


def _row_median_block(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Reduce one bounded block to its row medians."""
    medians = np.full(len(x), np.nan)
    width = x.shape[1]
    if not width:
        return medians

    missing = np.isnan(x)
    if not ignore_nan or not np.any(missing):
        available = ~np.any(missing, axis=1)
        del missing
        if not np.any(available):
            return medians
        values = (
            np.array(x, copy=True, order="C")
            if np.all(available)
            else np.ascontiguousarray(x[available])
        )
        middle = width // 2
        if width < 128:
            values.sort(axis=1)
        else:
            # One partition avoids the slower multiple-position selection path.
            values.partition(middle, axis=1)
        if width % 2:
            medians[available] = values[:, middle]
        else:
            lower = values[:, middle - 1] if width < 128 else values[:, :middle].max(axis=1)
            medians[available] = _median_midpoints(lower, values[:, middle])
        return medians

    counts = width - np.count_nonzero(missing, axis=1)
    del missing
    if not np.any(counts):
        return medians
    # Sorting places NaNs after all observations, including positive infinity.
    # One owned buffer avoids masked-array sorting and its additional copies.
    values = np.array(x, copy=True, order="C")
    values.sort(axis=1)
    rows = np.arange(len(x))
    middle = counts // 2
    medians[:] = values[rows, middle]
    even = (counts > 0) & (counts % 2 == 0)
    medians[even] = _median_midpoints(values[rows[even], middle[even] - 1], medians[even])
    medians[counts == 0] = np.nan
    return medians


def _median_midpoints(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Average ordered median endpoints without overflowing or losing tiny values."""
    left = np.asarray(lower, dtype=float)
    right = np.asarray(upper, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result: np.ndarray = (left + right) * 0.5
        overflow = np.isinf(result) & np.isfinite(left) & np.isfinite(right)
        # Add before halving ordinary values to preserve subnormal midpoints.
        result[overflow] = left[overflow] * 0.5 + right[overflow] * 0.5
    if lower.dtype.kind in "iu" and lower.dtype.itemsize == 8:
        large = upper > 2**53
        if lower.dtype.kind == "i":
            large |= lower < -(2**53)
        if np.any(large):
            # Preserve integer cancellation and round only the final midpoint.
            result[large] = np.fromiter(
                ((int(a) + int(b)) / 2 for a, b in zip(lower[large], upper[large], strict=True)),
                dtype=float,
                count=np.count_nonzero(large),
            )
    return result


def _row_mean_std_block(
    x: np.ndarray,
    *,
    ignore_nan: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a bounded block of rows or grouped rows over its last axis."""
    means, deviations, _ = _row_mean_std_counts_block(x, ignore_nan=ignore_nan)
    return means, deviations


def _row_mean_counts_block(
    x: np.ndarray,
    *,
    ignore_nan: bool,
    check_integer_range: bool = True,
    integer_sum: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Accumulate means in double precision, retaining a mask only for missing data."""
    valid = ~np.isnan(x) if ignore_nan and x.dtype.kind not in "iub" else None
    if valid is not None and np.all(valid):
        valid = None
    counts = (
        valid.sum(axis=-1, dtype=np.intp)
        if valid is not None
        else np.full(x.shape[:-1], x.shape[-1], dtype=np.intp)
    )
    if check_integer_range:
        _, integer_sum = _integer_reduction_parameters(x)
    if x.dtype.kind in "iu" and x.shape[-1] and not integer_sum:
        return _integer_means(x), counts, None
    overflowed = False
    with np.errstate(over="raise", invalid="ignore"):
        try:
            means = np.asarray(
                np.sum(
                    x,
                    axis=-1,
                    dtype=(np.uint64 if x.dtype.kind == "u" else np.int64)
                    if integer_sum
                    else float,
                    where=valid if valid is not None else True,
                ),
                dtype=float,
            )
        except FloatingPointError:
            # Ordinary NaNs and input infinities do not require an overflow scan.
            overflowed = True
            with np.errstate(over="ignore"):
                means = np.sum(x, axis=-1, dtype=float, where=valid if valid is not None else True)
    nonempty = counts > 0
    with np.errstate(under="ignore"):
        np.divide(means, counts, out=means, where=nonempty)
    means[~nonempty] = np.nan
    if overflowed:
        _repair_scaled_moments(
            x, means, None, ~np.isfinite(means) & nonempty, ignore_nan=ignore_nan
        )
    return means, counts, valid


def _integer_reduction_parameters(x: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Find exceptional integer rows and whether a native integer total is exact."""
    if x.dtype.kind not in "iu" or not x.size:
        return None, False
    exact_limit = 2**53 // x.shape[-1]
    if x.dtype.itemsize < 8:
        limits = np.iinfo(x.dtype)
        return None, limits.max <= exact_limit and limits.min >= -exact_limit
    # Scalar reductions reject ordinary categorical blocks without allocating
    # per-row minima, maxima, or flags. Only exceptional blocks need those arrays.
    upper = int(np.max(x))
    lower = 0 if x.dtype.kind == "u" else int(np.min(x))
    if upper <= 2**53 and lower >= -(2**53):
        return None, upper <= exact_limit and lower >= -exact_limit
    large: np.ndarray = np.max(x, axis=-1) > 2**53
    if x.dtype.kind == "i":
        large |= np.min(x, axis=-1) < -(2**53)
    return large, False


def _integer_means(x: np.ndarray) -> np.ndarray:
    """Accumulate exceptional integer totals exactly, then round the final mean."""
    return np.asarray(_integer_totals(x) / x.shape[-1], dtype=float)


def _integer_totals(x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Sum 64-bit integer observations exactly with bounded native workspaces."""
    x = np.asarray(x, dtype=np.uint64 if x.dtype.kind == "u" else np.int64)
    if x.shape[axis] <= 2**31:
        # Each 32-bit part has a safe signed 64-bit total for these row lengths.
        # Only the reduced totals need Python integers, rather than every cell.
        high = np.sum(x >> 32, axis=axis, dtype=np.int64)
        low = np.sum(x & (2**32 - 1), axis=axis, dtype=np.int64)
        totals: np.ndarray = high.astype(object) * 2**32 + low.astype(object)
    else:
        totals = np.sum(x, axis=axis, dtype=object)
    return totals


def _integer_mean_std_counts_block(
    x: np.ndarray, large: np.ndarray, *, ignore_nan: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift large integer profiles before conversion, retaining adjacent values."""
    means = np.empty(x.shape[:-1])
    deviations = np.empty_like(means)
    counts = np.full(x.shape[:-1], x.shape[-1], dtype=np.intp)
    values = x[large]
    means[large] = _integer_means(values)
    offsets = _integer_offsets(values)
    offsets -= np.mean(offsets, axis=-1)[:, np.newaxis]
    deviations[large] = np.sqrt(np.einsum("ij,ij->i", offsets, offsets) / x.shape[-1])
    if not np.all(large):
        means[~large], deviations[~large], _ = _row_mean_std_counts_block(
            x[~large], ignore_nan=ignore_nan
        )
    return means, deviations, counts


def _integer_offsets(
    x: np.ndarray, valid: np.ndarray | None = None, *, minima: np.ndarray | None = None
) -> np.ndarray:
    """Convert exact integer distances from each row's observed minimum to float."""
    if minima is None:
        minima = np.min(
            x,
            axis=-1,
            where=valid if valid is not None else True,
            initial=np.iinfo(x.dtype).max,
        )
    lower = minima.astype(np.uint64)
    # Modular unsigned subtraction also handles the full signed 64-bit range.
    offsets: np.ndarray = (x.astype(np.uint64) - lower[..., np.newaxis]).astype(float)
    if valid is not None:
        np.copyto(offsets, np.nan, where=~valid)
    return offsets


def _row_mean_std_counts_block(
    x: np.ndarray, *, ignore_nan: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce one bounded block, retaining counts for other missing-aware summaries."""
    large, integer_sum = _integer_reduction_parameters(x)
    if large is not None:
        return _integer_mean_std_counts_block(x, large, ignore_nan=ignore_nan)
    means, counts, valid = _row_mean_counts_block(
        x, ignore_nan=ignore_nan, check_integer_range=False, integer_sum=integer_sum
    )
    usable = np.isfinite(means)
    if np.count_nonzero(usable) < means.size / 2:
        # When most rows are unavailable, only center the remaining responses.
        deviations = np.full(means.shape, np.nan)
        if np.any(usable):
            means[usable], deviations[usable], _ = _row_mean_std_counts_block(
                x[usable], ignore_nan=ignore_nan
            )
        return means, deviations, counts
    centered = np.empty_like(x, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        np.subtract(x, means[..., np.newaxis], out=centered)
        if valid is not None:
            np.copyto(centered, 0.0, where=~valid)
        squared_deviations = np.einsum("...j,...j->...", centered, centered)
        deviations = np.divide(
            squared_deviations,
            counts,
            out=np.full(x.shape[:-1], np.nan),
            where=counts > 0,
        )
        unstable = _unstable_moments(centered, deviations, means=means) & usable
        np.sqrt(deviations, out=deviations)
    if np.any(unstable):
        _repair_scaled_moments(x, means, deviations, unstable, ignore_nan=ignore_nan)
    return means, deviations, counts


def _unstable_moments(
    centered: np.ndarray,
    squares: np.ndarray,
    *,
    means: np.ndarray | None = None,
    counts: np.ndarray | int = 1,
) -> np.ndarray:
    """Find overflow, underflow, and variance vulnerable to baseline rounding."""
    unstable: np.ndarray = ~np.isfinite(squares) | (
        (squares > 0) & (squares < np.finfo(float).tiny)
    )
    zero = squares == 0
    if np.any(zero):
        unstable[zero] = np.any(centered[zero] != 0, axis=-1)
    if means is not None:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            # Compare squared quantities without another square-root allocation.
            threshold = (means * np.finfo(float).eps) * means * counts
            unstable |= (squares > 0) & (squares < threshold)
    return unstable


def _repair_scaled_moments(
    x: np.ndarray,
    means: np.ndarray,
    deviations: np.ndarray | None,
    unstable: np.ndarray,
    *,
    ignore_nan: bool,
) -> None:
    """Recompute exceptional finite rows at a safe power-of-two scale."""
    finite = np.isfinite(x)
    if ignore_nan:
        finite |= np.isnan(x)
    unstable &= np.all(finite, axis=-1)
    if not np.any(unstable):
        return

    positions = np.flatnonzero(unstable)
    values = np.asarray(x[unstable], dtype=float)
    valid = ~np.isnan(values)
    counts = valid.sum(axis=-1, dtype=np.intp)
    first = np.argmax(valid, axis=-1)
    anchors = values[np.arange(len(values)), first]
    constant = np.all((values == anchors[:, np.newaxis]) | ~valid, axis=-1)
    if np.any(constant):
        means.flat[positions[constant]] = anchors[constant]
        if deviations is not None:
            deviations.flat[positions[constant]] = 0.0
        if np.all(constant):
            return
        positions = positions[~constant]
        values, valid, counts, first = (
            values[~constant],
            valid[~constant],
            counts[~constant],
            first[~constant],
        )
    scale = np.max(np.abs(values), axis=-1, where=valid, initial=0.0)
    _, exponents = np.frexp(scale)
    with np.errstate(under="ignore"):
        np.ldexp(values, -exponents[:, np.newaxis], out=values)
        lower = np.min(values, axis=-1, where=valid, initial=np.inf)
        upper = np.max(values, axis=-1, where=valid, initial=-np.inf)
        close = ((lower > 0) & (upper <= 2 * lower)) | ((upper < 0) & (lower >= 2 * upper))
        # Only shift nearby observations: otherwise reconstructing a small mean
        # from a large anchor could discard cancellation between opposite signs.
        anchors = np.where(close, values[np.arange(len(values)), first], 0.0)
        values -= anchors[:, np.newaxis]
        offsets = np.sum(values, axis=-1, where=valid) / counts
        scaled_means = anchors + offsets
        # Rounding must not move a finite mean outside its observed range.
        np.clip(scaled_means, lower, upper, out=scaled_means)
        means.flat[positions] = np.ldexp(scaled_means, exponents)
        if deviations is not None:
            values -= offsets[:, np.newaxis]
            np.copyto(values, 0.0, where=~valid)
            scaled_deviations = np.sqrt(np.einsum("ij,ij->i", values, values) / counts)
            # Population deviation cannot exceed the largest observed magnitude.
            np.minimum(scaled_deviations, np.maximum(-lower, upper), out=scaled_deviations)
            deviations.flat[positions] = np.ldexp(scaled_deviations, exponents)
