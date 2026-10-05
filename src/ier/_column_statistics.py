"""Bounded column-wise statistical reductions."""

from decimal import Decimal, localcontext

import numpy as np

from ier._row_statistics import (
    _integer_offsets,
    _integer_reduction_parameters,
    _integer_totals,
    row_slices,
)
from ier._validation import validate_integer


def column_mean(x: np.ndarray, *, ignore_nan: bool, center_integers: bool = False) -> np.ndarray:
    """Calculate item means, optionally shifting integer totals for item profiles.

    A common shift preserves small differences for ordering and correlations;
    callers requesting raw means retain their original units by default.
    """
    dtype = np.result_type(x.dtype, np.float64)
    if not len(x):
        return np.full(x.shape[1], np.nan, dtype=dtype)
    if x.dtype.kind in "iu" and x.shape[1]:
        totals = _integer_column_totals(x)
        if center_integers:
            totals -= np.min(totals)
        return np.asarray(totals / len(x), dtype=float)
    if not ignore_nan:
        with np.errstate(over="raise", invalid="ignore"):
            try:
                result: np.ndarray = np.mean(x, axis=0, dtype=dtype)
            except FloatingPointError:
                with np.errstate(over="ignore"):
                    result = np.mean(x, axis=0, dtype=dtype)
                _repair_column_means(x, result, ~np.isfinite(result), ignore_nan=False)
        return result

    sums = np.zeros(x.shape[1], dtype=dtype)
    counts = np.zeros(x.shape[1], dtype=np.intp)
    with np.errstate(over="ignore", invalid="ignore"):
        for start, stop in row_slices(len(x), x.shape[1]):
            block = x[start:stop]
            valid = ~np.isnan(block)
            if np.all(valid):
                sums += np.sum(block, axis=0, dtype=dtype)
                counts += len(block)
            else:
                sums += np.sum(block, axis=0, dtype=dtype, where=valid)
                counts += np.sum(valid, axis=0, dtype=np.intp)
            del valid

        means = np.divide(
            sums,
            counts,
            out=np.full(x.shape[1], np.nan, dtype=dtype),
            where=counts > 0,
        )
    unstable = ~np.isfinite(means) & (counts > 0)
    if np.any(unstable):
        _repair_column_means(x, means, unstable, ignore_nan=True)
    return means


def _integer_column_totals(x: np.ndarray) -> np.ndarray:
    """Accumulate exact integer item totals without retaining response objects."""
    totals = np.zeros(x.shape[1], dtype=object)
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        _, integer_sum = _integer_reduction_parameters(block.T)
        totals += (
            np.sum(block, axis=0, dtype=np.uint64 if x.dtype.kind == "u" else np.int64)
            if integer_sum
            else _integer_totals(block, axis=0)
        )
    return totals


def column_mean_order(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Order items easiest first: largest mean first, ties by column position.

    Integer items are ranked by exact totals, so nearby large means that would
    round to one float still order correctly. Unavailable (``NaN``) means sort
    last, because negation preserves ``NaN`` and NumPy sorts it after numbers.
    With ``ignore_nan=True`` only items without any observed response are
    unavailable; with ``ignore_nan=False`` one missing response suffices.
    """
    if x.dtype.kind in "iu":
        # Rank exact totals before conversion. Float keys retain the same
        # tie handling as the ordinary floating-point difficulty path.
        _, keys = np.unique(_integer_column_totals(x), return_inverse=True)
        return np.argsort(-keys.reshape(-1).astype(float), kind="stable")
    return np.argsort(-column_mean(x, ignore_nan=ignore_nan), kind="stable")


def column_mean_profile(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Retain item-mean differences without restoring the original response units.

    Correlations tolerate a common shift and positive scale. Exceptional floating
    profiles need that transformation before averaging: rounding their means in
    the original units can erase subnormal or small baseline-relative differences.
    """
    means = column_mean(x, ignore_nan=ignore_nan, center_integers=True)
    if x.dtype.kind != "f":
        return means
    columns = np.flatnonzero(np.isfinite(means))
    if len(columns) < 2:
        return means
    observed_means = means[columns]
    precision = np.finfo(means.dtype)
    magnitude = np.max(np.abs(observed_means))
    with np.errstate(over="ignore"):
        span = np.max(observed_means) - np.min(observed_means)
    if span > precision.tiny and span >= magnitude * np.sqrt(precision.eps):
        return means

    scales, usable, _ = _column_scales(x, columns, ignore_nan=ignore_nan)
    columns = columns[usable]
    if len(columns) < 2:
        return means
    magnitude = np.max(scales[usable])
    if magnitude == 0:
        return means
    if means.dtype.itemsize <= 8 and np.max(np.abs(observed_means)) < magnitude * np.sqrt(
        precision.eps
    ):
        # A small profile after cancellation can coexist with enormous individual
        # responses. A common float scale would erase the small residual responses.
        _exact_mean_profile(x, means, columns, ignore_nan=ignore_nan)
        return means
    _, exponent = np.frexp(magnitude)
    anchor = np.ldexp(means[columns[0]], -exponent)
    totals = np.zeros(len(columns), dtype=means.dtype)
    counts = np.zeros(len(columns), dtype=np.intp)
    centered_minimum, centered_maximum = float("inf"), -float("inf")
    all_columns = len(columns) == x.shape[1]
    for start, stop in row_slices(len(x), len(columns)):
        block = (
            np.array(x[start:stop], dtype=means.dtype, copy=True)
            if all_columns
            else np.asarray(x[start:stop, columns], dtype=means.dtype)
        )
        with np.errstate(under="ignore"):
            np.ldexp(block, -exponent, out=block)
        block -= anchor
        centered_minimum = min(
            centered_minimum, float(np.fmin.reduce(block, axis=None, initial=float("inf")))
        )
        centered_maximum = max(
            centered_maximum, float(np.fmax.reduce(block, axis=None, initial=-float("inf")))
        )
        if ignore_nan:
            valid = ~np.isnan(block)
            totals += np.sum(block, axis=0, where=valid)
            counts += np.count_nonzero(valid, axis=0)
            del valid
        else:
            totals += np.sum(block, axis=0)
            counts += len(block)
        del block
    means[columns] = totals / counts
    span = np.max(means[columns]) - np.min(means[columns])
    response_span = centered_maximum - centered_minimum
    if means.dtype.itemsize <= 8 and span < response_span * np.sqrt(precision.eps):
        # Differently ordered cancellation can manufacture item differences,
        # even when every exact sample mean is identical. Resolve small profile
        # residuals against the actual centered response scale before correlating.
        _exact_mean_profile(x, means, columns, ignore_nan=ignore_nan)
    return means


def _exact_mean_profile(
    x: np.ndarray, means: np.ndarray, columns: np.ndarray, *, ignore_nan: bool
) -> None:
    """Retain finite item-mean residuals across the complete float64 exponent range."""
    with localcontext() as context:
        context.prec = 800
        totals = [Decimal(0) for _ in columns]
        counts = np.zeros(len(columns), dtype=np.intp)
        for start, stop in row_slices(len(x), len(columns)):
            block = x[start:stop, columns]
            for position, column in enumerate(block.T):
                observed = column[~np.isnan(column)] if ignore_nan else column
                totals[position] += sum(
                    (Decimal.from_float(float(value)) for value in observed), start=Decimal(0)
                )
                counts[position] += len(observed)
        averages = [total / int(count) for total, count in zip(totals, counts, strict=True)]
        offsets = [value - averages[0] for value in averages]
        scale = max(abs(value) for value in offsets)
        means[columns] = [float(value / scale) if scale else 0.0 for value in offsets]


def column_correlations(x: np.ndarray) -> np.ndarray:
    """Correlate items with bounded centering workspaces and NaN propagation.

    Missing observations invalidate their entire item, as in ``np.corrcoef``.
    Constant items and samples with fewer than two rows have undefined
    correlations. The item-by-item output is necessarily quadratic in width.
    """
    dtype = np.result_type(x.dtype, np.float64)
    if len(x) < 2:
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=dtype)
    large, integer_sum = _integer_reduction_parameters(x.T)
    if large is not None:
        return _rescaled_column_correlations(x)

    with np.errstate(over="raise", invalid="ignore"):
        try:
            means = np.asarray(
                np.sum(x, axis=0, dtype=np.uint64 if x.dtype.kind == "u" else np.int64) / len(x)
                if integer_sum
                else np.mean(x, axis=0, dtype=dtype)
            )
        except FloatingPointError:
            return _rescaled_column_correlations(x)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
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
        squares = correlations.diagonal().real
        precision = np.finfo(squares.dtype)
        unstable = ~np.isfinite(squares) | (squares <= precision.tiny)
        magnitude = np.maximum(np.abs(means.real), np.abs(means.imag))
        relative_deviation = np.sqrt(squares) / np.sqrt(len(x)) / magnitude
        # Means near a constant baseline can manufacture variance through
        # rounding. Shift those columns before accumulating their mean instead.
        unstable |= relative_deviation < np.sqrt(precision.eps)
        if np.any(unstable):
            candidates = np.flatnonzero(unstable)
            constant = candidates[_constant_columns(x, available[candidates])]
            correlations[constant, :] = np.nan
            correlations[:, constant] = np.nan
            unstable[constant] = False
            if np.any(unstable):
                return _rescaled_column_correlations(x)

    return _normalize_correlations(correlations, available, x.shape[1])


_PAIRWISE_CANCELLATION = 1e-2
_CROSS_PRODUCT_BUDGETS = 4


def pairwise_column_correlations(x: np.ndarray, *, min_pairs: int = 3) -> np.ndarray:
    """Correlate each item pair over the rows that observe both responses.

    This follows ``cor(use="pairwise.complete.obs")`` in R, except that a pair
    needs ``min_pairs`` shared observations (R accepts two). Two shared rows
    always correlate perfectly, so items rarely answered together, for example
    under skip logic, would otherwise become spurious pairs. Infinite responses
    invalidate their item, and items constant over a pair's shared rows leave
    that pair undefined. Without missing responses every pair shares all rows:
    fewer than ``min_pairs`` rows leave every pair undefined, and otherwise the
    result is exactly ``column_correlations(x)``.

    Each row block contributes shared counts, sums, squares, and cross-products
    as four item-by-item matrix products. Items are first scaled by a power of
    two and shifted by a median observation, so squares cannot overflow and
    integer-valued responses stay exact. Pairs whose one-pass moments could have
    lost precision, through cancellation against that shift or through underflow
    of responses far below their item's largest one, are recomputed from their
    shared rows. The moments are finished in place, so the workspace stays at the
    four accumulators, Boolean pair masks, and bounded blocks.
    """
    min_pairs = validate_integer(
        min_pairs, message="min_pairs must be an integer of at least 2", minimum=2
    )
    n_items = x.shape[1]
    dtype = np.result_type(x.dtype, np.float64)
    if x.dtype.kind != "f" or not any(
        np.isnan(x[start:stop]).any() for start, stop in row_slices(len(x), n_items)
    ):
        # Integer and Boolean responses cannot be missing, so pairs share every row.
        if len(x) < min_pairs:
            return np.full((n_items, n_items), np.nan, dtype=dtype)
        return column_correlations(x)

    scales, usable, counts = _column_scales(x, np.arange(n_items), ignore_nan=True)
    columns = np.flatnonzero(usable & (counts >= min_pairs))
    if not len(columns):
        return np.full((n_items, n_items), np.nan, dtype=dtype)
    _, exponents = np.frexp(scales[columns])
    correlations = _shared_row_correlations(x, columns, exponents, min_pairs=min_pairs)
    if len(columns) == n_items:
        return correlations
    result = np.full((n_items, n_items), np.nan, dtype=dtype)
    result[np.ix_(columns, columns)] = correlations
    return result


def _shared_row_moments(
    x: np.ndarray, columns: np.ndarray, exponents: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate shared counts, sums, squares, and cross-products for item pairs.

    Entry ``(i, j)`` of the sums and squares holds item ``i``'s moments over the
    rows it shares with item ``j``, after scaling and shifting the item.
    """
    dtype = np.result_type(x.dtype, np.float64)
    n_columns = len(columns)
    all_columns = n_columns == x.shape[1]
    anchors = np.zeros(n_columns, dtype=dtype)
    anchored = np.zeros(n_columns, dtype=bool)
    shared = np.zeros((n_columns, n_columns), dtype=dtype)
    sums = np.zeros((n_columns, n_columns), dtype=dtype)
    squares = np.zeros((n_columns, n_columns), dtype=dtype)
    products = np.zeros((n_columns, n_columns), dtype=dtype)
    # The scaled block, its mask, and its weights share the element budget.
    for start, stop in row_slices(len(x), 3 * n_columns):
        block = (
            np.array(x[start:stop], dtype=dtype, copy=True)
            if all_columns
            else np.asarray(x[start:stop, columns], dtype=dtype)
        )
        observed = ~np.isnan(block)
        _ldexp_inplace(block, -exponents)
        pending = ~anchored & np.any(observed, axis=0)
        if np.any(pending):
            # Earlier blocks contributed nothing to items they never observed. A
            # block median lies within one standard deviation of the block mean,
            # and integers stay exact. A sparsely observed first block can still
            # leave an outlying shift, which the cancellation guard repairs.
            positions = np.flatnonzero(pending)
            candidates = np.where(observed[:, positions], block[:, positions], np.inf)
            candidates.sort(axis=0)
            middle = (np.count_nonzero(observed[:, positions], axis=0) - 1) // 2
            anchors[positions] = candidates[middle, np.arange(len(positions))]
            anchored |= pending
            del candidates
        block -= anchors
        block[~observed] = 0.0
        weights = observed.astype(dtype)
        del observed
        _add_cross_products(shared, weights, weights)
        _add_cross_products(sums, block, weights)
        _add_cross_products(products, block, block)
        np.square(block, out=block)
        _add_cross_products(squares, block, weights)
        del block, weights
    return shared, sums, squares, products


def _add_cross_products(total: np.ndarray, left: np.ndarray, right: np.ndarray) -> None:
    """Add ``left.T @ right`` through products of at most four row-batch budgets.

    NumPy evaluates an unsliced ``block.T @ block`` as a faster symmetric
    product, so the product is sliced only when it exceeds that bound.
    """
    width = max(1, total.shape[1] // _CROSS_PRODUCT_BUDGETS)
    for start, stop in row_slices(len(total), width):
        total[start:stop] += left[:, start:stop].T @ right


def _shared_row_correlations(
    x: np.ndarray, columns: np.ndarray, exponents: np.ndarray, *, min_pairs: int
) -> np.ndarray:
    """Correlate usable items from shared-row moments finished in place.

    This function owns the four accumulators, so each is released once it is
    consumed, and the cross-products become the returned correlations. One-pass
    variances lose about ``eps * squares / variance`` of their relative precision,
    so pairs whose variance falls below ``_PAIRWISE_CANCELLATION`` of the squares
    are recomputed, bounding the correlation error near ``eps / 1e-2``. Responses
    far below their item's largest one can underflow after scaling; pairs whose
    variance sits near that underflow floor are recomputed when the item has such
    responses.
    """
    shared, sums, squares, products = _shared_row_moments(x, columns, exponents)
    n_columns = len(columns)
    precision = np.finfo(products.dtype)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        # Covariances replace the cross-products. Transposed operands need a
        # temporary, so it is filled in bounded row slices.
        for start, stop in row_slices(n_columns, n_columns):
            correction = sums[start:stop] * sums[:, start:stop].T
            correction /= shared[start:stop]
            products[start:stop] -= correction
            del correction
        # The sums become item i's variance over the rows it shares with item j.
        variances = sums
        del sums
        np.multiply(variances, variances, out=variances)
        variances /= shared
        np.subtract(squares, variances, out=variances)
        enough = shared >= min_pairs
        # Exact zeros fall below this floor too, so only items whose shifted
        # responses can underflow when squared are recomputed.
        shared *= precision.tiny / precision.eps
        floor = variances < shared
        del shared
        squares *= _PAIRWISE_CANCELLATION
        unstable = variances < squares
        del squares
        floor &= enough
        floor &= ~unstable
        items = np.flatnonzero(floor.any(axis=1))
        if len(items):
            fragile = np.zeros(n_columns, dtype=bool)
            fragile[items] = _underflowing_columns(x, columns[items], exponents[items])
            floor &= fragile[:, None]
            unstable |= floor
        del floor
        unstable |= unstable.T
        unstable &= enough
        deviations = np.sqrt(variances, out=variances)
        del variances
        defined = deviations > 0
        defined &= defined.T
        defined &= enough
        del enough
        correlations = products
        del products
        correlations /= deviations
        correlations /= deviations.T
        del deviations
    nonfinite = ~np.isfinite(correlations)
    nonfinite &= defined
    unstable |= nonfinite
    del nonfinite
    correlations[~defined] = np.nan
    del defined
    np.clip(correlations, -1.0, 1.0, out=correlations)
    # Repair the lower triangle, then mirror it so the result stays symmetric.
    for row, column in zip(*np.nonzero(np.tril(unstable)), strict=True):
        correlations[row, column] = _shared_row_correlation(x, columns[row], columns[column])
    _mirror_lower_triangle(correlations)
    return correlations


def _underflowing_columns(x: np.ndarray, columns: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """Find items with responses whose scaled, shifted squares can underflow.

    Scaled responses at or above ``sqrt(tiny) / eps**2`` share a grid, so their
    nonzero differences from the shift square to normal numbers. Smaller nonzero
    responses, relative to the item's power-of-two scale, can underflow.
    """
    dtype = np.result_type(x.dtype, np.float64)
    precision = np.finfo(dtype)
    limits = np.ldexp(np.sqrt(precision.tiny) / precision.eps**2, exponents)
    fragile = np.zeros(len(columns), dtype=bool)
    for start, stop in row_slices(len(x), len(columns)):
        magnitude = np.abs(np.asarray(x[start:stop, columns], dtype=dtype))
        fragile |= np.any((magnitude > 0) & (magnitude < limits), axis=0)
        del magnitude
    return fragile


def _mirror_lower_triangle(matrix: np.ndarray) -> None:
    """Copy the lower triangle over the upper one in bounded row slices."""
    n_rows = len(matrix)
    for start, stop in row_slices(n_rows, n_rows):
        matrix[start:stop, stop:] = matrix[stop:, start:stop].T
        diagonal = matrix[start:stop, start:stop]
        upper = np.triu_indices(stop - start, k=1)
        diagonal[upper] = diagonal.T[upper]


def _shared_row_correlation(x: np.ndarray, first: int, second: int) -> np.floating:
    """Correlate one item pair exactly over the rows that observe both items."""
    pair = np.array([first, second])
    shared = np.concatenate(
        [
            block[~np.isnan(block).any(axis=1)]
            for block in (x[start:stop, pair] for start, stop in row_slices(len(x), 2))
        ]
    )
    correlation: np.floating = column_correlations(shared)[0, 1]
    return correlation


def _constant_columns(x: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Verify candidate constant items against their original observations."""
    constant = np.ones(len(columns), dtype=bool)
    anchor = x[0, columns]
    for start, stop in row_slices(len(x), len(columns)):
        candidates = np.flatnonzero(constant)
        if not len(candidates):
            break
        constant[candidates] = np.all(
            x[start:stop, columns[candidates]] == anchor[candidates], axis=0
        )
    return constant


def _normalize_correlations(
    correlations: np.ndarray, available: np.ndarray, n_items: int
) -> np.ndarray:
    """Normalize cross-products directly; the covariance divisor cancels out."""
    deviations = np.sqrt(correlations.diagonal().real)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations /= deviations[:, None]
        correlations /= deviations[None, :]

    np.clip(correlations.real, -1.0, 1.0, out=correlations.real)
    if np.iscomplexobj(correlations):
        np.clip(correlations.imag, -1.0, 1.0, out=correlations.imag)
    if len(available) != n_items:
        result = np.full((n_items, n_items), np.nan, dtype=correlations.dtype)
        result[np.ix_(available, available)] = correlations
        return result
    return correlations


def _column_scales(
    x: np.ndarray, columns: np.ndarray, *, ignore_nan: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find finite component scales and observation counts without full column copies."""
    dtype = np.result_type(x.dtype, np.float64)
    scales = np.zeros(len(columns), dtype=np.empty((), dtype=dtype).real.dtype)
    usable = np.ones(len(columns), dtype=bool)
    counts = np.zeros(len(columns), dtype=np.intp)
    for start, stop in row_slices(len(x), len(columns)):
        block = np.asarray(x[start:stop, columns], dtype=dtype)
        finite = np.isfinite(block)
        if ignore_nan:
            missing = np.isnan(block)
            finite |= missing
            counts += len(block) - np.count_nonzero(missing, axis=0)
            del missing
        else:
            counts += len(block)
        usable &= np.all(finite, axis=0)
        magnitude = np.abs(block.real)
        if np.iscomplexobj(block):
            np.maximum(magnitude, np.abs(block.imag), out=magnitude)
        np.fmax(scales, np.fmax.reduce(magnitude, axis=0, initial=0.0), out=scales)
        del block, finite, magnitude
    usable &= (counts > 0) & np.isfinite(scales)
    return scales, usable, counts


def _ldexp_inplace(values: np.ndarray, exponents: np.ndarray) -> None:
    """Scale owned real or complex values by column powers of two."""
    with np.errstate(over="ignore", under="ignore"):
        np.ldexp(values.real, exponents, out=values.real)
        if np.iscomplexobj(values):
            np.ldexp(values.imag, exponents, out=values.imag)


def _repair_column_means(
    x: np.ndarray, means: np.ndarray, unstable: np.ndarray, *, ignore_nan: bool
) -> None:
    """Recompute overflowing finite item means at bounded power-of-two scales."""
    columns = np.flatnonzero(unstable)
    scales, usable, counts = _column_scales(x, columns, ignore_nan=ignore_nan)
    if not np.any(usable):
        return
    columns, scales, counts = columns[usable], scales[usable], counts[usable]
    _, exponents = np.frexp(scales)
    normalized = np.zeros(len(columns), dtype=means.dtype)
    for start, stop in row_slices(len(x), len(columns)):
        block = np.asarray(x[start:stop, columns], dtype=means.dtype)
        _ldexp_inplace(block, -exponents)
        valid = ~np.isnan(block) if ignore_nan else True
        normalized += np.sum(block, axis=0, where=valid)
        del block, valid
    normalized /= counts
    _ldexp_inplace(normalized, exponents)
    # A rounded mean must remain within the finite component magnitude bound.
    np.clip(normalized.real, -scales, scales, out=normalized.real)
    if np.iscomplexobj(normalized):
        np.clip(normalized.imag, -scales, scales, out=normalized.imag)
    means[columns] = normalized


def _rescaled_column_correlations(x: np.ndarray) -> np.ndarray:
    """Correlate exceptional columns after rescaling and shifting their baseline."""
    columns = np.arange(x.shape[1])
    scales, usable, _ = _column_scales(x, columns, ignore_nan=False)
    columns = columns[usable]
    dtype = np.result_type(x.dtype, np.float64)
    if not len(columns):
        return np.full((x.shape[1], x.shape[1]), np.nan, dtype=dtype)
    _, exponents = np.frexp(scales[usable])
    integer_minima = np.min(x, axis=0) if x.dtype.kind in "iu" else None
    anchor = (
        np.zeros(len(columns), dtype=dtype)
        if integer_minima is not None
        else np.asarray(x[0, columns], dtype=dtype)
    )
    _ldexp_inplace(anchor, -exponents)
    offsets = np.zeros(len(columns), dtype=dtype)
    for start, stop in row_slices(len(x), len(columns)):
        block = (
            _integer_offsets(x[start:stop].T, minima=integer_minima).T
            if integer_minima is not None
            else np.asarray(x[start:stop, columns], dtype=dtype)
        )
        _ldexp_inplace(block, -exponents)
        block -= anchor
        offsets += np.sum(block, axis=0)
        del block
    offsets /= len(x)

    correlations = np.zeros((len(columns), len(columns)), dtype=dtype)
    for start, stop in row_slices(len(x), len(columns)):
        centered = (
            _integer_offsets(x[start:stop].T, minima=integer_minima).T
            if integer_minima is not None
            else np.asarray(x[start:stop, columns], dtype=dtype)
        )
        _ldexp_inplace(centered, -exponents)
        centered -= anchor
        centered -= offsets
        right = centered.conj() if np.iscomplexobj(centered) else centered
        correlations += centered.T @ right
        del centered, right
    return _normalize_correlations(correlations, columns, x.shape[1])
