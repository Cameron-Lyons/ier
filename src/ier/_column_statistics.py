"""Bounded column-wise statistical reductions."""

import numpy as np

from ier._row_statistics import (
    _integer_offsets,
    _integer_reduction_parameters,
    _integer_totals,
    row_slices,
)


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
    """Order item means, preserving exact integer totals when float ranks would tie."""
    if x.dtype.kind in "iu":
        # Rank exact totals before conversion. Float keys retain the same
        # tie handling as the ordinary floating-point difficulty path.
        _, keys = np.unique(_integer_column_totals(x), return_inverse=True)
        return np.argsort(keys.astype(float))
    return np.argsort(column_mean(x, ignore_nan=ignore_nan))


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
