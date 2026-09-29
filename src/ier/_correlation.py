"""Allocation-conscious row-wise correlation helpers."""

from collections.abc import Callable

import numpy as np

from ier._row_statistics import (
    _integer_offsets,
    _integer_reduction_parameters,
    _unstable_moments,
    row_slices,
)


def row_correlations(
    left: np.ndarray,
    right: np.ndarray,
    *,
    zero_variance: float = 0.0,
) -> np.ndarray:
    """Return pairwise-complete row correlations with bounded temporary storage."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right must be two-dimensional")
    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must have the same number of rows")

    n_rows = left.shape[0]
    n_columns = min(left.shape[1], right.shape[1])
    if n_columns < 2:
        return np.full(n_rows, np.nan)

    correlations = np.empty(n_rows)
    # Include the per-row moments and masks when batching narrow matrices.
    for start, stop in row_slices(n_rows, max(n_columns, 16)):
        correlations[start:stop] = _row_correlations_block(
            left[start:stop, :n_columns],
            right[start:stop, :n_columns],
            zero_variance=zero_variance,
        )
    return correlations


def selected_row_correlations(
    x: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    has_missing: bool,
    zero_variance: float,
) -> np.ndarray:
    """Correlate a bounded block's selected columns, reusing the owned selection buffers."""

    def restore(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return x[np.ix_(rows, left_indices)], x[np.ix_(rows, right_indices)]

    return _row_correlations_block(
        x[:, left_indices],
        x[:, right_indices],
        zero_variance=zero_variance,
        has_missing=has_missing,
        restore=restore,
    )


def _row_correlations_block(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    zero_variance: float,
    rescale: bool = True,
    has_missing: bool | None = None,
    restore: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """Correlate one block, restoring exceptional rows when selection buffers are reused.

    A restore callback permits in-place centering of owned selection buffers;
    it must return the original paired values for the requested Boolean row mask.
    The final rescaled pass also owns its buffers. Otherwise inputs remain untouched.
    """
    if left_values.shape[1] == 2:
        return _two_pair_correlations(left_values, right_values, zero_variance=zero_variance)
    n_rows = len(left_values)
    if has_missing is None:
        has_missing = bool(np.isnan(left_values).any() or np.isnan(right_values).any())

    integer_sum_left = integer_sum_right = False
    if rescale:
        large_left, integer_sum_left = _integer_reduction_parameters(left_values)
        large_right, integer_sum_right = _integer_reduction_parameters(right_values)
        if large_left is not None or large_right is not None:
            paired = ~(np.isnan(left_values) | np.isnan(right_values)) if has_missing else None
            return _row_correlations_block(
                _integer_offsets(left_values, paired)
                if left_values.dtype.kind in "iu"
                else left_values,
                _integer_offsets(right_values, paired)
                if right_values.dtype.kind in "iu"
                else right_values,
                zero_variance=zero_variance,
                has_missing=has_missing,
            )

    left_dtype = (
        (np.uint64 if left_values.dtype.kind == "u" else np.int64) if integer_sum_left else float
    )
    right_dtype = (
        (np.uint64 if right_values.dtype.kind == "u" else np.int64) if integer_sum_right else float
    )

    enough_values: np.ndarray | None = None
    two_pairs: np.ndarray | None = None
    exact_pairs: np.ndarray | None = None
    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        if has_missing:
            valid = ~(np.isnan(left_values) | np.isnan(right_values))
            valid_counts = valid.sum(axis=1)
            two_pairs = valid_counts == 2
            if np.any(two_pairs):
                rows = np.flatnonzero(two_pairs)
                first = np.argmax(valid[two_pairs], axis=1)
                last = valid.shape[1] - 1 - np.argmax(valid[two_pairs, ::-1], axis=1)
                exact_pairs = _two_pair_correlations(
                    np.column_stack((left_values[rows, first], left_values[rows, last])),
                    np.column_stack((right_values[rows, first], right_values[rows, last])),
                    zero_variance=zero_variance,
                )
            nonempty = valid_counts > 0
            left_mean = np.divide(
                np.sum(left_values, axis=1, dtype=left_dtype, where=valid),
                valid_counts,
                out=np.zeros(n_rows),
                where=nonempty,
            )
            right_mean = np.divide(
                np.sum(right_values, axis=1, dtype=right_dtype, where=valid),
                valid_counts,
                out=np.zeros(n_rows),
                where=nonempty,
            )
            left_mean = left_mean[:, np.newaxis]
            right_mean = right_mean[:, np.newaxis]
            enough_values = valid_counts >= 2
        else:
            left_mean = (
                np.sum(left_values, axis=1, dtype=left_dtype, keepdims=True) / left_values.shape[1]
            )
            right_mean = (
                np.sum(right_values, axis=1, dtype=right_dtype, keepdims=True)
                / right_values.shape[1]
            )

        if restore is None and rescale:
            left_centered = left_values - left_mean
            right_centered = right_values - right_mean
        else:
            left_centered = np.asarray(left_values, dtype=float)
            right_centered = np.asarray(right_values, dtype=float)
            left_centered -= left_mean
            right_centered -= right_mean
        if has_missing:
            invalid = ~valid
            np.copyto(left_centered, 0.0, where=invalid)
            np.copyto(right_centered, 0.0, where=invalid)

        covariance = np.einsum("ij,ij->i", left_centered, right_centered)
        left_squares = np.einsum("ij,ij->i", left_centered, left_centered)
        right_squares = np.einsum("ij,ij->i", right_centered, right_centered)
        usable = (left_squares > 0) & (right_squares > 0)
        # Dividing by each norm avoids overflow/underflow in their product.
        correlations: np.ndarray = np.divide(
            covariance,
            np.sqrt(left_squares),
            out=np.full(n_rows, zero_variance),
            where=usable,
        )
        np.divide(correlations, np.sqrt(right_squares), out=correlations, where=usable)

    if rescale:
        counts = valid_counts if has_missing else left_values.shape[1]
        unstable = _unstable_moments(
            left_centered, left_squares, means=left_mean[:, 0], counts=counts
        )
        unstable |= _unstable_moments(
            right_centered, right_squares, means=right_mean[:, 0], counts=counts
        )
        if exact_pairs is not None:
            unstable[two_pairs] = False
        if np.any(unstable):
            if restore is None:
                original_left, original_right = left_values[unstable], right_values[unstable]
            else:
                original_left, original_right = restore(unstable)
            correlations[unstable] = _rescaled_correlations(
                original_left, original_right, zero_variance=zero_variance
            )
    np.clip(correlations, -1.0, 1.0, out=correlations)
    if enough_values is not None:
        correlations[~enough_values] = np.nan
    if exact_pairs is not None:
        correlations[two_pairs] = exact_pairs

    return correlations


def _two_pair_correlations(
    left: np.ndarray, right: np.ndarray, *, zero_variance: float
) -> np.ndarray:
    """Two nonconstant paired observations correlate by direction alone."""
    left_direction = (left[:, 1] > left[:, 0]).astype(np.int8)
    left_direction -= left[:, 1] < left[:, 0]
    right_direction = (right[:, 1] > right[:, 0]).astype(np.int8)
    right_direction -= right[:, 1] < right[:, 0]
    correlations = (left_direction * right_direction).astype(float)
    correlations[(left_direction == 0) | (right_direction == 0)] = zero_variance
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    correlations[~valid] = np.nan
    np.clip(correlations, -1.0, 1.0, out=correlations)
    return correlations


def _rescaled_correlations(
    left: np.ndarray, right: np.ndarray, *, zero_variance: float
) -> np.ndarray:
    """Rescale and shift owned exceptional pairs; infinities stay unavailable."""
    missing = np.isnan(left) | np.isnan(right)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    left[missing] = right[missing] = np.nan
    valid = ~missing
    finite = ~np.isinf(left).any(axis=1) & ~np.isinf(right).any(axis=1)
    finite &= np.count_nonzero(valid, axis=1) >= 2
    correlations = np.full(len(left), np.nan)
    if not np.any(finite):
        return correlations
    positions = np.flatnonzero(finite)
    if not np.all(finite):
        left, right, valid = left[finite], right[finite], valid[finite]
    first = np.argmax(valid, axis=1)
    rows = np.arange(len(left))
    constant = np.all((left == left[rows, first, None]) | ~valid, axis=1)
    constant |= np.all((right == right[rows, first, None]) | ~valid, axis=1)
    if np.any(constant):
        correlations[positions[constant]] = zero_variance
        if np.all(constant):
            return correlations
        positions = positions[~constant]
        left, right, valid, first = (
            left[~constant],
            right[~constant],
            valid[~constant],
            first[~constant],
        )
    rows = np.arange(len(left))
    with np.errstate(under="ignore"):
        for values in (left, right):
            scale = np.fmax.reduce(np.abs(values), axis=1)
            _, exponents = np.frexp(scale)
            np.ldexp(values, -exponents[:, None], out=values)
            np.subtract(values, values[rows, first, None], out=values)
    correlations[positions] = _row_correlations_block(
        left,
        right,
        zero_variance=zero_variance,
        rescale=False,
        has_missing=bool(np.any(~valid)),
    )
    return correlations
