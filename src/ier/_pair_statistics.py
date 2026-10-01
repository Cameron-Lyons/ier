"""Bounded reductions for predefined item pairs."""

import math
from collections.abc import Sequence
from fractions import Fraction
from operator import index

import numpy as np

from ier._row_statistics import _integer_reduction_parameters, row_mean, row_slices


def validate_item_pairs(
    item_pairs: Sequence[tuple[int, int]], n_columns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate explicit pairs without truncating entries or coercing item indices."""
    if len(item_pairs) == 0:
        raise ValueError("item_pairs cannot be empty")
    try:
        left, right = zip(*item_pairs, strict=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "each item pair must contain exactly two integer column indices"
        ) from error
    return validate_paired_item_indices(
        left, right, n_columns, left_name="item_pairs", right_name="item_pairs"
    )


def validate_paired_item_indices(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    n_columns: int,
    *,
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate two ordered, equally sized item-index lists."""
    if len(left_indices) == 0 or len(right_indices) == 0:
        raise ValueError(f"{left_name} and {right_name} cannot be empty")
    if len(left_indices) != len(right_indices):
        raise ValueError(f"{left_name} and {right_name} must contain the same number of items")

    normalized: list[np.ndarray] = []
    for name, values in ((left_name, left_indices), (right_name, right_indices)):
        result = np.empty(len(values), dtype=np.intp)
        for position, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must contain integer column indices")
            try:
                item_index = index(value)
            except TypeError as error:
                raise ValueError(f"{name} must contain integer column indices") from error
            if item_index < 0 or item_index >= n_columns:
                raise ValueError(
                    f"item index {item_index} out of bounds for data with {n_columns} columns"
                )
            result[position] = item_index
        normalized.append(result)

    return normalized[0], normalized[1]


def paired_mean_absolute_difference(
    x: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    right_bounds: tuple[float, float] | None,
    ignore_nan: bool,
    normalizers: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce paired differences, optionally dividing before restoring extreme units."""
    if len(left_indices) != len(right_indices):
        raise ValueError("paired index arrays must contain the same number of items")
    n_pairs = len(left_indices)
    if n_pairs == 0:
        raise ValueError("paired index arrays cannot be empty")
    scores = np.empty(len(x))
    reflection = None
    integer_pairs = x.dtype.kind in "iu"
    exact_reflection = None
    centered_reflection = False
    if right_bounds is not None:
        reflection = right_bounds[0] + right_bounds[1]
        centered_reflection = (
            not math.isfinite(reflection)
            or abs(reflection) * math.sqrt(np.finfo(float).eps) > right_bounds[1] - right_bounds[0]
        )
        integer_pairs &= all(math.isfinite(value) for value in right_bounds)
        if integer_pairs:
            exact_reflection = Fraction(right_bounds[0]) + Fraction(right_bounds[1])

    for start, stop in row_slices(len(x), n_pairs):
        divisors = None if normalizers is None else normalizers[start:stop]
        left_values = x[start:stop, left_indices]
        right_values = x[start:stop, right_indices]
        if integer_pairs:
            large_left, _ = _integer_reduction_parameters(left_values)
            large_right, _ = _integer_reduction_parameters(right_values)
            if large_left is not None or large_right is not None:
                scores[start:stop] = _integer_pair_means(
                    left_values, right_values, exact_reflection, divisors
                )
                del left_values, right_values
                continue
            if exact_reflection is None or (
                exact_reflection.denominator == 1 and abs(exact_reflection.numerator) <= 2**53
            ):
                # Ordinary categories and bounds fit signed arithmetic exactly.
                left = np.asarray(left_values, dtype=np.int64)
                right = np.asarray(right_values, dtype=np.int64)
                if exact_reflection is None:
                    left -= right
                else:
                    left += right
                    left -= exact_reflection.numerator
                np.abs(left, out=left)
                block_scores = row_mean(left, ignore_nan=False)
                if divisors is not None:
                    with np.errstate(
                        over="ignore", invalid="ignore", divide="ignore", under="ignore"
                    ):
                        block_scores /= divisors
                scores[start:stop] = block_scores
                del left, right, left_values, right_values
                continue
        left = np.asarray(left_values, dtype=float)
        right = np.asarray(right_values, dtype=float)
        del left_values, right_values
        with np.errstate(over="raise", invalid="ignore", under="ignore"):
            try:
                if centered_reflection and right_bounds is not None:
                    if reflection is not None and not math.isfinite(reflection):
                        smaller = np.minimum(left, right)
                        np.maximum(left, right, out=right)
                        left = smaller
                        del smaller
                    np.subtract(left, right_bounds[0], out=left)
                    np.subtract(right, right_bounds[1], out=right)
                    np.add(left, right, out=left)
                else:
                    if reflection is not None:
                        np.subtract(reflection, right, out=right)
                    np.subtract(left, right, out=left)
            except FloatingPointError:
                scores[start:stop] = _rescaled_pair_difference(
                    x[start:stop], left_indices, right_indices, right_bounds, ignore_nan, divisors
                )
                del left, right
                continue
        np.abs(left, out=left)
        block_scores = row_mean(left, ignore_nan=ignore_nan)
        if divisors is not None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                block_scores /= divisors
        scores[start:stop] = block_scores
        del left, right

    return scores


def _integer_pair_means(
    left_values: np.ndarray,
    right_values: np.ndarray,
    reflection: Fraction | None,
    normalizers: np.ndarray | None,
) -> np.ndarray:
    """Subtract large integers before rounding, including fractional scale bounds."""
    left = left_values.astype(object)
    right = right_values.astype(object)
    denominator = 1
    if reflection is None:
        left -= right
    else:
        denominator = reflection.denominator
        left += right
        left *= denominator
        left -= reflection.numerator
    np.abs(left, out=left)
    totals = np.sum(left, axis=1)
    means = np.empty(len(left))
    for row, total in enumerate(totals):
        value = Fraction(total, denominator * left.shape[1])
        if normalizers is not None:
            value /= Fraction(float(normalizers[row]))
        try:
            means[row] = float(value)
        except OverflowError:
            means[row] = np.inf
    return means


def _rescaled_pair_difference(
    x: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    bounds: tuple[float, float] | None,
    ignore_nan: bool,
    normalizers: np.ndarray | None,
) -> np.ndarray:
    """Recover overflowing differences in a common power-of-two coordinate system."""
    left = np.asarray(x[:, left_indices], dtype=float)
    right = np.asarray(x[:, right_indices], dtype=float)
    reflection = 0.0
    remainder = 0.0
    if bounds is not None:
        if all(math.isfinite(value) for value in bounds):
            exact_reflection = Fraction(bounds[0]) + Fraction(bounds[1])
            try:
                reflection = float(exact_reflection)
            except OverflowError:
                reflection = math.inf
            if math.isfinite(reflection):
                remainder = float(exact_reflection - Fraction(reflection))
        else:
            reflection = bounds[0] + bounds[1]
    magnitudes = np.maximum(
        np.max(np.abs(left), axis=1, where=np.isfinite(left), initial=0),
        np.max(np.abs(right), axis=1, where=np.isfinite(right), initial=0),
    )
    if bounds is not None:
        bound = (
            abs(reflection) if math.isfinite(reflection) else max(abs(bounds[0]), abs(bounds[1]))
        )
        magnitudes = np.maximum(magnitudes, bound)
    _, exponents = np.frexp(magnitudes)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        np.ldexp(left, -exponents[:, None], out=left)
        np.ldexp(right, -exponents[:, None], out=right)
        if bounds is None:
            left -= right
        elif reflection == 0:
            left += right
        elif math.isfinite(reflection):
            scaled_reflection = np.ldexp(reflection, -exponents)[:, None]
            # Sum the two largest terms first so cancellation preserves the small term.
            left_magnitudes, right_magnitudes = np.abs(left), np.abs(right)
            sum_first = np.abs(scaled_reflection) <= np.minimum(left_magnitudes, right_magnitudes)
            right_larger = right_magnitudes > left_magnitudes
            larger = np.where(right_larger, right, left)
            smaller = np.where(right_larger, left, right)
            np.subtract(larger, scaled_reflection, out=larger, where=~sum_first)
            larger += smaller
            np.subtract(larger, scaled_reflection, out=larger, where=sum_first)
            if remainder:
                larger -= np.ldexp(remainder, -exponents)[:, None]
            left = larger
            del larger, smaller, left_magnitudes, right_magnitudes, sum_first, right_larger
        else:
            smaller = np.minimum(left, right)
            np.maximum(left, right, out=right)
            left = smaller
            left -= np.ldexp(bounds[0], -exponents)[:, None]
            right -= np.ldexp(bounds[1], -exponents)[:, None]
            left += right
        del right
        np.abs(left, out=left)
        means = row_mean(left, ignore_nan=ignore_nan)
        repair = np.isfinite(means) & (means <= 8 * np.finfo(float).eps)
        nonzero = np.isfinite(means) & (means > 0)
        if normalizers is None:
            np.ldexp(means, exponents, out=means)
        else:
            means /= np.ldexp(normalizers, -exponents)
        repair |= nonzero & ((np.abs(means) < np.finfo(float).tiny) | ~np.isfinite(means))
        if normalizers is not None:
            repair &= np.isfinite(normalizers) & (normalizers != 0)
        for row in np.flatnonzero(repair):
            means[row] = _exact_pair_mean(
                x[row],
                left_indices,
                right_indices,
                bounds,
                ignore_nan,
                None if normalizers is None else float(normalizers[row]),
            )
    return means


def _exact_pair_mean(
    row: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    bounds: tuple[float, float] | None,
    ignore_nan: bool,
    normalizer: float | None,
) -> float:
    """Repair a cancellation or subnormal mean before its final float rounding."""
    reflection = Fraction(0) if bounds is None else Fraction(bounds[0]) + Fraction(bounds[1])
    total = Fraction(0)
    count = 0
    for left_index, right_index in zip(left_indices, right_indices, strict=True):
        left, right = float(row[left_index]), float(row[right_index])
        if math.isnan(left) or math.isnan(right):
            if ignore_nan:
                continue
            return math.nan
        if not math.isfinite(left) or not math.isfinite(right):
            return math.nan
        difference = (
            Fraction(left) - Fraction(right)
            if bounds is None
            else Fraction(left) + Fraction(right) - reflection
        )
        total += abs(difference)
        count += 1
    if not count:
        return math.nan
    result = total / count
    if normalizer is not None:
        if normalizer == 0 or not math.isfinite(normalizer):
            return math.nan
        result /= Fraction(normalizer)
    try:
        return float(result)
    except OverflowError:
        return math.inf if result >= 0 else -math.inf
