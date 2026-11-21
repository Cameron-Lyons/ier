"""
Resampled individual reliability for detecting careless responding.

This method estimates the reliability/consistency of each individual's
responses using split-half or bootstrap approaches.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ier._correlation import row_correlations, selected_row_correlations
from ier._flagging import validate_threshold
from ier._row_statistics import _ROW_BATCH_ELEMENTS, row_slices
from ier._scale_halves import (
    HalfMeans,
    constant_rows,
    factor_bounds,
    group_batches,
    spearman_brown,
    validate_factor_columns,
    validate_factors,
)
from ier._validation import MatrixLike, validate_integer, validate_matrix_input

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _split_permutation_source(
    random_seed: int | np.random.Generator | None,
) -> Callable[[int], np.ndarray]:
    """Return an isolated permutation stream that never touches NumPy's global state."""
    if random_seed is None:
        return np.random.default_rng().permutation
    if isinstance(random_seed, np.random.Generator):
        return random_seed.permutation
    # Integer seeds keep the established RandomState split sequence.
    return np.random.RandomState(random_seed).permutation


def individual_reliability(
    x: MatrixLike,
    n_splits: int = 100,
    random_seed: int | np.random.Generator | None = None,
    *,
    factors: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Calculate resampled individual reliability for each person.

    Estimates how consistent each individual's responses are by repeatedly
    splitting items into halves and correlating the split scores.
    Low reliability suggests inconsistent (potentially careless) responding.

    With ``factors`` (recommended), each resample randomly splits every scale into
    two halves of ``size // 2`` items (Curran, 2016; Huang et al., 2012). Each
    respondent's two vectors of scale half means, ignoring missing responses, are
    correlated across scales. Each half mean is its exact mean rounded once, so a
    split whose half means do not vary is unusable rather than scored from rounding
    noise. The valid correlations are averaged over resamples
    and Spearman–Brown corrected as ``2r / (1 + r)``, clamped below at -1. Scales
    with fewer than two items contribute no halves.

    Without ``factors``, the legacy method pairs randomly permuted items across the
    whole questionnaire and correlates the paired item responses. Items of one
    scale share a trait level, so these correlations carry little signal for
    attentive respondents; prefer ``factors`` when the scale structure is known.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - n_splits: Number of random split-half iterations (default 100).
    - random_seed: Optional integer seed or ``np.random.Generator`` for an isolated
                   reproducible random stream; a Generator is advanced by the call.
                   Without a seed, fresh entropy is used and NumPy's global random
                   state is neither read nor advanced.
    - factors: Optional positive scale lengths, in column order, summing to the
               number of columns. At least two scales are required.

    Returns:
    - A numpy array of Spearman–Brown corrected mean split correlations.
      Higher values indicate more consistent responding. Without ``factors``,
      finite values are at most 1 and can be below -1, and respondents without
      valid splits or with a mean split correlation of -1 receive ``NaN``. With
      ``factors``, finite values lie in [-1, 1] and respondents without a valid
      split receive ``NaN``. Constant response profiles always receive ``NaN``.

    Raises:
    - ValueError: If inputs are invalid, there are too few items, or factors are
                  invalid, fewer than two, or don't sum to the number of columns

    Examples:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 1, 5], [3, 3, 3, 3, 3, 3]]
        >>> rel = individual_reliability(data, n_splits=50, random_seed=0)
        >>> print(np.isnan(rel).tolist())  # The constant profile is undefined
        [False, False, True]
        >>> data = [[1, 1, 3, 3, 5, 5], [1, 5, 5, 1, 2, 4], [3, 3, 3, 3, 3, 3]]
        >>> rel = individual_reliability(data, n_splits=20, random_seed=0, factors=[2, 2, 2])
        >>> print(np.round(rel, 2).tolist())
        [1.0, -1.0, nan]
    """
    x_array = validate_matrix_input(x, min_columns=4)
    n_persons = x_array.shape[0]
    n_items = x_array.shape[1]

    n_splits = validate_integer(n_splits, message="n_splits must be a positive integer", minimum=1)

    permutation = _split_permutation_source(random_seed)
    if factors is not None:
        factor_sizes = validate_factors(factors)
        if len(factor_sizes) < 2:
            raise ValueError("scale-aware individual_reliability requires at least two factors")
        validate_factor_columns(factor_sizes, n_items)
        return _half_scale_reliability(x_array, factor_sizes, n_splits, permutation)

    half = n_items // 2
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_splits):
        indices = permutation(n_items)
        splits.append((indices[:half], indices[half : 2 * half]))

    correlation_sum = np.zeros(n_persons)
    valid_split_counts = np.zeros(n_persons, dtype=np.intp)
    # Account for both selected halves and the shared correlation workspaces.
    for start, stop in row_slices(n_persons, max(2 * half, 16)):
        block = x_array[start:stop]
        missing = np.isnan(block)
        has_missing = bool(missing.any())
        constant = constant_rows(block, missing if has_missing else None)
        positions: slice | np.ndarray = slice(start, stop)
        if np.any(constant):
            active = np.flatnonzero(~constant)
            if not len(active):
                continue
            block = block[active]
            has_missing = bool(missing[active].any())
            positions = start + active
        del missing
        block_sum = correlation_sum[positions]
        block_counts = valid_split_counts[positions]
        for first_half, second_half in splits:
            split_corr = selected_row_correlations(
                block,
                first_half,
                second_half,
                zero_variance=np.nan,
                has_missing=has_missing,
            )
            usable = ~np.isnan(split_corr)
            np.add(block_sum, split_corr, out=block_sum, where=usable)
            block_counts += usable
        correlation_sum[positions] = block_sum
        valid_split_counts[positions] = block_counts

    reliability = np.divide(
        correlation_sum,
        valid_split_counts,
        out=np.full(n_persons, np.nan),
        where=valid_split_counts > 0,
    )

    result: np.ndarray = np.divide(
        2 * reliability,
        1 + reliability,
        out=np.full(n_persons, np.nan),
        where=reliability > -1,
    )
    return result


# Float64 represents every integer up to this magnitude, so sums below it are exact.
_EXACT_INTEGER_LIMIT = 2**53
# A row's scaled moments are exact while n * L * max|response| stays at or below this.
_EXACT_MOMENT_LIMIT = math.sqrt(_EXACT_INTEGER_LIMIT)
# Matrix products of at most this many multiply-adds stay on one OpenBLAS thread
# (its default threshold). Threading these thin products gains nothing and can
# stall when the cores are busy, as when screen() scores indices concurrently.
_SINGLE_THREAD_PRODUCT = 2**18

# One random split: each factor's first-half and second-half column positions.
_Split: TypeAlias = tuple[list[np.ndarray], list[np.ndarray]]
# Adds each usable split correlation of a block's rows to running totals.
_SplitScorer: TypeAlias = "Callable[[np.ndarray, np.ndarray, np.ndarray], None]"


def _half_scale_reliability(
    x: np.ndarray,
    factor_sizes: list[int],
    n_splits: int,
    permutation: Callable[[int], np.ndarray],
) -> np.ndarray:
    """Average across-scale correlations of random within-scale half means.

    Each respondent is scored by the exact integer scorer when their own responses
    allow it and by the correctly rounded scorer otherwise, so a score never depends
    on other respondents' responses.
    """
    bounds = factor_bounds(factor_sizes)
    n_factors = len(bounds)
    # Draw every split first; the random stream is the same for either scorer.
    splits: list[_Split] = []
    for _ in range(n_splits):
        first_groups: list[np.ndarray] = []
        second_groups: list[np.ndarray] = []
        for start, stop in bounds:
            half = (stop - start) // 2
            order = start + permutation(stop - start)
            first_groups.append(order[:half])
            second_groups.append(order[half : 2 * half])
        splits.append((first_groups, second_groups))
    exact = _ExactSplitCorrelations.build(splits, bounds)
    rounded = _RoundedSplitCorrelations(splits, n_factors)

    n_persons = len(x)
    correlation_sum = np.zeros(n_persons)
    valid_split_counts = np.zeros(n_persons, dtype=np.intp)
    # Budget the source rows, their split halves, both half-mean matrices, and the
    # correlation workspace.
    for start, stop in row_slices(n_persons, 2 * x.shape[1] + 4 * n_factors):
        block = x[start:stop]
        missing = np.isnan(block)
        # Constant profiles have equal correctly rounded half means in every split, so
        # they never correlate; skip them before preparing the split halves.
        constant = constant_rows(block, missing if missing.any() else None)
        del missing
        active = np.flatnonzero(~constant)
        if not len(active):
            continue
        if len(active) < len(block):
            block = block[active]
        integral = np.zeros(len(block), dtype=bool) if exact is None else exact.eligible(block)
        scorers: tuple[tuple[_SplitScorer | None, np.ndarray], ...] = (
            (exact, integral),
            (rounded, ~integral),
        )
        for scorer, selected in scorers:
            if scorer is None or not selected.any():
                continue
            positions = start + active[selected]
            block_sum = correlation_sum[positions]
            block_counts = valid_split_counts[positions]
            scorer(block if selected.all() else block[selected], block_sum, block_counts)
            correlation_sum[positions] = block_sum
            valid_split_counts[positions] = block_counts

    reliability = np.divide(
        correlation_sum,
        valid_split_counts,
        out=np.full(n_persons, np.nan),
        where=valid_split_counts > 0,
    )
    return spearman_brown(reliability)


class _RoundedSplitCorrelations:
    """Correlate correctly rounded half means, one split at a time."""

    def __init__(self, splits: Sequence[_Split], n_factors: int) -> None:
        self._splits = splits
        self._batches: list[list[tuple[np.ndarray, np.ndarray]]] | None = None
        self._n_factors = n_factors

    def __call__(self, block: np.ndarray, block_sum: np.ndarray, block_counts: np.ndarray) -> None:
        """Add each usable split correlation of ``block`` rows to the running totals."""
        if self._batches is None:
            # Prepared on first use, since integer surveys may never need them.
            self._batches = [group_batches(first + second) for first, second in self._splits]
        n_factors = self._n_factors
        half_means = HalfMeans(block)
        for batches in self._batches:
            means = half_means(batches, 2 * n_factors)
            split_corr = row_correlations(
                means[:, :n_factors], means[:, n_factors:], zero_variance=np.nan
            )
            usable = ~np.isnan(split_corr)
            np.add(block_sum, split_corr, out=block_sum, where=usable)
            block_counts += usable


class _ExactSplitCorrelations:
    """Correlate exact half means of integer responses for many splits at once.

    Half means of integer responses are fractions whose denominators are observed
    half sizes, so multiplying them by a common multiple ``L`` of the possible sizes
    gives integers, and correlations do not depend on that scale. A complete row's
    halves hold exactly the half sizes, so ``L`` is their least common multiple; a
    row with missing responses can leave any count from 1 to the half size, so ``L``
    is the least common multiple of 1 to the largest half size.

    Each half total is one row of a 0/1 selection matrix applied to the factor's
    responses, computed for many splits by one matrix product per factor. With ``n``
    contributing factors, every partial sum and the moments
    ``n * sum(a * b) - sum(a) * sum(b)`` are exact integers while
    ``n * L * max|response|`` is at most ``2**26.5``, so only the final square roots
    and quotient round. Half means are equal exactly when their integers are, so
    splits whose half means do not vary are unusable, and two available factors
    correlate at exactly -1 or 1, as in the rounded scorer.
    """

    def __init__(
        self,
        splits: Sequence[_Split],
        bounds: Sequence[tuple[int, int]],
        complete_scale: int,
        missing_scale: int,
    ) -> None:
        self._n_splits = len(splits)
        # Per contributing factor: its columns, its (split, half) by item 0/1
        # selections, and those selections weighted by the complete-data scale over
        # the half size, which give complete halves their scaled means directly.
        self._factors: list[tuple[slice, np.ndarray, np.ndarray]] = []
        for factor, (start, stop) in enumerate(bounds):
            half_size = (stop - start) // 2
            if half_size == 0:
                continue  # A one-item scale has no halves in any split.
            selection = np.zeros((self._n_splits, 2, stop - start))
            for split, halves in enumerate(splits):
                for half, groups in enumerate(halves):
                    selection[split, half, groups[factor] - start] = 1.0
            weighted = selection * float(complete_scale // half_size)
            self._factors.append((slice(start, stop), selection, weighted))
        n_factors = len(self._factors)
        self._complete_scale = float(complete_scale)
        # Scales too large for any row are never applied; keep them out of float range.
        self._missing_scale = (
            float(missing_scale) if n_factors * missing_scale <= _EXACT_MOMENT_LIMIT else math.inf
        )

    @classmethod
    def build(
        cls, splits: Sequence[_Split], bounds: Sequence[tuple[int, int]]
    ) -> _ExactSplitCorrelations | None:
        """Return the scorer, or None when no respondent's moments could be exact."""
        half_sizes = [(stop - start) // 2 for start, stop in bounds]
        contributing = [size for size in half_sizes if size > 0]
        complete_scale = math.lcm(*contributing)
        # A varying integer row has a response of magnitude at least 1.
        if len(contributing) < 2 or len(contributing) * complete_scale > _EXACT_MOMENT_LIMIT:
            return None
        missing_scale = math.lcm(*range(1, max(contributing) + 1))
        return cls(splits, bounds, complete_scale, missing_scale)

    def eligible(self, block: np.ndarray) -> np.ndarray:
        """Flag rows whose scaled half means and moments are exact integers.

        A row qualifies when its observed responses are finite integers and
        ``n * L * max|response|`` is at most ``2**26.5`` for its own scale ``L``, so
        the decision depends on the row alone.
        """
        n_factors = len(self._factors)
        if block.dtype.kind in "biu":
            # Integer and Boolean responses are complete integers.
            magnitudes = np.max(np.abs(block.astype(float)), axis=1)
            within: np.ndarray = (
                n_factors * self._complete_scale * magnitudes <= _EXACT_MOMENT_LIMIT
            )
            return within
        missing = np.isnan(block)
        integral = np.all((np.isfinite(block) & (np.trunc(block) == block)) | missing, axis=1)
        complete = ~missing.any(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            # Extended-precision magnitudes beyond float64 become infinite here.
            magnitudes = np.fmax.reduce(np.abs(block), axis=1).astype(float)
            scales = np.where(complete, self._complete_scale, self._missing_scale)
            bounded = n_factors * scales * magnitudes <= _EXACT_MOMENT_LIMIT
        eligible: np.ndarray = integral & bounded
        return eligible

    def _totals(
        self, responses: np.ndarray, first: int, last: int, *, weighted: bool
    ) -> np.ndarray:
        """Sum item-major ``responses`` over each half of splits ``first`` to ``last``.

        Returns factor-major totals shaped (factor, split, half, respondent).
        """
        n_rows = responses.shape[1]
        totals = np.empty((len(self._factors), last - first, 2, n_rows))
        for position, (columns, selection, scaled) in enumerate(self._factors):
            chosen = (scaled if weighted else selection)[first:last]
            chosen = chosen.reshape(-1, chosen.shape[-1])
            factor_totals = totals[position].reshape(-1, n_rows)
            step = max(1, _SINGLE_THREAD_PRODUCT // chosen.size)
            for start in range(0, n_rows, step):
                rows = slice(start, start + step)
                np.matmul(chosen, responses[columns, rows], out=factor_totals[:, rows])
        return totals

    def __call__(self, block: np.ndarray, block_sum: np.ndarray, block_counts: np.ndarray) -> None:
        """Add each usable split correlation of eligible ``block`` rows to the totals."""
        n_rows = len(block)
        n_factors = len(self._factors)
        # Item-major rows make each factor's responses one contiguous matrix.
        values = np.array(block.T, dtype=float, order="C")
        missing = np.isnan(values)
        observed: np.ndarray | None = None
        row_scales: np.ndarray | None = None
        if missing.any():
            np.copyto(values, 0.0, where=missing)
            row_scales = np.where(missing.any(axis=0), self._missing_scale, self._complete_scale)
            observed = np.logical_not(missing, out=missing).astype(float)
        del missing
        chunk = max(1, _ROW_BATCH_ELEMENTS // (2 * n_factors * n_rows))
        for first in range(0, self._n_splits, chunk):
            last = min(first + chunk, self._n_splits)
            # Every response of a complete block is observed, so weighted totals
            # are already the scaled half means.
            scaled = self._totals(values, first, last, weighted=observed is None)
            counts: np.ndarray | int = n_factors
            if observed is not None:
                assert row_scales is not None
                sizes = self._totals(observed, first, last, weighted=False)
                # Each observed size divides its row's scale, so total * scale / size
                # is an exact integer; empty halves give 0 / 0 and are cleared below.
                scaled *= row_scales
                with np.errstate(invalid="ignore"):
                    scaled /= sizes
                paired = sizes[:, :, 0] > 0
                paired &= sizes[:, :, 1] > 0
                np.copyto(scaled, 0.0, where=~paired[:, :, np.newaxis])
                counts = np.count_nonzero(paired, axis=0)
            correlations, usable = _exact_correlations(scaled[:, :, 0], scaled[:, :, 1], counts)
            # Accumulate in split order, as the rounded scorer does.
            for split_correlations, split_usable in zip(correlations, usable, strict=True):
                np.add(block_sum, split_correlations, out=block_sum, where=split_usable)
                block_counts += split_usable


def _exact_correlations(
    left: np.ndarray, right: np.ndarray, counts: np.ndarray | int
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate integer-valued vectors along their first axis from exact moments.

    Unavailable pairs hold zeros and are excluded by ``counts``. Returns the
    correlations and a mask of the defined ones; the rest are NaN.
    """
    left_total = left.sum(axis=0)
    right_total = right.sum(axis=0)
    left_spread = counts * np.einsum("f...,f...->...", left, left) - left_total * left_total
    right_spread = counts * np.einsum("f...,f...->...", right, right) - right_total * right_total
    covariance = counts * np.einsum("f...,f...->...", left, right) - left_total * right_total
    usable = (left_spread > 0) & (right_spread > 0)
    correlations = np.divide(
        covariance, np.sqrt(left_spread), out=np.full(usable.shape, np.nan), where=usable
    )
    np.divide(correlations, np.sqrt(right_spread), out=correlations, where=usable)
    np.clip(correlations, -1.0, 1.0, out=correlations)
    return correlations, usable


def individual_reliability_flag(
    x: MatrixLike,
    threshold: float = 0.3,
    n_splits: int = 100,
    random_seed: int | np.random.Generator | None = None,
    *,
    factors: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Flag individuals with low reliability scores.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - threshold: Reliability threshold below which to flag (default 0.3).
    - n_splits: Number of split-half iterations.
    - random_seed: Optional integer seed or ``np.random.Generator`` for an isolated
                   reproducible random stream.
    - factors: Optional scale lengths for scale-aware split halves; see
               ``individual_reliability``.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Raises:
    - ValueError: If ``threshold`` is not a finite number or scoring inputs are invalid.

    Examples:
        >>> data = [[1, 1, 3, 3, 5, 5], [1, 5, 5, 1, 2, 4], [3, 3, 3, 3, 3, 3]]
        >>> flags = individual_reliability_flag(data, n_splits=20, factors=[2, 2, 2])
        >>> print(flags.tolist())
        [False, True, True]
    """
    validated_threshold = validate_threshold(threshold)
    if validated_threshold is None:
        raise ValueError("threshold must be a finite number")

    rel = individual_reliability(x, n_splits=n_splits, random_seed=random_seed, factors=factors)
    result: np.ndarray = (rel < validated_threshold) | np.isnan(rel)
    return result
