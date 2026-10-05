"""
Resampled individual reliability for detecting careless responding.

This method estimates the reliability/consistency of each individual's
responses using split-half or bootstrap approaches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ier._correlation import row_correlations, selected_row_correlations
from ier._row_statistics import row_slices
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

    Example:
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


def _half_scale_reliability(
    x: np.ndarray,
    factor_sizes: list[int],
    n_splits: int,
    permutation: Callable[[int], np.ndarray],
) -> np.ndarray:
    """Average across-scale correlations of random within-scale half means."""
    bounds = factor_bounds(factor_sizes)
    n_factors = len(bounds)
    splits: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for _ in range(n_splits):
        first_groups: list[np.ndarray] = []
        second_groups: list[np.ndarray] = []
        for start, stop in bounds:
            half = (stop - start) // 2
            order = start + permutation(stop - start)
            first_groups.append(order[:half])
            second_groups.append(order[half : 2 * half])
        splits.append(group_batches(first_groups + second_groups))

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
        positions: slice | np.ndarray = slice(start, stop)
        if np.any(constant):
            active = np.flatnonzero(~constant)
            if not len(active):
                continue
            block = block[active]
            positions = start + active
        block_sum = correlation_sum[positions]
        block_counts = valid_split_counts[positions]
        half_means = HalfMeans(block)
        for batches in splits:
            means = half_means(batches, 2 * n_factors)
            split_corr = row_correlations(
                means[:, :n_factors], means[:, n_factors:], zero_variance=np.nan
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
    return spearman_brown(reliability)


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

    Example:
        >>> data = [[1, 1, 3, 3, 5, 5], [1, 5, 5, 1, 2, 4], [3, 3, 3, 3, 3, 3]]
        >>> flags = individual_reliability_flag(data, n_splits=20, factors=[2, 2, 2])
        >>> print(flags.tolist())
        [False, True, True]
    """
    rel = individual_reliability(x, n_splits=n_splits, random_seed=random_seed, factors=factors)
    result: np.ndarray = (rel < threshold) | np.isnan(rel)
    return result
