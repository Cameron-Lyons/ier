"""This module contains the evenodd function for calculating even-odd consistency scores."""

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np

from ier._correlation import row_correlations
from ier._row_statistics import row_slices
from ier._scale_halves import (
    HalfMeans,
    factor_bounds,
    group_batches,
    spearman_brown,
    validate_factor_columns,
    validate_factors,
)
from ier._validation import MatrixLike, validate_matrix_input
from ier.types import EvenOddMethod


def calculate_correlations(even_cols: np.ndarray, odd_cols: np.ndarray) -> np.ndarray:
    """
    Calculates correlations between even and odd columns for each individual.

    Parameters:
    - even_cols: Array of even-indexed columns (rows are individuals)
    - odd_cols: Array of odd-indexed columns (rows are individuals)

    Returns:
    - Array of correlation coefficients for each individual
    """
    return row_correlations(even_cols, odd_cols)


@overload
def evenodd(
    x: MatrixLike,
    factors: Sequence[int],
    diag: Literal[False] = False,
    *,
    method: EvenOddMethod = "item_pairs",
) -> np.ndarray: ...


@overload
def evenodd(
    x: MatrixLike,
    factors: Sequence[int],
    diag: Literal[True],
    *,
    method: EvenOddMethod = "item_pairs",
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def evenodd(
    x: MatrixLike,
    factors: Sequence[int],
    diag: bool,
    *,
    method: EvenOddMethod = "item_pairs",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...


def evenodd(
    x: MatrixLike,
    factors: Sequence[int],
    diag: bool = False,
    *,
    method: EvenOddMethod = "item_pairs",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.

    Two algorithms are available; higher scores always indicate more consistent responding.

    ``method="halves"`` (recommended) is the classic even-odd index of Johnson (2005),
    Meade and Craig (2012), and Curran (2016). Each factor is split into the mean of
    its odd-position items and the mean of its even-position items, ignoring missing
    responses. The two vectors of factor half means are correlated within each
    respondent across factors, using factors where both halves are available, and the
    correlation is Spearman–Brown corrected as ``2r / (1 + r)`` and clamped below at -1.
    Each half mean is its exact mean rounded once, as R's extended-precision ``mean()``
    computes it, so equal exact half means are identical and the score equals
    ``-careless::evenodd``. Where half means vary only in their last binary digit, R's
    long-double ``cor()`` keeps rounding error of up to about 1e-7 that the correlation
    here avoids. Respondents with fewer than two available factors, or whose half means
    do not vary across factors, receive ``NaN``; straight-lined responses are therefore
    unavailable here and are covered by ``longstring``.

    ``method="item_pairs"`` (the default, kept for compatibility) correlates odd items
    with the adjacent even items within each factor and averages the available factor
    correlations. Factors with fewer than four items cannot provide two paired
    observations and do not contribute to the average. For odd-sized factors, the
    final item is unpaired. Because the items of one factor share a trait level, these
    within-factor correlations carry little signal for attentive respondents.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - factors: Positive integers specifying the length of each factor in the dataset.
               The sum of factors should equal the number of columns in x.
               ``method="halves"`` requires at least two factors.
    - diag: Boolean to optionally return diagnostic values. For ``"item_pairs"`` this
            is the number of valid factor correlations per individual; for ``"halves"``
            it is the number of factors whose two half means are both finite.
    - method: ``"item_pairs"`` (default) or ``"halves"``.

    Returns:
    - A numpy array of even-odd consistency scores. A respondent without a valid
      score receives ``NaN``.
    - If diag=True, returns a tuple of (scores, diagnostic_values)

    Raises:
    - ValueError: If factors are invalid, don't sum to the number of columns, if data
                  is empty, if method is unknown, or if ``method="halves"`` receives
                  fewer than two factors.

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        >>> factors = [6]
        >>> scores = evenodd(data, factors)
        >>> print(scores)
        [1. 1.]
        >>> data = [[1, 1, 3, 3, 5, 5], [1, 5, 5, 1, 2, 4], [3, 3, 3, 3, 3, 3]]
        >>> scores, available = evenodd(data, [2, 2, 2], diag=True, method="halves")
        >>> print(np.round(scores, 2).tolist())
        [1.0, -1.0, nan]
        >>> print(available.tolist())
        [3, 3, 3]
    """
    if method not in ("item_pairs", "halves"):
        raise ValueError("method must be 'item_pairs' or 'halves'")
    factor_sizes = validate_factors(factors)
    if method == "halves" and len(factor_sizes) < 2:
        raise ValueError("method='halves' requires at least two factors")

    x_array = validate_matrix_input(x, allow_1d=True)
    num_individuals = x_array.shape[0]

    validate_factor_columns(factor_sizes, x_array.shape[1])

    if method == "halves":
        scores, available = _half_scale_scores(x_array, factor_sizes)
        return (scores, available) if diag else scores

    correlation_sum = np.zeros(num_individuals)
    diag_vals = np.zeros(num_individuals, dtype=np.intp)

    start_col = 0
    for factor_size in factor_sizes:
        if factor_size < 4:
            start_col += factor_size
            continue

        end_col = start_col + factor_size

        even_cols = x_array[:, start_col:end_col:2]
        odd_cols = x_array[:, start_col + 1 : end_col : 2]

        corrs = calculate_correlations(even_cols, odd_cols)
        valid = ~np.isnan(corrs)
        np.add(correlation_sum, corrs, out=correlation_sum, where=valid)
        np.add(diag_vals, valid, out=diag_vals, casting="unsafe")

        start_col = end_col

    avg_correlations = np.divide(
        correlation_sum,
        diag_vals,
        out=np.full(num_individuals, np.nan),
        where=diag_vals > 0,
    )

    return (avg_correlations, diag_vals) if diag else avg_correlations


def _half_scale_scores(x: np.ndarray, factor_sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Correlate odd- and even-position factor means across factors, then step them up."""
    bounds = factor_bounds(factor_sizes)
    n_factors = len(bounds)
    # 1-based odd positions start each factor; 1-based even positions follow them.
    batches = group_batches(
        [np.arange(start + 1, stop, 2) for start, stop in bounds]
        + [np.arange(start, stop, 2) for start, stop in bounds]
    )
    correlations = np.empty(len(x))
    available = np.empty(len(x), dtype=np.intp)
    # Budget the source rows, their split halves, both half-mean matrices, and the
    # correlation workspace.
    for start, stop in row_slices(len(x), 2 * x.shape[1] + 4 * n_factors):
        # Correctly rounded half means are equal whenever their exact means are, so a
        # profile whose half means do not vary has zero variance, as in careless.
        means = HalfMeans(x[start:stop])(batches, 2 * n_factors)
        even_means, odd_means = means[:, :n_factors], means[:, n_factors:]
        available[start:stop] = np.count_nonzero(
            np.isfinite(even_means) & np.isfinite(odd_means), axis=1
        )
        correlations[start:stop] = row_correlations(even_means, odd_means, zero_variance=np.nan)
    return spearman_brown(correlations), available
