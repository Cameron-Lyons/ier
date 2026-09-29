"""This module contains the evenodd function for calculating even-odd consistency scores."""

from collections.abc import Sequence
from operator import index

import numpy as np

from ier._correlation import row_correlations
from ier._validation import MatrixLike, validate_matrix_input


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


def _validate_factors(factors: Sequence[int]) -> list[int]:
    """Return factor sizes as positive Python integers."""
    if len(factors) == 0:
        raise ValueError("factors cannot be empty")

    normalized: list[int] = []
    for factor_size in factors:
        if isinstance(factor_size, (bool, np.bool_)):
            raise ValueError("factors must contain positive integers")
        try:
            size = index(factor_size)
        except TypeError as error:
            raise ValueError("factors must contain positive integers") from error
        if size < 1:
            raise ValueError("factors must contain positive integers")
        normalized.append(size)
    return normalized


def evenodd(
    x: MatrixLike, factors: Sequence[int], diag: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.

    This function splits each factor into even and odd columns, calculates correlations
    between corresponding pairs, and returns the average correlation for each individual.
    Factors with fewer than four items cannot provide two paired observations and do not
    contribute to the average. For odd-sized factors, the final item is unpaired.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - factors: Positive integers specifying the length of each factor in the dataset.
               The sum of factors should equal the number of columns in x.
    - diag: Boolean to optionally return diagnostic values
            (number of valid correlations per individual).

    Returns:
    - A numpy array of even-odd consistency scores (average correlations per individual).
      A respondent without any valid factor correlation receives ``NaN``.
    - If diag=True, returns a tuple of (scores, diagnostic_values)

    Raises:
    - ValueError: If factors are invalid, don't sum to the number of columns, or if data
                  is empty.

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        >>> factors = [6]
        >>> scores = evenodd(data, factors)
        >>> print(scores)
        [1. 1.]
    """
    factor_sizes = _validate_factors(factors)

    x_array = validate_matrix_input(x, allow_1d=True, check_type=False)
    if x_array.dtype.kind not in "biuf":
        x_array = np.asarray(x_array, dtype=float)
    num_individuals = x_array.shape[0]

    expected_cols = sum(factor_sizes)
    if x_array.shape[1] != expected_cols:
        raise ValueError(
            f"sum of factors ({expected_cols}) must equal number of columns ({x_array.shape[1]})"
        )

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
