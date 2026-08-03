"""This module contains the evenodd function for calculating even-odd consistency scores."""

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


def evenodd(
    x: MatrixLike, factors: list[int], diag: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate even-odd consistency scores for each individual based on the provided factors.

    This function splits each factor into even and odd columns, calculates correlations
    between corresponding pairs, and returns the average correlation for each individual.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are their responses.
          Can be a 2D list or numpy array.
    - factors: List of integers specifying the length of each factor in the dataset.
               The sum of factors should equal the number of columns in x.
    - diag: Boolean to optionally return diagnostic values
            (number of valid correlations per individual).

    Returns:
    - A numpy array of even-odd consistency scores (average correlations per individual)
    - If diag=True, returns a tuple of (scores, diagnostic_values)

    Raises:
    - ValueError: If factors don't sum to the number of columns, or if data is empty

    Example:
        >>> data = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        >>> factors = [4, 2]
        >>> scores = evenodd(data, factors)
        >>> print(scores)
        [0.5, 0.5]
    """

    if not factors:
        raise ValueError("factors list cannot be empty")

    x_array = validate_matrix_input(x, allow_1d=True, dtype=float, check_type=False)
    num_individuals = x_array.shape[0]

    expected_cols = sum(factors)
    if x_array.shape[1] != expected_cols:
        raise ValueError(
            f"sum of factors ({expected_cols}) must equal number of columns ({x_array.shape[1]})"
        )

    correlation_sum = np.zeros(num_individuals)
    diag_vals = np.zeros(num_individuals, dtype=np.intp)
    has_scored_factor = False

    start_col = 0
    for factor_size in factors:
        if factor_size < 2:
            start_col += factor_size
            continue

        end_col = start_col + factor_size

        even_cols = x_array[:, start_col:end_col:2]
        odd_cols = x_array[:, start_col + 1 : end_col : 2]

        corrs = calculate_correlations(even_cols, odd_cols)
        valid = ~np.isnan(corrs)
        np.add(correlation_sum, corrs, out=correlation_sum, where=valid)
        np.add(diag_vals, valid, out=diag_vals, casting="unsafe")
        has_scored_factor = True

        start_col = end_col

    if has_scored_factor:
        avg_correlations = np.divide(
            correlation_sum,
            diag_vals,
            out=np.zeros(num_individuals),
            where=diag_vals > 0,
        )
    else:
        avg_correlations = np.full(num_individuals, np.nan)

    return (avg_correlations, diag_vals) if diag else avg_correlations
