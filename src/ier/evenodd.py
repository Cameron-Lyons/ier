"""This module contains the evenodd function for calculating even-odd consistency scores."""

import numpy as np

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
    if even_cols.shape[0] != odd_cols.shape[0]:
        raise ValueError("even_cols and odd_cols must have the same number of rows")

    num_individuals = even_cols.shape[0]
    min_cols = min(even_cols.shape[1], odd_cols.shape[1])

    if min_cols < 2:
        return np.full(num_individuals, np.nan)

    even_vals = even_cols[:, :min_cols]
    odd_vals = odd_cols[:, :min_cols]
    has_missing = bool(np.isnan(even_vals).any() or np.isnan(odd_vals).any())

    enough_values: np.ndarray | None = None
    with np.errstate(invalid="ignore", divide="ignore"):
        if has_missing:
            valid_mask = ~(np.isnan(even_vals) | np.isnan(odd_vals))
            valid_counts = valid_mask.sum(axis=1)
            nonempty = valid_counts > 0
            even_mean = np.divide(
                np.sum(even_vals, axis=1, where=valid_mask),
                valid_counts,
                out=np.zeros(num_individuals),
                where=nonempty,
            )
            odd_mean = np.divide(
                np.sum(odd_vals, axis=1, where=valid_mask),
                valid_counts,
                out=np.zeros(num_individuals),
                where=nonempty,
            )
            even_centered = np.zeros(even_vals.shape, dtype=float)
            odd_centered = np.zeros(odd_vals.shape, dtype=float)
            np.subtract(
                even_vals,
                even_mean[:, np.newaxis],
                out=even_centered,
                where=valid_mask,
            )
            np.subtract(
                odd_vals,
                odd_mean[:, np.newaxis],
                out=odd_centered,
                where=valid_mask,
            )
            enough_values = valid_counts >= 2
        else:
            even_centered = even_vals - np.mean(even_vals, axis=1, keepdims=True)
            odd_centered = odd_vals - np.mean(odd_vals, axis=1, keepdims=True)

        covariance = np.einsum("ij,ij->i", even_centered, odd_centered)
        denominator = np.einsum("ij,ij->i", even_centered, even_centered)
        denominator *= np.einsum("ij,ij->i", odd_centered, odd_centered)
        np.sqrt(denominator, out=denominator)
        correlations: np.ndarray = np.divide(
            covariance,
            denominator,
            out=np.zeros(num_individuals),
            where=denominator > 0,
        )

    np.clip(correlations, -1.0, 1.0, out=correlations)
    if enough_values is not None:
        correlations[~enough_values] = np.nan

    return correlations


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
