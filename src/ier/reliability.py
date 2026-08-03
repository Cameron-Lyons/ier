"""
Resampled individual reliability for detecting careless responding.

This method estimates the reliability/consistency of each individual's
responses using split-half or bootstrap approaches.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def individual_reliability(
    x: MatrixLike,
    n_splits: int = 100,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Calculate resampled individual reliability for each person.

    Estimates how consistent each individual's responses are by repeatedly
    splitting items into halves and correlating the split scores.
    Low reliability suggests inconsistent (potentially careless) responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - n_splits: Number of random split-half iterations (default 100).
    - random_seed: Optional seed for an isolated reproducible random stream.

    Returns:
    - A numpy array of reliability estimates for each individual.
      Values range from -1 to 1, with higher values indicating more
      consistent responding.

    Raises:
    - ValueError: If inputs are invalid or too few items

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 1, 5], [3, 3, 3, 3, 3, 3]]
        >>> rel = individual_reliability(data, n_splits=50)
        >>> print(rel)  # First person: high, second: variable, third: undefined
    """
    x_array = validate_matrix_input(x, min_columns=4)
    n_persons = x_array.shape[0]
    n_items = x_array.shape[1]

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")

    random_state = np.random.RandomState(random_seed) if random_seed is not None else None
    correlation_sum = np.zeros(n_persons)
    valid_split_counts = np.zeros(n_persons, dtype=np.intp)
    half = n_items // 2
    has_missing = bool(np.isnan(x_array).any())

    for _ in range(n_splits):
        indices = (
            np.random.permutation(n_items)
            if random_state is None
            else random_state.permutation(n_items)
        )
        first_half = indices[:half]
        second_half = indices[half : 2 * half]

        half1 = x_array[:, first_half]
        half2 = x_array[:, second_half]
        split_corr, usable = _paired_split_correlations(half1, half2, has_missing)
        correlation_sum += split_corr
        valid_split_counts += usable

    reliability = np.divide(
        correlation_sum,
        valid_split_counts,
        out=np.full(n_persons, np.nan),
        where=valid_split_counts > 0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        result: np.ndarray = (2 * reliability) / (1 + reliability)

    return result


def _paired_split_correlations(
    half1: np.ndarray,
    half2: np.ndarray,
    has_missing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid row correlations for one paired random split."""
    enough_values: np.ndarray | None = None
    if has_missing:
        valid = ~np.isnan(half1) & ~np.isnan(half2)
        valid_counts = valid.sum(axis=1)
        nonempty = valid_counts > 0
        mean1 = np.divide(
            np.sum(half1, axis=1, where=valid),
            valid_counts,
            out=np.zeros(len(half1)),
            where=nonempty,
        )[:, None]
        mean2 = np.divide(
            np.sum(half2, axis=1, where=valid),
            valid_counts,
            out=np.zeros(len(half2)),
            where=nonempty,
        )[:, None]
        centered1 = np.where(valid, half1 - mean1, 0.0)
        centered2 = np.where(valid, half2 - mean2, 0.0)
        enough_values = valid_counts >= 2
    else:
        centered1 = half1 - np.mean(half1, axis=1, keepdims=True)
        centered2 = half2 - np.mean(half2, axis=1, keepdims=True)

    covariance = np.einsum("ij,ij->i", centered1, centered2)
    sum_squares1 = np.einsum("ij,ij->i", centered1, centered1)
    sum_squares2 = np.einsum("ij,ij->i", centered2, centered2)
    denominator = np.sqrt(sum_squares1 * sum_squares2)
    usable = denominator > 0
    if enough_values is not None:
        usable &= enough_values

    correlations = np.divide(
        covariance,
        denominator,
        out=np.zeros(len(half1)),
        where=usable,
    )
    return correlations, usable


def individual_reliability_flag(
    x: MatrixLike,
    threshold: float = 0.3,
    n_splits: int = 100,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Flag individuals with low reliability scores.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - threshold: Reliability threshold below which to flag (default 0.3).
    - n_splits: Number of split-half iterations.
    - random_seed: Optional seed for an isolated reproducible random stream.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 1, 5], [3, 3, 3, 3, 3, 3]]
        >>> flags = individual_reliability_flag(data, threshold=0.5)
    """
    rel = individual_reliability(x, n_splits=n_splits, random_seed=random_seed)
    result: np.ndarray = (rel < threshold) | np.isnan(rel)
    return result
