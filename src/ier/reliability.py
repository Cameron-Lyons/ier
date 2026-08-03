"""
Resampled individual reliability for detecting careless responding.

This method estimates the reliability/consistency of each individual's
responses using split-half or bootstrap approaches.
"""

import numpy as np

from ier._row_statistics import row_slices
from ier._validation import MatrixLike, validate_matrix_input

_CORRELATION_CANCELLATION_TOLERANCE = 64.0 * np.finfo(float).eps


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
    half = n_items // 2
    has_missing = any(
        np.isnan(x_array[start:stop]).any() for start, stop in row_slices(n_persons, n_items)
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_splits):
        indices = (
            np.random.permutation(n_items)
            if random_state is None
            else random_state.permutation(n_items)
        )
        splits.append((indices[:half], indices[half : 2 * half]))

    correlation_sum = np.zeros(n_persons)
    valid_split_counts = np.zeros(n_persons, dtype=np.intp)
    for start, stop in row_slices(n_persons, half):
        block = x_array[start:stop]
        for first_half, second_half in splits:
            half1 = block[:, first_half]
            half2 = block[:, second_half]
            split_corr, usable = _paired_split_correlations(half1, half2, has_missing)
            correlation_sum[start:stop] += split_corr
            valid_split_counts[start:stop] += usable

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
    """Return row correlations from raw moments with stable edge handling."""
    valid: np.ndarray | None = None
    enough_values: np.ndarray | None = None
    if has_missing:
        valid = ~np.isnan(half1) & ~np.isnan(half2)
        valid_counts = np.asarray(np.sum(valid, axis=1, dtype=np.intp), dtype=np.intp)
        nonempty = valid_counts > 0
        inverse_counts: float | np.ndarray = np.divide(
            1.0,
            valid_counts,
            out=np.zeros(len(half1)),
            where=nonempty,
        )
        missing = ~valid
        np.copyto(half1, 0.0, where=missing)
        np.copyto(half2, 0.0, where=missing)
        enough_values = valid_counts >= 2
    else:
        inverse_counts = 1.0 / half1.shape[1]

    sums1 = np.sum(half1, axis=1, dtype=float)
    sums2 = np.sum(half2, axis=1, dtype=float)
    covariance = np.einsum("ij,ij->i", half1, half2, dtype=float)
    covariance -= sums1 * sums2 * inverse_counts

    raw_squares1 = np.einsum("ij,ij->i", half1, half1, dtype=float)
    raw_squares2 = np.einsum("ij,ij->i", half2, half2, dtype=float)
    sum_squares1 = raw_squares1 - sums1 * sums1 * inverse_counts
    sum_squares2 = raw_squares2 - sums2 * sums2 * inverse_counts
    cancellation_prone = (sum_squares1 <= _CORRELATION_CANCELLATION_TOLERANCE * raw_squares1) | (
        sum_squares2 <= _CORRELATION_CANCELLATION_TOLERANCE * raw_squares2
    )
    if enough_values is not None:
        cancellation_prone &= enough_values

    np.maximum(sum_squares1, 0.0, out=sum_squares1)
    np.maximum(sum_squares2, 0.0, out=sum_squares2)
    denominator = sum_squares1 * sum_squares2
    np.sqrt(denominator, out=denominator)
    usable = denominator > 0.0
    if enough_values is not None:
        usable &= enough_values

    correlations = np.divide(
        covariance,
        denominator,
        out=np.zeros(len(half1)),
        where=usable,
    )

    if np.any(cancellation_prone):
        stable_correlations, stable_usable = _stable_paired_split_correlations(
            half1[cancellation_prone],
            half2[cancellation_prone],
            None if valid is None else valid[cancellation_prone],
        )
        correlations[cancellation_prone] = stable_correlations
        usable[cancellation_prone] = stable_usable

    return correlations, usable


def _stable_paired_split_correlations(
    half1: np.ndarray,
    half2: np.ndarray,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Center numerically cancellation-prone row pairs before correlation."""
    enough_values: np.ndarray | None = None
    if valid is not None:
        valid_counts = np.asarray(np.sum(valid, axis=1, dtype=np.intp), dtype=np.intp)
        nonempty = valid_counts > 0
        mean1 = np.divide(
            np.sum(half1, axis=1, where=valid),
            valid_counts,
            out=np.zeros(len(half1)),
            where=nonempty,
        )
        mean2 = np.divide(
            np.sum(half2, axis=1, where=valid),
            valid_counts,
            out=np.zeros(len(half2)),
            where=nonempty,
        )
        centered1 = np.zeros(half1.shape, dtype=float)
        centered2 = np.zeros(half2.shape, dtype=float)
        np.subtract(half1, mean1[:, None], out=centered1, where=valid)
        np.subtract(half2, mean2[:, None], out=centered2, where=valid)
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
