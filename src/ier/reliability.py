"""
Resampled individual reliability for detecting careless responding.

This method estimates the reliability/consistency of each individual's
responses using split-half or bootstrap approaches.
"""

from operator import index

import numpy as np

from ier._correlation import selected_row_correlations
from ier._row_statistics import row_slices
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
    - A numpy array of Spearman–Brown corrected mean split correlations.
      Higher values indicate more consistent responding. Finite values are
      at most 1 and can be below -1. Respondents without valid splits or with
      a mean split correlation of -1 receive ``NaN``.

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

    if isinstance(n_splits, (bool, np.bool_)):
        raise ValueError("n_splits must be a positive integer")
    try:
        n_splits = index(n_splits)
    except TypeError as error:
        raise ValueError("n_splits must be a positive integer") from error
    if n_splits < 1:
        raise ValueError("n_splits must be a positive integer")

    random_state = np.random.RandomState(random_seed) if random_seed is not None else None
    half = n_items // 2
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
    # Account for both selected halves and the shared correlation workspaces.
    for start, stop in row_slices(n_persons, max(2 * half, 16)):
        block = x_array[start:stop]
        missing = np.isnan(block)
        has_missing = bool(missing.any())
        if has_missing:
            first = np.argmax(~missing, axis=1)
            anchors = block[np.arange(len(block)), first, None]
            constant = np.all((block == anchors) | missing, axis=1)
        else:
            constant = np.all(block == block[:, :1], axis=1)
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
