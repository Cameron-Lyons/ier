"""
Guttman errors for person-fit analysis in detecting careless responding.

Guttman errors count the number of response reversals relative to item
difficulty ordering. High error counts suggest inconsistent or careless responding.
"""

import warnings

import numpy as np

from ier._row_statistics import row_slices
from ier._validation import MatrixLike, validate_matrix_input

_MAX_CATEGORIES = 64
_GUTTMAN_BATCH_CELLS = 1_000_000


def guttman(
    x: MatrixLike,
    na_rm: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calculate Guttman errors for each individual.

    Guttman errors measure the number of times a person's responses violate
    the expected ordering based on item difficulty (mean endorsement).
    An error occurs when a person scores higher on a harder item than an
    easier item.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - na_rm: If True, handle missing values by excluding them from comparisons.
    - normalize: If True, return proportion of errors (0-1 scale).
                 If False, return raw error counts.

    Returns:
    - A numpy array of Guttman error scores for each individual.
      Higher values indicate more inconsistent responding.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        >>> scores = guttman(data)
        >>> print(scores)  # Second person has high errors (reversed pattern)
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_persons = x_array.shape[0]
    n_items = x_array.shape[1]

    item_difficulty = _item_difficulties(x_array, ignore_nan=na_rm)
    difficulty_order = np.argsort(item_difficulty)
    categories = _small_categorical_values(x_array)
    errors, valid_counts = _count_guttman_errors(
        x_array,
        difficulty_order,
        categories,
        count_valid=na_rm,
    )

    comparisons: np.ndarray
    if valid_counts is not None:
        comparisons = valid_counts * (valid_counts - 1.0) / 2.0
    else:
        comparisons = np.full(n_persons, n_items * (n_items - 1) / 2.0, dtype=float)

    result: np.ndarray
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = errors / comparisons
        result = np.where(comparisons == 0, np.nan, result)
    else:
        result = errors

    return result


def _item_difficulties(x: np.ndarray, *, ignore_nan: bool) -> np.ndarray:
    """Calculate item means without a complete missing-value mask."""
    if not ignore_nan:
        result: np.ndarray = np.mean(x, axis=0)
        return result

    sums = np.zeros(x.shape[1])
    counts = np.zeros(x.shape[1], dtype=np.intp)
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        valid = ~np.isnan(block)
        sums += np.sum(block, axis=0, dtype=float, where=valid)
        counts += np.sum(valid, axis=0, dtype=np.intp)

    means: np.ndarray = np.divide(
        sums,
        counts,
        out=np.full(x.shape[1], np.nan),
        where=counts > 0,
    )
    return means


def _small_categorical_values(x: np.ndarray) -> np.ndarray | None:
    """Return up to 64 ordered categories using only bounded scan workspaces."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        minimum = float(np.nanmin(x))
        maximum = float(np.nanmax(x))
    if np.isnan(minimum) or np.isnan(maximum):
        return np.array([], dtype=float)

    span = maximum - minimum
    if span < _MAX_CATEGORIES:
        for start, stop in row_slices(len(x), x.shape[1]):
            block = x[start:stop]
            values = block[~np.isnan(block)]
            if np.any(values != np.floor(values)):
                break
        else:
            return minimum + np.arange(int(span) + 1, dtype=float)

    categories = np.array([], dtype=float)
    for start, stop in row_slices(len(x), x.shape[1]):
        block = x[start:stop]
        values = block[~np.isnan(block)]
        block_categories = np.unique(values)
        if len(block_categories) > _MAX_CATEGORIES:
            return None
        categories = np.union1d(categories, block_categories)
        if len(categories) > _MAX_CATEGORIES:
            return None
    return categories


def _count_guttman_errors(
    x: np.ndarray,
    difficulty_order: np.ndarray,
    categories: np.ndarray | None,
    *,
    count_valid: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Count increasing response pairs in bounded row batches."""
    n_people, n_items = x.shape
    batch_rows = max(1, _GUTTMAN_BATCH_CELLS // n_items)
    errors = np.zeros(n_people, dtype=np.int64)
    valid_counts = np.empty(n_people) if count_valid else None

    for start in range(0, n_people, batch_rows):
        stop = min(start + batch_rows, n_people)
        block = x[start:stop, difficulty_order]
        if valid_counts is not None:
            valid_counts[start:stop] = np.count_nonzero(~np.isnan(block), axis=1)
        if categories is None:
            errors[start:stop] = _count_pairwise_errors(block)
        else:
            errors[start:stop] = _count_categorical_errors(block, categories)

    return errors.astype(float), valid_counts


def _count_categorical_errors(
    x_sorted: np.ndarray,
    categories: np.ndarray,
) -> np.ndarray:
    """Count increasing pairs by grouping positions on an ordered response scale."""
    errors = np.zeros(x_sorted.shape[0], dtype=np.int64)
    lower_categories = np.zeros(x_sorted.shape, dtype=bool)

    for category in range(1, len(categories)):
        lower_categories |= x_sorted == categories[category - 1]
        prior_lower = np.cumsum(lower_categories, axis=1, dtype=np.int32)
        errors += np.einsum(
            "ij,ij->i",
            prior_lower,
            x_sorted == categories[category],
            dtype=np.int64,
        )

    return errors


def _count_pairwise_errors(x_sorted: np.ndarray) -> np.ndarray:
    """Count pairs for one bounded high-cardinality response block."""
    n_people, n_items = x_sorted.shape
    errors = np.zeros(n_people, dtype=np.int64)

    for column in range(1, n_items):
        errors += np.count_nonzero(
            x_sorted[:, :column] < x_sorted[:, column, np.newaxis],
            axis=1,
        )

    return errors


def guttman_flag(
    x: MatrixLike,
    threshold: float = 0.5,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Flag individuals with high Guttman error rates.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - threshold: Error rate threshold for flagging (default 0.5).
    - na_rm: If True, handle missing values.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        >>> flags = guttman_flag(data, threshold=0.4)
    """
    scores = guttman(x, na_rm=na_rm, normalize=True)
    return scores > threshold
