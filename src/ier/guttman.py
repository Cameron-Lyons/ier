"""
Guttman errors for person-fit analysis in detecting careless responding.

Guttman errors count the number of response reversals relative to item
difficulty ordering. High error counts suggest inconsistent or careless responding.
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input

_MAX_CATEGORIES = 64
_MAX_CATEGORY_CELLS = 10_000_000
_PAIRWISE_CHUNK_CELLS = 1_000_000


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

    if na_rm:
        item_counts = np.sum(~np.isnan(x_array), axis=0)
        item_difficulty = np.divide(
            np.nansum(x_array, axis=0),
            item_counts,
            out=np.full(n_items, np.nan),
            where=item_counts != 0,
        )
    else:
        item_difficulty = np.mean(x_array, axis=0)

    difficulty_order = np.argsort(item_difficulty)
    x_sorted = x_array[:, difficulty_order]

    if na_rm:
        valid_counts = np.sum(~np.isnan(x_sorted), axis=1).astype(float)
        comparisons = valid_counts * (valid_counts - 1.0) / 2.0
    else:
        comparisons = np.full(n_persons, n_items * (n_items - 1) / 2.0, dtype=float)

    errors = _count_guttman_errors(x_sorted)

    result: np.ndarray
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            result = errors / comparisons
        result = np.where(comparisons == 0, np.nan, result)
    else:
        result = errors

    return result


def _count_guttman_errors(x_sorted: np.ndarray) -> np.ndarray:
    """Count increasing response pairs without materializing every item pair."""
    valid = ~np.isnan(x_sorted)
    categorical = _encode_categorical_values(x_sorted, valid)
    if categorical is not None:
        encoded, n_categories = categorical
        return _count_categorical_errors(encoded, n_categories)
    return _count_pairwise_errors(x_sorted)


def _encode_categorical_values(
    x_sorted: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    """Encode small categorical scales, preferring bounded integer ranges."""
    values = x_sorted[valid]
    encoded = np.full(x_sorted.shape, -1, dtype=np.int16)
    if values.size == 0:
        return encoded, 0

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span < _MAX_CATEGORIES and np.all(values == np.floor(values)):
        raw_ids = (values - minimum).astype(np.intp)
        present = np.bincount(raw_ids, minlength=int(span) + 1) > 0
        n_categories = int(np.sum(present))
        if x_sorted.shape[0] * n_categories > _MAX_CATEGORY_CELLS:
            return None
        mapping = np.cumsum(present, dtype=np.int16) - 1
        encoded[valid] = mapping[raw_ids]
        return encoded, n_categories

    categories = np.unique(values)
    if (
        len(categories) > _MAX_CATEGORIES
        or x_sorted.shape[0] * len(categories) > _MAX_CATEGORY_CELLS
    ):
        return None
    encoded[valid] = np.searchsorted(categories, values)
    return encoded, len(categories)


def _count_categorical_errors(
    encoded: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """Count increasing pairs by grouping positions on a small response scale."""
    errors = np.zeros(encoded.shape[0], dtype=np.int64)
    lower_categories = np.zeros(encoded.shape, dtype=bool)

    for category in range(1, n_categories):
        lower_categories |= encoded == category - 1
        prior_lower = np.cumsum(lower_categories, axis=1, dtype=np.int32)
        errors += np.einsum(
            "ij,ij->i",
            prior_lower,
            encoded == category,
            dtype=np.int64,
        )

    return errors.astype(float)


def _count_pairwise_errors(x_sorted: np.ndarray) -> np.ndarray:
    """Count pairs in bounded row chunks for high-cardinality response data."""
    n_people, n_items = x_sorted.shape
    chunk_rows = max(1, _PAIRWISE_CHUNK_CELLS // n_items)
    errors = np.zeros(n_people, dtype=np.int64)

    for start in range(0, n_people, chunk_rows):
        stop = min(start + chunk_rows, n_people)
        block = x_sorted[start:stop]
        block_errors = np.zeros(stop - start, dtype=np.int64)
        for column in range(1, n_items):
            block_errors += np.count_nonzero(
                block[:, :column] < block[:, column, np.newaxis], axis=1
            )
        errors[start:stop] = block_errors

    return errors.astype(float)


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
