"""
Person-total correlation for detecting careless responding.

The person-total correlation measures how similar an individual's response pattern
is to the overall sample mean response pattern. Low correlations may indicate
careless or random responding.
"""

import numpy as np

from ier._correlation import row_correlations
from ier._validation import MatrixLike, validate_matrix_input

_PERSON_TOTAL_BATCH_ELEMENTS = 262_144


def person_total(
    x: MatrixLike,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Calculate person-total correlation for each individual.

    The person-total correlation (also called "personal biserial") measures
    the correlation between each individual's responses and the mean response
    across all individuals for each item. Low values suggest responses that
    deviate substantially from typical patterns.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - na_rm: If True, use pairwise complete observations for correlations.

    Returns:
    - A numpy array of person-total correlations for each individual.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        >>> scores = person_total(data)
        >>> print(scores)
        [1.0, -1.0, 1.0]
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_rows, n_items = x_array.shape
    batch_rows = max(1, _PERSON_TOTAL_BATCH_ELEMENTS // n_items)

    if not na_rm and np.isnan(x_array).any():
        return np.full(n_rows, np.nan)

    item_means = _item_means(x_array, na_rm=na_rm, batch_rows=batch_rows)
    correlations = np.empty(n_rows)
    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        block = x_array[start:stop]
        correlations[start:stop] = row_correlations(
            block,
            np.broadcast_to(item_means, block.shape),
            zero_variance=np.nan,
        )

    return correlations


def _item_means(
    x: np.ndarray,
    *,
    na_rm: bool,
    batch_rows: int,
) -> np.ndarray:
    """Calculate column means without a complete floating-point copy."""
    if not na_rm:
        result: np.ndarray = np.mean(x, axis=0)
        return result

    sums = np.zeros(x.shape[1])
    counts = np.zeros(x.shape[1], dtype=np.intp)
    for start in range(0, len(x), batch_rows):
        block = x[start : start + batch_rows]
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
