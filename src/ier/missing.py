"""Missing-response diagnostics for survey matrices.

Response omissions can be an important quality signal when they are not caused by
planned skip logic. The helpers here quantify missingness without imputing or
otherwise changing the response matrix.
"""

from collections.abc import Sequence

import numpy as np

from ier._flagging import threshold_flags, validate_percentile, validate_threshold
from ier._validation import MatrixLike, validate_matrix_input


def _select_items(x: np.ndarray, item_indices: Sequence[int] | None) -> np.ndarray:
    """Return the requested item columns after validating their indices."""
    if item_indices is None:
        return x

    selected = list(item_indices)
    if not selected:
        raise ValueError("item_indices cannot be empty")
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("item_indices must contain integer column indices")
        if index < 0 or index >= x.shape[1]:
            raise ValueError(f"item index {index} out of bounds for data with {x.shape[1]} columns")
    if len(set(selected)) != len(selected):
        raise ValueError("item_indices cannot contain duplicates")
    return x[:, selected]


def missing_rate(
    x: MatrixLike,
    item_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Calculate each respondent's proportion of missing item responses.

    Parameters:
    - x: A respondent × item response matrix.
    - item_indices: Optional 0-based subset of columns to evaluate. By default,
                    all item columns contribute equally.

    Returns:
    - A float array in ``[0, 1]``. Zero means a complete response row and one
      means every selected response is missing.

    Raises:
    - ValueError: If the matrix or item selection is invalid.

    Example:
        >>> import numpy as np
        >>> missing_rate([[1, np.nan, 3], [np.nan, np.nan, 2]])
        array([0.33333333, 0.66666667])
    """
    x_array = validate_matrix_input(x, dtype=float, check_type=False)
    selected = _select_items(x_array, item_indices)
    result: np.ndarray = np.mean(np.isnan(selected), axis=1)
    return result


def missing_rate_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 95.0,
    item_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate missing-response rates and flag unusually incomplete rows.

    An explicit threshold flags rates at or above the cutoff. Without a fixed
    threshold, rates strictly above the requested sample percentile are flagged.

    Parameters:
    - x: A respondent × item response matrix.
    - threshold: Optional fixed rate in ``[0, 1]``.
    - percentile: Sample percentile in ``[0, 100]`` used when threshold is None.
    - item_indices: Optional 0-based subset of columns to evaluate.

    Returns:
    - Tuple of ``(rates, flags)`` aligned to respondent rows.
    """
    percentile = validate_percentile(percentile)
    threshold = validate_threshold(threshold)
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite rate between 0 and 1")

    scores = missing_rate(x, item_indices=item_indices)
    flags = threshold_flags(
        scores,
        threshold=threshold,
        percentile=percentile,
        direction="high",
        inclusive=threshold is not None,
    )
    return scores, flags
