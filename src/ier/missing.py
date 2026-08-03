"""Missing-response diagnostics for survey matrices.

Response omissions can be an important quality signal when they are not caused by
planned skip logic. The helpers here quantify missingness without imputing or
otherwise changing the response matrix.
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ier._flagging import threshold_flags, validate_percentile, validate_threshold
from ier._row_statistics import row_slices
from ier._validation import MatrixLike, validate_matrix_input


def _validate_item_indices(
    n_columns: int,
    item_indices: Sequence[int] | None,
) -> np.ndarray | None:
    """Validate and normalize an optional item-column selection."""
    if item_indices is None:
        return None

    selected = list(item_indices)
    if not selected:
        raise ValueError("item_indices cannot be empty")
    for index in selected:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError("item_indices must contain integer column indices")
        if index < 0 or index >= n_columns:
            raise ValueError(f"item index {index} out of bounds for data with {n_columns} columns")
    if len(set(selected)) != len(selected):
        raise ValueError("item_indices cannot contain duplicates")
    return np.asarray(selected, dtype=np.intp)


def _select_item_block(
    x: np.ndarray, start: int, stop: int, item_indices: np.ndarray | None
) -> np.ndarray:
    """Select item columns from one bounded row block."""
    block = x[start:stop]
    return block if item_indices is None else block[:, item_indices]


def _validate_applicable_mask(
    expected_shape: tuple[int, ...],
    applicable_mask: ArrayLike | None,
) -> np.ndarray | None:
    """Validate a respondent-specific item-applicability mask without selecting it."""
    if applicable_mask is None:
        return None

    try:
        mask = np.asarray(applicable_mask)
    except (TypeError, ValueError) as error:
        raise ValueError("applicable_mask must be a rectangular boolean matrix") from error
    if mask.dtype.kind != "b":
        raise ValueError("applicable_mask must contain boolean values")
    if mask.shape != expected_shape:
        raise ValueError(f"applicable_mask must have shape {expected_shape}, got {mask.shape}")
    return mask


def missing_rate(
    x: MatrixLike,
    item_indices: Sequence[int] | None = None,
    applicable_mask: ArrayLike | None = None,
) -> np.ndarray:
    """Calculate each respondent's proportion of missing item responses.

    Parameters:
    - x: A respondent × item response matrix.
    - item_indices: Optional 0-based subset of columns to evaluate. By default,
                    all item columns contribute equally.
    - applicable_mask: Optional Boolean matrix matching ``x``. True cells are
                       expected responses; False cells are excluded from both
                       the missing count and the applicable-item count.

    Returns:
    - A float array in ``[0, 1]``. Zero means a complete response row and one
      means every selected, applicable response is missing. Rows without any
      applicable selected items return ``NaN``.

    Raises:
    - ValueError: If the matrix, item selection, or applicability mask is invalid.

    Example:
        >>> import numpy as np
        >>> missing_rate([[1, np.nan, 3], [np.nan, np.nan, 2]])
        array([0.33333333, 0.66666667])
    """
    x_array = validate_matrix_input(x, dtype=float, check_type=False)
    selected_indices = _validate_item_indices(x_array.shape[1], item_indices)
    selected_columns = x_array.shape[1] if selected_indices is None else len(selected_indices)
    applicable = _validate_applicable_mask(x_array.shape, applicable_mask)
    result = np.empty(len(x_array), dtype=float)

    for start, stop in row_slices(len(x_array), selected_columns):
        selected = _select_item_block(x_array, start, stop, selected_indices)
        missing = np.isnan(selected)
        block_result = result[start:stop]

        if applicable is None:
            missing_counts = np.count_nonzero(missing, axis=1)
            np.divide(missing_counts, selected_columns, out=block_result)
            continue

        selected_applicable = _select_item_block(applicable, start, stop, selected_indices)
        missing &= selected_applicable
        missing_counts = np.count_nonzero(missing, axis=1)
        applicable_counts = np.count_nonzero(selected_applicable, axis=1)
        block_result.fill(np.nan)
        np.divide(
            missing_counts,
            applicable_counts,
            out=block_result,
            where=applicable_counts > 0,
        )
    return result


def missing_rate_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 95.0,
    item_indices: Sequence[int] | None = None,
    applicable_mask: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate missing-response rates and flag unusually incomplete rows.

    An explicit threshold flags rates at or above the cutoff. Without a fixed
    threshold, rates strictly above the requested sample percentile are flagged.

    Parameters:
    - x: A respondent × item response matrix.
    - threshold: Optional fixed rate in ``[0, 1]``.
    - percentile: Sample percentile in ``[0, 100]`` used when threshold is None.
    - item_indices: Optional 0-based subset of columns to evaluate.
    - applicable_mask: Optional Boolean matrix matching ``x``. False cells do
                       not contribute to respondent-specific missing rates.

    Returns:
    - Tuple of ``(rates, flags)`` aligned to respondent rows.
    """
    percentile = validate_percentile(percentile)
    threshold = validate_threshold(threshold)
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite rate between 0 and 1")

    scores = missing_rate(
        x,
        item_indices=item_indices,
        applicable_mask=applicable_mask,
    )
    flags = threshold_flags(
        scores,
        threshold=threshold,
        percentile=percentile,
        direction="high",
    )
    return scores, flags
