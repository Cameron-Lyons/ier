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
from ier._validation import MatrixLike, validate_item_indices, validate_matrix_input


def _validate_applicable_mask(
    x: np.ndarray,
    applicable_mask: ArrayLike | None,
) -> np.ndarray | None:
    """Validate a respondent-specific item-applicability mask without selecting columns."""
    if applicable_mask is None:
        return None

    try:
        mask = np.asarray(applicable_mask)
    except (TypeError, ValueError) as error:
        raise ValueError("applicable_mask must be a rectangular boolean matrix") from error
    if mask.dtype.kind != "b":
        raise ValueError("applicable_mask must contain boolean values")
    if mask.shape != x.shape:
        raise ValueError(f"applicable_mask must have shape {x.shape}, got {mask.shape}")
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
    selected = (
        slice(None)
        if item_indices is None
        else validate_item_indices(item_indices, x_array.shape[1])
    )
    n_items = x_array.shape[1] if isinstance(selected, slice) else len(selected)
    applicable = _validate_applicable_mask(x_array, applicable_mask)
    result = np.full(len(x_array), np.nan)
    for start, stop in row_slices(len(x_array), n_items):
        missing = np.isnan(x_array[start:stop, selected])
        if applicable is None:
            result[start:stop] = np.count_nonzero(missing, axis=1) / n_items
        else:
            block_applicable = applicable[start:stop, selected]
            missing &= block_applicable
            applicable_counts = np.count_nonzero(block_applicable, axis=1)
            np.divide(
                np.count_nonzero(missing, axis=1),
                applicable_counts,
                out=result[start:stop],
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
