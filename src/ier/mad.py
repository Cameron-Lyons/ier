"""
Mean Absolute Difference (MAD) for detecting insufficient effort responding.

MAD calculates the average absolute difference between responses to positively and negatively
worded (reverse-coded) items. This helps detect respondents who fail to adjust their responses
for reverse-worded items, indicating inattentive or careless responding.

References:
- Huang, J. L., Curran, P. G., Keeney, J., Poposki, E. M., & DeShon, R. P. (2012).
  Detecting and deterring insufficient effort responding to surveys.
  Journal of Business and Psychology, 27(1), 99-114.
"""

import numpy as np

from ier._flagging import threshold_flags
from ier._pair_statistics import paired_mean_absolute_difference, resolve_scale_bounds
from ier._validation import MatrixLike, validate_matrix_input


def mad(
    x: MatrixLike,
    positive_items: list[int] | None = None,
    negative_items: list[int] | None = None,
    item_pairs: list[tuple[int, int]] | None = None,
    scale_max: float | None = None,
    na_rm: bool = True,
    *,
    scale_min: float | None = None,
) -> np.ndarray:
    """
    Calculate Mean Absolute Difference (MAD) for detecting careless responding.

    MAD measures the average absolute difference between responses to positively-worded
    and reverse-coded (negatively-worded) items. Higher MAD scores indicate greater
    inconsistency with item direction and may identify respondents who failed to
    attend to reverse-coded items.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
         Can be a 2D list or numpy array. Responses should NOT be pre-reversed.
    - positive_items: List of column indices (0-based) for positively-worded items.
    - negative_items: List of column indices (0-based) for negatively-worded items.
                     These items will be reverse-scored before comparison.
    - item_pairs: Alternative to positive/negative_items. List of (positive, negative)
                 index tuples representing paired items to compare.
    - scale_max: Maximum value of the response scale (required for reverse scoring).
                If None, inferred from max value in the data.
    - na_rm: Boolean indicating whether to ignore missing values during computation.
    - scale_min: Minimum value of the response scale. If None, inferred from data.

    Returns:
    - A numpy array of MAD scores for each individual. Higher scores suggest
      inconsistent responses to reverse-worded item pairs.

    Raises:
    - ValueError: If inputs are invalid or item indices are out of bounds.

    Example:
        >>> data = [[5, 1, 4, 2], [3, 3, 3, 3], [5, 2, 4, 1]]
        >>> mad_scores = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        >>> print(mad_scores)
        [0.0, 0.0, 1.0]
    """
    x_array = validate_matrix_input(x, check_type=False)
    n_cols = x_array.shape[1]

    if item_pairs is not None:
        if positive_items is not None or negative_items is not None:
            raise ValueError("cannot specify both item_pairs and positive/negative_items")
        positive_items = [p[0] for p in item_pairs]
        negative_items = [p[1] for p in item_pairs]

    if positive_items is None or negative_items is None:
        raise ValueError("must specify either item_pairs or both positive_items and negative_items")

    if len(positive_items) == 0 or len(negative_items) == 0:
        raise ValueError("positive_items and negative_items cannot be empty")

    for idx in positive_items + negative_items:
        if idx < 0 or idx >= n_cols:
            raise ValueError(f"item index {idx} out of bounds for data with {n_cols} columns")

    bounds = resolve_scale_bounds(
        x_array,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    if bounds is None:
        return np.full(len(x_array), np.nan)
    resolved_min, resolved_max = bounds

    return paired_mean_absolute_difference(
        x_array,
        np.asarray(positive_items, dtype=np.intp),
        np.asarray(negative_items, dtype=np.intp),
        right_reflection=resolved_min + resolved_max,
        ignore_nan=na_rm,
    )


def run_mad_index(
    x: MatrixLike,
    positive_items: list[int] | None,
    negative_items: list[int] | None,
    scale_min: float | None,
    scale_max: float | None,
    na_rm: bool,
) -> np.ndarray:
    """Compute MAD with shared validation for optional screen/composite configurations."""
    if positive_items is None or negative_items is None:
        raise ValueError(
            "mad_positive_items and mad_negative_items must be provided when using mad index"
        )

    return mad(
        x,
        positive_items=positive_items,
        negative_items=negative_items,
        scale_max=scale_max,
        na_rm=na_rm,
        scale_min=scale_min,
    )


def mad_flag(
    x: MatrixLike,
    positive_items: list[int] | None = None,
    negative_items: list[int] | None = None,
    item_pairs: list[tuple[int, int]] | None = None,
    scale_max: float | None = None,
    threshold: float | None = None,
    percentile: float = 95.0,
    na_rm: bool = True,
    *,
    scale_min: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate MAD scores and flag potential careless responders.

    High MAD scores indicate careless responding (not attending to item direction).

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - positive_items: List of column indices for positively-worded items.
    - negative_items: List of column indices for negatively-worded items.
    - item_pairs: Alternative list of (positive, negative) index tuples.
    - scale_max: Maximum value of the response scale.
    - threshold: Absolute MAD threshold at or above which to flag. If None, uses percentile.
    - percentile: Percentile cutoff for flagging (default 95th percentile).
    - na_rm: Boolean indicating whether to ignore missing values.
    - scale_min: Minimum value of the response scale. If None, inferred from data.

    Returns:
    - Tuple of (mad_scores, flags) where flags is True for suspected careless responders.

    Example:
        >>> data = [[5, 1, 4, 2], [5, 5, 5, 5], [5, 2, 4, 1]]
        >>> scores, flags = mad_flag(data, positive_items=[0, 2], negative_items=[1, 3])
        >>> print(flags)
        [False, True, False]
    """
    scores = mad(
        x,
        positive_items=positive_items,
        negative_items=negative_items,
        item_pairs=item_pairs,
        scale_max=scale_max,
        na_rm=na_rm,
        scale_min=scale_min,
    )

    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    return scores, flags
