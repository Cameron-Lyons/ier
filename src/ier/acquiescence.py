"""
Acquiescence index for detecting response bias in survey data.

Acquiescence bias is the tendency for respondents to agree with items regardless
of content. This module extends the basic mean-response measure in response_pattern()
with scale normalization and balanced-pair mode for isolating pure acquiescence bias.

References:
- Paulhus, D. L. (1991). Measurement and control of response bias. In J. P. Robinson,
  P. R. Shaver, & L. S. Wrightsman (Eds.), Measures of personality and social
  psychological attitudes (pp. 17-59). Academic Press.
- Hinz et al. (2007). The acquiescence effect in responding to a questionnaire.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2736523/
"""

import numpy as np

from ier._flagging import threshold_flags
from ier._pair_statistics import validate_paired_item_indices
from ier._row_statistics import row_mean, row_slices
from ier._validation import MatrixLike, validate_matrix_input


def acquiescence(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
    positive_items: list[int] | None = None,
    negative_items: list[int] | None = None,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Calculate acquiescence index for each respondent.

    In simple mode (no item lists), computes the normalized mean response per person
    on a [0, 1] scale where 0.5 indicates no bias.

    In balanced-pair mode (with positive/negative items), averages raw agreement
    responses across each pair, then normalizes. Negative items must NOT be
    reverse-scored: agreeing with both item polarities indicates acquiescence.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - scale_min: Minimum value of the response scale. If None, inferred from data.
    - scale_max: Maximum value of the response scale. If None, inferred from data.
    - positive_items: List of column indices (0-based) for positively-worded items.
                      Must be paired in order with ``negative_items``.
    - negative_items: Equally sized list of column indices (0-based) for
                      negatively-worded items.
    - na_rm: Boolean indicating whether to ignore missing values during computation.

    Returns:
    - A numpy array of acquiescence scores in [0, 1] for each individual.
      Values near 0.5 indicate no acquiescence bias, values near 1.0 indicate
      strong agreement bias.

    Raises:
    - ValueError: If inputs are invalid, paired lists differ in length, or item
                  indices are not integers within the matrix bounds.

    Example:
        >>> data = [[5, 5, 5, 5], [1, 1, 1, 1], [3, 3, 3, 3]]
        >>> scores = acquiescence(data, scale_min=1, scale_max=5)
        >>> print(scores)
        [1.0, 0.0, 0.5]
    """
    x_array = validate_matrix_input(x, check_type=False)

    has_positive = positive_items is not None
    has_negative = negative_items is not None

    if has_positive != has_negative:
        raise ValueError("must specify both positive_items and negative_items, or neither")

    positive_indices: np.ndarray | None = None
    negative_indices: np.ndarray | None = None
    if positive_items is not None and negative_items is not None:
        positive_indices, negative_indices = validate_paired_item_indices(
            positive_items,
            negative_items,
            x_array.shape[1],
            left_name="positive_items",
            right_name="negative_items",
        )

    if scale_min is None:
        scale_min = float(np.nanmin(x_array))
    if scale_max is None:
        scale_max = float(np.nanmax(x_array))

    scale_range = scale_max - scale_min
    if scale_range < 0:
        raise ValueError("scale_max must be greater than scale_min")
    if scale_range == 0:
        return np.full(x_array.shape[0], 0.5)

    if positive_indices is not None and negative_indices is not None:
        n_pairs = len(positive_indices)
        raw_scores = np.empty(len(x_array))
        for start, stop in row_slices(len(x_array), n_pairs):
            positive = np.asarray(x_array[start:stop, positive_indices], dtype=float)
            negative = np.asarray(
                x_array[start:stop, negative_indices],
                dtype=float,
            )
            np.add(positive, negative, out=positive)
            positive *= 0.5
            raw_scores[start:stop] = row_mean(positive, ignore_nan=na_rm)
    else:
        raw_scores = row_mean(x_array, ignore_nan=na_rm)

    normalized: np.ndarray = np.clip((raw_scores - scale_min) / scale_range, 0.0, 1.0)
    return normalized


def acquiescence_flag(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
    positive_items: list[int] | None = None,
    negative_items: list[int] | None = None,
    threshold: float | None = None,
    percentile: float = 95.0,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate acquiescence scores and flag potential biased responders.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - scale_min: Minimum value of the response scale.
    - scale_max: Maximum value of the response scale.
    - positive_items: List of column indices for positively-worded items.
    - negative_items: List of column indices for negatively-worded items.
    - threshold: Absolute threshold at or above which to flag. If None, uses percentile.
    - percentile: Percentile cutoff for flagging (default 95th percentile).
    - na_rm: Boolean indicating whether to ignore missing values.

    Returns:
    - Tuple of (acquiescence_scores, flags) where flags is True for suspected
      biased responders.

    Example:
        >>> data = [[5, 5, 5, 5], [3, 3, 3, 3], [1, 1, 1, 1]]
        >>> scores, flags = acquiescence_flag(data, scale_min=1, scale_max=5)
    """
    scores = acquiescence(
        x,
        scale_min=scale_min,
        scale_max=scale_max,
        positive_items=positive_items,
        negative_items=negative_items,
        na_rm=na_rm,
    )

    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="high")

    return scores, flags
