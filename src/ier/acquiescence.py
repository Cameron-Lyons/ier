"""
Acquiescence index for detecting response bias in survey data.

Acquiescence bias is the tendency for respondents to agree with items regardless
of content. This module extends the basic mean-response measure in response_pattern()
with scale normalization and balanced-pair mode for isolating pure acquiescence bias.

References:
- Paulhus, D. L. (1991). Measurement and control of response bias. In J. P. Robinson,
  P. R. Shaver, & L. S. Wrightsman (Eds.), Measures of personality and social
  psychological attitudes (pp. 17-59). Academic Press.
"""

import numpy as np

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

    In balanced-pair mode (with positive/negative items), isolates pure acquiescence
    bias by averaging (positive + reversed_negative) / 2 across pairs, then normalizing.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - scale_min: Minimum value of the response scale. If None, inferred from data.
    - scale_max: Maximum value of the response scale. If None, inferred from data.
    - positive_items: List of column indices (0-based) for positively-worded items.
    - negative_items: List of column indices (0-based) for negatively-worded items.
    - na_rm: Boolean indicating whether to ignore missing values during computation.

    Returns:
    - A numpy array of acquiescence scores in [0, 1] for each individual.
      Values near 0.5 indicate no acquiescence bias, values near 1.0 indicate
      strong agreement bias.

    Raises:
    - ValueError: If inputs are invalid or item indices are out of bounds.

    Example:
        >>> data = [[5, 5, 5, 5], [1, 1, 1, 1], [3, 3, 3, 3]]
        >>> scores = acquiescence(data, scale_min=1, scale_max=5)
        >>> print(scores)
        [1.0, 0.0, 0.5]
    """
    x_array = validate_matrix_input(x, check_type=False)

    if scale_min is None:
        scale_min = float(np.nanmin(x_array))
    if scale_max is None:
        scale_max = float(np.nanmax(x_array))

    scale_range = scale_max - scale_min
    if scale_range < 0:
        raise ValueError("scale_max must be greater than scale_min")
    if scale_range == 0:
        return np.full(x_array.shape[0], 0.5)

    has_positive = positive_items is not None
    has_negative = negative_items is not None

    if has_positive != has_negative:
        raise ValueError("must specify both positive_items and negative_items, or neither")

    if has_positive and has_negative:
        assert positive_items is not None
        assert negative_items is not None

        if len(positive_items) == 0 or len(negative_items) == 0:
            raise ValueError("positive_items and negative_items cannot be empty")

        n_cols = x_array.shape[1]
        for idx in positive_items + negative_items:
            if idx < 0 or idx >= n_cols:
                raise ValueError(f"item index {idx} out of bounds for data with {n_cols} columns")

        min_len = min(len(positive_items), len(negative_items))
        pos_responses = x_array[:, positive_items[:min_len]].astype(float)
        neg_responses = x_array[:, negative_items[:min_len]].astype(float)

        reversed_neg = (scale_max + scale_min) - neg_responses
        pair_means = (pos_responses + reversed_neg) / 2.0

        if na_rm:
            raw_scores: np.ndarray = np.nanmean(pair_means, axis=1)
        else:
            raw_scores = np.mean(pair_means, axis=1)
    else:
        if na_rm:
            raw_scores = np.nanmean(x_array.astype(float), axis=1)
        else:
            raw_scores = np.mean(x_array.astype(float), axis=1)

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
    - threshold: Absolute threshold above which to flag. If None, uses percentile.
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

    valid_scores = scores[~np.isnan(scores)]

    if threshold is None:
        if len(valid_scores) == 0:
            threshold = 0.0
        else:
            threshold = float(np.percentile(valid_scores, percentile))

    flags = np.zeros(len(scores), dtype=bool)
    valid_mask = ~np.isnan(scores)
    flags[valid_mask] = scores[valid_mask] > threshold

    return scores, flags
