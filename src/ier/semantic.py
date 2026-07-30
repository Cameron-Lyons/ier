"""
Semantic synonym/antonym consistency for detecting careless responding.

Unlike psychometric synonyms which are data-driven, semantic synonyms/antonyms
are predefined based on item content (e.g., "I am happy" vs "I am sad").
"""

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input


def semantic_syn(
    x: MatrixLike,
    item_pairs: list[tuple[int, int]],
    anto: bool = False,
    *,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> np.ndarray:
    """
    Calculate semantic synonym/antonym consistency scores.

    Computes within-person correlations for predefined item pairs based on
    semantic content. For synonyms, consistent responders should show positive
    correlations. For antonyms, consistent responders should show negative
    correlations.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - item_pairs: List of (i, j) tuples specifying semantically related item pairs.
                  Indices are 0-based.
    - anto: If True, reverse-score the second item in each antonym pair before
            comparing it with the first. If False, compare synonym pairs directly.
    - scale_min: Minimum response-scale value used to reverse-score antonyms.
                 If None, inferred from the data.
    - scale_max: Maximum response-scale value used to reverse-score antonyms.
                 If None, inferred from the data.

    Returns:
    - A numpy array of consistency scores for each individual.
      Higher values indicate greater consistency for both synonyms and antonyms.

    Raises:
    - ValueError: If inputs are invalid or item_pairs is empty

    Example:
        >>> data = [[1, 2, 5, 4], [1, 1, 5, 5], [3, 1, 3, 5]]
        >>> pairs = [(0, 1), (2, 3)]  # semantic synonym pairs
        >>> scores = semantic_syn(data, pairs)
    """
    x_array = validate_matrix_input(x, min_columns=2)
    n_items = x_array.shape[1]

    if not item_pairs:
        raise ValueError("item_pairs cannot be empty")

    for i, j in item_pairs:
        if i < 0 or i >= n_items or j < 0 or j >= n_items:
            raise ValueError(f"item pair ({i}, {j}) contains invalid indices")
        if i == j:
            raise ValueError(f"item pair ({i}, {j}) contains duplicate indices")

    pairs_array = np.array(item_pairs)
    response_i = x_array[:, pairs_array[:, 0]].astype(float)
    response_j = x_array[:, pairs_array[:, 1]].astype(float)

    if anto:
        observed = x_array[~np.isnan(x_array)]
        if len(observed) == 0:
            return np.full(x_array.shape[0], np.nan, dtype=float)

        resolved_min = float(np.min(observed)) if scale_min is None else scale_min
        resolved_max = float(np.max(observed)) if scale_max is None else scale_max
        if resolved_max < resolved_min:
            raise ValueError("scale_max must be greater than or equal to scale_min")
        response_j = (resolved_min + resolved_max) - response_j

    pair_diffs = np.abs(response_i - response_j)
    invalid_mask = np.isnan(response_i) | np.isnan(response_j)
    pair_diffs[invalid_mask] = np.nan

    pair_mean_diffs = np.nanmean(pair_diffs, axis=1)
    row_std = np.nanstd(x_array, axis=1)
    scores = np.full(x_array.shape[0], np.nan, dtype=float)

    valid_rows = ~np.isnan(pair_mean_diffs)
    nonzero_std = valid_rows & (row_std > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores[nonzero_std] = 1 - pair_mean_diffs[nonzero_std] / row_std[nonzero_std]

    zero_std = valid_rows & (row_std == 0)
    if np.any(zero_std):
        scores[zero_std] = np.where(np.isclose(pair_mean_diffs[zero_std], 0.0), 1.0, -1.0)

    result: np.ndarray = np.clip(scores, -1, 1)
    return result


def semantic_ant(
    x: MatrixLike,
    item_pairs: list[tuple[int, int]],
    *,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> np.ndarray:
    """
    Calculate semantic antonym consistency scores.

    Convenience wrapper for semantic_syn with anto=True.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - item_pairs: List of (i, j) tuples specifying semantic antonym pairs.
    - scale_min: Minimum response-scale value. If None, inferred from the data.
    - scale_max: Maximum response-scale value. If None, inferred from the data.

    Returns:
    - A numpy array of consistency scores for each individual.

    Example:
        >>> data = [[1, 5, 2, 4], [1, 5, 1, 5], [3, 3, 3, 3]]
        >>> pairs = [(0, 1), (2, 3)]  # semantic antonym pairs (e.g., happy/sad)
        >>> scores = semantic_ant(data, pairs)
    """
    return semantic_syn(
        x,
        item_pairs,
        anto=True,
        scale_min=scale_min,
        scale_max=scale_max,
    )
