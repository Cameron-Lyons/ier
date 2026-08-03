"""
Semantic synonym/antonym consistency for detecting careless responding.

Unlike psychometric synonyms which are data-driven, semantic synonyms/antonyms
are predefined based on item content (e.g., "I am happy" vs "I am sad").
"""

import numpy as np

from ier._flagging import threshold_flags
from ier._pair_statistics import paired_mean_absolute_difference, resolve_scale_bounds
from ier._row_statistics import row_std
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

    Computes mean absolute differences for predefined item pairs and normalizes
    them by each person's response standard deviation. Synonyms are compared
    directly; antonyms reverse-score the second response before comparison.

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

    pairs_array = np.asarray(item_pairs, dtype=np.intp)
    reflection: float | None = None

    if anto:
        bounds = resolve_scale_bounds(
            x_array,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        if bounds is None:
            return np.full(x_array.shape[0], np.nan, dtype=float)
        resolved_min, resolved_max = bounds
        reflection = resolved_min + resolved_max

    pair_mean_diffs = paired_mean_absolute_difference(
        x_array,
        pairs_array[:, 0],
        pairs_array[:, 1],
        right_reflection=reflection,
        ignore_nan=True,
    )
    row_deviations = row_std(x_array, ignore_nan=True)
    scores = np.full(x_array.shape[0], np.nan, dtype=float)

    valid_rows = ~np.isnan(pair_mean_diffs)
    nonzero_std = valid_rows & (row_deviations > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores[nonzero_std] = 1 - pair_mean_diffs[nonzero_std] / row_deviations[nonzero_std]

    zero_std = valid_rows & (row_deviations == 0)
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


def semantic_syn_flag(
    x: MatrixLike,
    item_pairs: list[tuple[int, int]],
    threshold: float | None = None,
    percentile: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Score semantic synonym consistency and flag unusually low values."""
    scores = semantic_syn(x, item_pairs)
    flags = threshold_flags(
        scores,
        threshold=threshold,
        percentile=percentile,
        direction="low",
    )
    return scores, flags


def semantic_ant_flag(
    x: MatrixLike,
    item_pairs: list[tuple[int, int]],
    threshold: float | None = None,
    percentile: float = 5.0,
    *,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score semantic antonym consistency and flag unusually low values."""
    scores = semantic_ant(
        x,
        item_pairs,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    flags = threshold_flags(
        scores,
        threshold=threshold,
        percentile=percentile,
        direction="low",
    )
    return scores, flags
