"""
Semantic synonym/antonym consistency for detecting careless responding.

Unlike psychometric synonyms which are data-driven, semantic synonyms/antonyms
are predefined based on item content (e.g., "I am happy" vs "I am sad").
"""

from fractions import Fraction

import numpy as np

from ier._flagging import threshold_flags
from ier._pair_statistics import paired_mean_absolute_difference, validate_item_pairs
from ier._row_statistics import (
    _row_mean_std_block,
    _scaled_subnormal_moment_rows,
    row_slices,
)
from ier._validation import MatrixLike, resolve_scale_bounds, validate_matrix_input


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
    - ValueError: If item_pairs is empty, a pair does not contain two distinct
                  integer column indices, or an index is outside the matrix bounds.

    Example:
        >>> data = [[1, 2, 5, 4], [1, 1, 5, 5], [3, 1, 3, 5]]
        >>> pairs = [(0, 1), (2, 3)]  # semantic synonym pairs
        >>> scores = semantic_syn(data, pairs)
    """
    x_array = validate_matrix_input(x, min_columns=2)
    left_indices, right_indices = validate_item_pairs(item_pairs, x_array.shape[1])
    if np.any(left_indices == right_indices):
        raise ValueError("item pairs cannot contain duplicate indices within a pair")
    bounds: tuple[float, float] | None = None

    if anto:
        bounds = resolve_scale_bounds(
            x_array,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        if bounds is None:
            return np.full(x_array.shape[0], np.nan, dtype=float)

    scores = np.full(x_array.shape[0], np.nan, dtype=float)
    for start, stop in row_slices(len(x_array), max(x_array.shape[1], len(left_indices))):
        block = x_array[start:stop]
        row_means, row_deviations = _row_mean_std_block(block, ignore_nan=True)
        scaled = _scaled_subnormal_moment_rows(block, row_means, row_deviations)
        if bounds is not None and not np.isfinite(bounds).all():
            scaled = None
        regular_block, regular_deviations = block, row_deviations
        if scaled is not None:
            positions, values, exponents = scaled
            regular = np.ones(len(block), dtype=bool)
            regular[positions] = False
            regular_block, regular_deviations = block[regular], row_deviations[regular]
        regular_differences = (
            paired_mean_absolute_difference(
                regular_block,
                left_indices,
                right_indices,
                right_bounds=bounds,
                ignore_nan=True,
                normalizers=np.where(regular_deviations > 0, regular_deviations, 1.0),
            )
            if len(regular_block)
            else np.empty(0)
        )
        differences = regular_differences
        if scaled is not None:
            differences = np.full(len(block), np.nan)
            differences[regular] = regular_differences
        block_scores = scores[start:stop]
        valid_rows = ~np.isnan(differences)
        nonzero_std = valid_rows & (row_deviations > 0)
        block_scores[nonzero_std] = 1 - differences[nonzero_std]

        zero_std = valid_rows & (row_deviations == 0)
        if np.any(zero_std):
            block_scores[zero_std] = np.where(np.isclose(differences[zero_std], 0.0), 1.0, -1.0)

        if scaled is not None:
            positions, values, exponents = scaled
            for exponent in np.unique(exponents):
                selected = exponents == exponent
                scaled_bounds = (
                    None if bounds is None else _rescaled_antonym_bounds(bounds, int(exponent))
                )
                if bounds is not None and scaled_bounds is None:
                    # The finite reflection is too far from these observations
                    # to fit float64 after scaling, so every observed pair is inconsistent.
                    paired = values[selected][:, left_indices] + values[selected][:, right_indices]
                    block_scores[positions[selected]] = np.where(
                        np.any(~np.isnan(paired), axis=1), -1.0, np.nan
                    )
                    continue
                scaled_values = values[selected]
                _, scaled_deviations = _row_mean_std_block(scaled_values, ignore_nan=True)
                scaled_differences = paired_mean_absolute_difference(
                    scaled_values,
                    left_indices,
                    right_indices,
                    right_bounds=scaled_bounds,
                    ignore_nan=True,
                    normalizers=scaled_deviations,
                )
                block_scores[positions[selected]] = 1 - scaled_differences

    np.clip(scores, -1, 1, out=scores)
    return scores


def _rescaled_antonym_bounds(
    bounds: tuple[float, float], exponent: int
) -> tuple[float, float] | None:
    """Scale reflection endpoints, preserving cancellation if either overflows."""
    with np.errstate(over="ignore", under="ignore"):
        scaled = np.ldexp(np.asarray(bounds), -exponent)
    if scaled.dtype.itemsize <= 8 and np.isfinite(scaled).all():
        return float(scaled[0]), float(scaled[1])
    reflection = Fraction(*bounds[0].as_integer_ratio()) + Fraction(*bounds[1].as_integer_ratio())
    reflection = reflection / 2**exponent if exponent >= 0 else reflection * 2 ** (-exponent)
    try:
        rounded = float(reflection)
    except OverflowError:
        return None
    return rounded, float(reflection - Fraction(rounded))


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
