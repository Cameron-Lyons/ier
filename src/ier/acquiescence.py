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

import math
from decimal import Decimal, localcontext

import numpy as np

from ier._flagging import threshold_flags
from ier._pair_statistics import validate_paired_item_indices
from ier._row_statistics import row_mean, row_slices
from ier._validation import MatrixLike, resolve_scale_bounds, validate_matrix_input


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
      strong agreement bias. Unavailable respondent means remain ``NaN``,
      including when equal scale bounds give observed respondents a score of 0.5.

    Raises:
    - ValueError: If inputs are invalid, paired lists differ in length, or item
                  indices are not integers within the matrix bounds.

    Example:
        >>> data = [[5, 5, 5, 5], [1, 1, 1, 1], [3, 3, 3, 3]]
        >>> scores = acquiescence(data, scale_min=1, scale_max=5)
        >>> print(scores.tolist())
        [1.0, 0.0, 0.5]
    """
    x_array = validate_matrix_input(x)

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

    bounds = resolve_scale_bounds(x_array, scale_min=scale_min, scale_max=scale_max)
    if bounds is None:
        return np.full(len(x_array), np.nan)
    scale_min, scale_max = bounds
    scale_range = scale_max - scale_min
    normalize_first = (
        scale_range != 0
        and math.isfinite(scale_min)
        and math.isfinite(scale_max)
        and (
            not math.isfinite(scale_range)
            or scale_range < np.finfo(float).tiny
            or abs(scale_min) * math.sqrt(np.finfo(float).eps) > scale_range
        )
    )

    if positive_indices is not None and negative_indices is not None:
        n_pairs = len(positive_indices)
        paired_indices = np.concatenate((positive_indices, negative_indices))
        raw_scores = np.empty(len(x_array))
        for start, stop in row_slices(len(x_array), 2 * n_pairs):
            values = x_array[start:stop, paired_indices]
            if not normalize_first:
                left = np.asarray(values[:, :n_pairs], dtype=float)
                right = np.asarray(values[:, n_pairs:], dtype=float)
                with np.errstate(over="raise", invalid="ignore", under="ignore"):
                    try:
                        np.add(left, right, out=left)
                    except FloatingPointError:
                        # Restore values before averaging the complete endpoints.
                        values = x_array[start:stop, paired_indices]
                    else:
                        left *= 0.5
                        raw_scores[start:stop] = row_mean(left, ignore_nan=na_rm)
                        continue
            if na_rm:
                missing = np.isnan(values[:, :n_pairs]) | np.isnan(values[:, n_pairs:])
                if np.any(missing):
                    np.copyto(values[:, :n_pairs], np.nan, where=missing)
                    np.copyto(values[:, n_pairs:], np.nan, where=missing)
            # Average both endpoints directly; individual pair sums may overflow.
            raw_scores[start:stop] = (
                _normalized_row_mean(values, scale_min, scale_max, ignore_nan=na_rm)
                if normalize_first
                else row_mean(values, ignore_nan=na_rm)
            )
    elif normalize_first:
        raw_scores = np.empty(len(x_array))
        for start, stop in row_slices(len(x_array), x_array.shape[1]):
            raw_scores[start:stop] = _normalized_row_mean(
                x_array[start:stop], scale_min, scale_max, ignore_nan=na_rm
            )
    else:
        raw_scores = row_mean(x_array, ignore_nan=na_rm)

    if scale_range == 0:
        return np.where(np.isnan(raw_scores), np.nan, 0.5)

    if not normalize_first:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            raw_scores -= scale_min
            raw_scores /= scale_range
    np.clip(raw_scores, 0.0, 1.0, out=raw_scores)
    return raw_scores


def _normalized_row_mean(
    values: np.ndarray, lower: float, upper: float, *, ignore_nan: bool
) -> np.ndarray:
    """Move exceptional scales to response proportions before reducing a block."""
    width = upper - lower
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        if math.isinf(width) and math.isfinite(lower) and math.isfinite(upper):
            offsets = np.asarray(values, dtype=float) * 0.5 - lower * 0.5
            width = upper * 0.5 - lower * 0.5
        elif values.dtype.kind in "iu":
            # Subtract the exact endpoint before converting adjacent large integers.
            origin = int(lower) if float(lower).is_integer() else lower
            offsets = np.asarray(values.astype(object) - origin, dtype=float)
        else:
            offsets = np.asarray(values, dtype=float) - lower
        offsets /= width
    means = row_mean(offsets, ignore_nan=ignore_nan)
    overflowed = np.any(np.isinf(offsets) & np.isfinite(values), axis=1)
    if np.any(overflowed):
        # Explicit very narrow bounds can make finite out-of-range responses
        # unrepresentable as proportions, even when their normalized mean is small.
        with localcontext() as context:
            context.prec = 800
            lo, hi = Decimal(lower), Decimal(upper)
            for row in np.flatnonzero(overflowed):
                observed = values[row]
                if ignore_nan:
                    observed = observed[~np.isnan(observed)]
                if not len(observed) or not np.isfinite(observed).all():
                    continue
                exact = [
                    Decimal(int(value))
                    if values.dtype.kind in "iu"
                    else Decimal.from_float(float(value))
                    for value in observed
                ]
                mean = (sum(exact, start=Decimal(0)) / len(exact) - lo) / (hi - lo)
                means[row] = float(min(Decimal(1), max(Decimal(0), mean)))
    return means


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
