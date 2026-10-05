"""
Infrequency / bogus item scoring for detecting insufficient effort responding.

Counts the number of failed attention-check (bogus/infrequency) items per respondent.
These are items with known correct answers that attentive respondents should get right
(e.g., "Please select 'Strongly Agree' for this item"). Items with several correct
answers, such as either disagreement category for "I have been to every country in
the world", or self-reported diligence items, use inclusive acceptable ranges instead.

References:
- Huang, J. L., Curran, P. G., Keeney, J., Poposki, E. M., & DeShon, R. P. (2012).
  Detecting and deterring insufficient effort responding to surveys.
  Journal of Business and Psychology, 27(1), 99-114.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

import math
import numbers
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ier._flagging import validate_threshold
from ier._row_statistics import row_slices
from ier._validation import MatrixLike, validate_item_indices, validate_matrix_input
from ier.types import InfrequencyMissingPolicy

_MISSING_POLICIES = {"pass", "fail", "omit", "propagate"}
_RANGE_MESSAGE = "acceptable_ranges must contain (low, high) pairs of real numbers"


def _prepare_expected_responses(
    values: list[float], dtype: np.dtype[np.generic]
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate answers once and retain exact integer-category comparisons."""
    try:
        supplied = np.asarray(values)
        if supplied.dtype.kind == "c":
            raise ValueError("complex answers are not real response categories")
        with np.errstate(over="ignore", invalid="ignore"):
            expected = np.asarray(supplied, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("expected_responses must contain finite numeric values") from error
    if expected.ndim != 1 or not np.isfinite(expected).all():
        raise ValueError("expected_responses must contain finite numeric values")

    if dtype.kind not in "iub":
        comparison_dtype = np.result_type(np.empty((), dtype=dtype).real.dtype, float)
        expected = expected.astype(comparison_dtype, copy=False)
        # An integer answer that the comparison dtype cannot represent cannot
        # equal a response. Avoid accepting its rounded neighbour.
        for item, value in enumerate(values):
            if isinstance(value, (int, np.integer)):
                target = np.asarray([int(value)], dtype=comparison_dtype)[0]
                expected[item] = target if int(target) == int(value) else np.nan
        return expected, None

    if dtype.kind == "b":
        lower, upper = 0, 1
    else:
        limits = np.iinfo(dtype.name)
        lower, upper = limits.min, limits.max
    targets = np.zeros(len(expected), dtype=dtype)
    impossible = np.ones(len(expected), dtype=bool)
    for item, value in enumerate(values):
        # Preserve integer answers before float conversion, including unsigned
        # endpoints. Fractional and out-of-range answers cannot match a category.
        answer = int(value) if isinstance(value, (int, np.integer)) else float(expected[item])
        if lower <= answer <= upper and answer == int(answer):
            targets[item] = int(answer)
            impossible[item] = False
    return targets, impossible if np.any(impossible) else None


def _range_bound(value: object) -> int | float:
    """Return one range endpoint as an exact Python integer or a float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(_RANGE_MESSAGE)
    if isinstance(value, numbers.Integral):
        return int(value)
    bound = float(value)
    if math.isnan(bound):
        raise ValueError("acceptable_ranges bounds cannot be NaN")
    return bound


def _sequence_values(values: object) -> list[object]:
    """Return the items of a non-text iterable for range validation."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(_RANGE_MESSAGE)
    try:
        return list(values)
    except TypeError as error:
        raise ValueError(_RANGE_MESSAGE) from error


def _validate_acceptable_ranges(
    ranges: object, n_items: int
) -> list[tuple[int | float, int | float]]:
    """Validate one inclusive ``(low, high)`` range per attention-check item."""
    pairs = _sequence_values(ranges)
    if len(pairs) != n_items:
        raise ValueError(
            f"item_indices ({n_items}) and acceptable_ranges ({len(pairs)}) "
            "must have the same length"
        )
    bounds: list[tuple[int | float, int | float]] = []
    for pair in pairs:
        endpoints = _sequence_values(pair)
        if len(endpoints) != 2:
            raise ValueError(_RANGE_MESSAGE)
        low, high = _range_bound(endpoints[0]), _range_bound(endpoints[1])
        if low > high:
            raise ValueError("acceptable_ranges must satisfy low <= high")
        bounds.append((low, high))
    return bounds


def _float_range_bound(value: int | float, dtype: np.dtype[Any], *, lower: bool) -> object:
    """Round an integer endpoint inward so it accepts no unrepresentable neighbour."""
    if isinstance(value, float):
        return value
    largest = np.finfo(dtype).max
    if value > int(largest):
        return math.inf if lower else largest
    if value < -int(largest):
        return -largest if lower else -math.inf
    target = np.asarray(value, dtype=dtype)[()]
    if lower and int(target) < value:
        return np.nextafter(target, dtype.type(math.inf))
    if not lower and int(target) > value:
        return np.nextafter(target, dtype.type(-math.inf))
    return target


def _integer_range(
    low: int | float, high: int | float, minimum: int, maximum: int
) -> tuple[int, int] | None:
    """Return the clamped integer categories inside an inclusive range, or None."""
    if low == math.inf or high == -math.inf:
        return None
    # Integers satisfy x >= low exactly when x >= ceil(low), and x <= high
    # exactly when x <= floor(high).
    first = minimum if low == -math.inf else max(math.ceil(low), minimum)
    last = maximum if high == math.inf else min(math.floor(high), maximum)
    return (first, last) if first <= last else None


def _prepare_acceptable_ranges(
    bounds: list[tuple[int | float, int | float]], dtype: np.dtype[np.generic]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return per-item bounds that compare exactly with responses of ``dtype``."""
    if dtype.kind not in "iub":
        # Array operands keep NumPy 1.x from rounding float64 bounds to the
        # response dtype before comparing float16/float32 observations.
        comparison_dtype = np.result_type(dtype, float)
        lows = np.array(
            [_float_range_bound(low, comparison_dtype, lower=True) for low, _ in bounds],
            dtype=comparison_dtype,
        )
        highs = np.array(
            [_float_range_bound(high, comparison_dtype, lower=False) for _, high in bounds],
            dtype=comparison_dtype,
        )
        return lows, highs, None

    if dtype.kind == "b":
        minimum, maximum = 0, 1
    else:
        limits = np.iinfo(dtype.name)
        minimum, maximum = int(limits.min), int(limits.max)
    # Comparing in the response dtype keeps large integer categories exact.
    lows = np.zeros(len(bounds), dtype=dtype)
    highs = np.zeros(len(bounds), dtype=dtype)
    impossible = np.zeros(len(bounds), dtype=bool)
    for item, (low, high) in enumerate(bounds):
        categories = _integer_range(low, high, minimum, maximum)
        if categories is None:
            impossible[item] = True
        else:
            lows[item], highs[item] = categories
    return lows, highs, impossible if np.any(impossible) else None


def infrequency(
    x: MatrixLike,
    item_indices: list[int],
    expected_responses: list[float] | None = None,
    proportion: bool = False,
    missing: InfrequencyMissingPolicy = "pass",
    *,
    acceptable_ranges: Sequence[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Count failed attention-check items per respondent.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - item_indices: Column indices (0-based) of the attention-check items.
    - expected_responses: Expected correct response for each attention-check item.
    - proportion: If True, return proportion of failed items instead of count.
    - missing: Missing-response policy. ``"pass"`` preserves the legacy behavior
               of treating missing checks as correct; ``"fail"`` treats them as
               failures; ``"omit"`` excludes them from proportional denominators;
               and ``"propagate"`` returns ``NaN`` when any check is missing.
    - acceptable_ranges: Keyword-only alternative to ``expected_responses``: one
                         inclusive ``(low, high)`` range of correct answers per
                         item, with ``-inf`` or ``inf`` for an open end. Responses
                         outside the range fail. Supply exactly one of
                         ``expected_responses`` and ``acceptable_ranges``.

    Returns:
    - A numpy array of failure counts (or proportions) per respondent. Under
      ``missing="omit"``, rows without observed checks return ``NaN``.

    Raises:
    - ValueError: If the policy, item selection, expected responses, acceptable
                  ranges, or proportion control is invalid.

    Example:
        >>> data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        >>> scores = infrequency(data, item_indices=[0, 2], expected_responses=[5, 1])
        >>> print(scores)
        [0. 1. 2.]

        A bogus item may accept either disagreement category (1 or 2 on a 1-5
        scale), and a diligence item any self-rating of 4 or more:

        >>> ranges = [(1, 2), (4, float("inf"))]
        >>> infrequency([[2, 3, 5], [4, 3, 3]], [0, 2], acceptable_ranges=ranges).tolist()
        [0.0, 2.0]
    """
    x_array = validate_matrix_input(x)
    n_cols = x_array.shape[1]

    if not isinstance(proportion, bool):
        raise ValueError("proportion must be a boolean")
    if not isinstance(missing, str) or missing not in _MISSING_POLICIES:
        raise ValueError(f"missing must be one of: {sorted(_MISSING_POLICIES)}")
    if (expected_responses is None) == (acceptable_ranges is None):
        raise ValueError("provide exactly one of expected_responses or acceptable_ranges")
    if len(item_indices) == 0:
        raise ValueError("item_indices cannot be empty")

    bounds: list[tuple[int | float, int | float]] | None = None
    if acceptable_ranges is not None:
        bounds = _validate_acceptable_ranges(acceptable_ranges, len(item_indices))
    elif expected_responses is not None and len(item_indices) != len(expected_responses):
        raise ValueError(
            f"item_indices ({len(item_indices)}) and expected_responses "
            f"({len(expected_responses)}) must have the same length"
        )

    selected = validate_item_indices(item_indices, n_cols)

    if bounds is None:
        assert expected_responses is not None
        expected, impossible = _prepare_expected_responses(expected_responses, x_array.dtype)
    else:
        lows, highs, impossible = _prepare_acceptable_ranges(bounds, x_array.dtype)

    failures = np.empty(len(x_array), dtype=float)
    for start, stop in row_slices(len(x_array), len(selected)):
        block = x_array[start:stop, selected]
        if bounds is None:
            mismatch = block != expected
        else:
            # A missing response lies outside every range; the policy below
            # decides whether it counts.
            mismatch = block >= lows
            mismatch &= block <= highs
            np.logical_not(mismatch, out=mismatch)
        if impossible is not None:
            mismatch |= impossible
        observed = None if missing == "fail" or block.dtype.kind in "iub" else ~np.isnan(block)
        del block  # Release selected responses before allocating the next batch.
        if observed is not None:
            mismatch &= observed
        scores = failures[start:stop]
        np.sum(mismatch, axis=1, dtype=float, out=scores)
        if missing == "omit" and observed is not None:
            counts = np.count_nonzero(observed, axis=1)
            if proportion:
                np.divide(scores, counts, out=scores, where=counts > 0)
            scores[counts == 0] = np.nan
        elif proportion:
            scores /= len(selected)
        if missing == "propagate" and observed is not None:
            scores[~np.all(observed, axis=1)] = np.nan

    return failures


def infrequency_flag(
    x: MatrixLike,
    item_indices: list[int],
    expected_responses: list[float] | None = None,
    threshold: float = 1.0,
    proportion: bool = False,
    missing: InfrequencyMissingPolicy = "pass",
    *,
    acceptable_ranges: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count failed attention-check items and flag respondents exceeding a threshold.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - item_indices: Column indices (0-based) of the attention-check items.
    - expected_responses: Expected correct response for each attention-check item.
    - threshold: Failure count or proportion at or above which to flag (default 1).
    - proportion: If True, flag failure proportions instead of counts.
    - missing: Missing-response policy passed to ``infrequency()``.
    - acceptable_ranges: Keyword-only inclusive ``(low, high)`` ranges passed to
                         ``infrequency()`` instead of ``expected_responses``.

    Returns:
    - Tuple of (failure_scores, flags) where flags is True for flagged respondents.

    Example:
        >>> data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        >>> scores, flags = infrequency_flag(data, [0, 2], [5, 1], threshold=2)
        >>> print(flags)
        [False False  True]
    """
    if not isinstance(proportion, bool):
        raise ValueError("proportion must be a boolean")
    resolved_threshold = validate_threshold(threshold)
    assert resolved_threshold is not None
    if resolved_threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if proportion and resolved_threshold > 1:
        raise ValueError("proportion threshold must be between 0 and 1")

    scores = infrequency(
        x,
        item_indices,
        expected_responses,
        proportion=proportion,
        missing=missing,
        acceptable_ranges=acceptable_ranges,
    )
    flags = np.zeros(len(scores), dtype=bool)
    available = ~np.isnan(scores)
    flags[available] = scores[available] >= resolved_threshold
    return scores, flags
