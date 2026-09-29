"""
U3 polytomous index for detecting unusual response patterns.

The U3 polytomous index measures the proportion of extreme responses,
which can indicate careless responding patterns like extreme response style
or random clicking at scale endpoints.
"""

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from ier._row_statistics import _row_mean_std_counts_block, row_slices
from ier._validation import MatrixLike, resolve_scale_bounds, validate_matrix_input


def _endpoint_matcher(
    dtype: np.dtype[np.generic], lower: float, upper: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Prepare endpoint comparisons without rounding responses to scalar bounds."""
    endpoints: list[int | np.float64] = []
    for value in (lower, upper):
        if dtype.kind in "iu":
            limits = np.iinfo(dtype.name)
            if (
                not math.isfinite(value)
                or value != math.floor(value)
                or not limits.min <= value <= limits.max
            ):
                continue
            endpoint: int | np.float64 = int(value)
        else:
            # A double scalar also prevents weak-scalar casting to float32.
            endpoint = np.float64(value)
            if isinstance(value, int) and float(endpoint) != value:
                continue
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    if not endpoints:
        return lambda block: np.zeros(block.shape, dtype=bool)
    # One-element arrays promote consistently on both NumPy 1.x and 2.x.
    targets = np.asarray(endpoints, dtype=dtype if dtype.kind in "iu" else float)
    if len(endpoints) == 1:
        return lambda block: block == targets[:1]
    return lambda block: (block == targets[:1]) | (block == targets[1:])


def _midpoint_matcher(
    dtype: np.dtype[np.generic], lower: float, upper: float, tolerance: float = 0.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve midpoint limits once, preserving exact integer comparisons."""
    if dtype.kind in "iu":
        if not math.isfinite(lower) or not math.isfinite(upper):
            return lambda block: np.zeros(block.shape, dtype=bool)
        # Fraction preserves both integer endpoints and fractional tolerances;
        # only integer observations inside this closed interval can match.
        midpoint = (Fraction(lower) + Fraction(upper)) / 2
        radius = Fraction(tolerance.item() if isinstance(tolerance, np.generic) else tolerance)
        limits = np.iinfo(dtype.name)
        first = max(limits.min, math.ceil(midpoint - radius))
        last = min(limits.max, math.floor(midpoint + radius))
        if first > last:
            return lambda block: np.zeros(block.shape, dtype=bool)
        if first == last:
            return lambda block: block == first
        return lambda block: (block >= first) & (block <= last)

    # Add before halving to retain subnormal values; halve first only on overflow.
    total = lower + upper
    midpoint_value = (
        lower / 2 + upper / 2
        if math.isinf(total) and math.isfinite(lower) and math.isfinite(upper)
        else total / 2
    )
    first_float = np.array([midpoint_value - float(tolerance)])
    last_float = np.array([midpoint_value + float(tolerance)])
    if tolerance == 0:
        return lambda block: block == first_float
    return lambda block: (block >= first_float) & (block <= last_float)


def _proportion(matches: np.ndarray, valid_counts: np.ndarray) -> np.ndarray:
    """Normalize matching counts, leaving empty respondents unavailable."""
    result: np.ndarray = np.divide(
        np.sum(matches, axis=1, dtype=np.intp),
        valid_counts,
        out=np.full(len(matches), np.nan),
        where=valid_counts > 0,
    )
    return result


def _valid_proportion(
    x_array: np.ndarray,
    matcher: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Calculate matching-response proportions in bounded row batches."""
    scores = np.empty(len(x_array))
    for start, stop in row_slices(len(x_array), x_array.shape[1]):
        block = x_array[start:stop]
        valid_counts: np.ndarray = (
            np.full(len(block), block.shape[1], dtype=np.intp)
            if block.dtype.kind in "iub"
            else (~np.isnan(block)).sum(axis=1, dtype=np.intp)
        )
        scores[start:stop] = _proportion(matcher(block), valid_counts)
    return scores


def u3_poly(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> np.ndarray:
    """
    Calculate U3 polytomous index for each individual.

    The U3 index measures the proportion of responses at the extreme ends
    of the response scale. High values may indicate extreme response style
    or careless responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - scale_min: Minimum value of the response scale. If None, inferred from data.
    - scale_max: Maximum value of the response scale. If None, inferred from data.

    Returns:
    - A numpy array of U3 values (proportion of extreme responses) for each individual.
      Values range from 0 to 1. Rows without observed responses or unavailable
      scale bounds receive ``NaN``.

    Raises:
    - ValueError: If inputs are invalid

    Example:
        >>> data = [[1, 1, 5, 5, 3], [3, 3, 3, 3, 3], [1, 5, 1, 5, 1]]
        >>> u3 = u3_poly(data, scale_min=1, scale_max=5)
        >>> print(u3)  # Third person has highest extreme responding
    """
    x_array = validate_matrix_input(x, min_columns=1)

    bounds = resolve_scale_bounds(x_array, scale_min=scale_min, scale_max=scale_max)
    if bounds is None:
        return np.full(len(x_array), np.nan)
    scale_min, scale_max = bounds

    return _valid_proportion(
        x_array,
        _endpoint_matcher(x_array.dtype, scale_min, scale_max),
    )


def midpoint_responding(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
    tolerance: float = 0.0,
) -> np.ndarray:
    """
    Calculate proportion of midpoint responses for each individual.

    Excessive midpoint responding may indicate satisficing or
    inattentive responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - scale_min: Minimum value of the response scale.
    - scale_max: Maximum value of the response scale.
    - tolerance: Finite, nonnegative range around midpoint to count as midpoint response.

    Returns:
    - A numpy array of midpoint response proportions. Rows without observed
      responses or unavailable scale bounds receive ``NaN``.

    Raises:
    - ValueError: If scale bounds are inverted or tolerance is negative or nonfinite.

    Example:
        >>> data = [[1, 2, 5, 4, 3], [3, 3, 3, 3, 3], [1, 5, 1, 5, 1]]
        >>> mid = midpoint_responding(data, scale_min=1, scale_max=5)
        >>> print(mid)  # Second person has all midpoint responses
    """
    x_array = validate_matrix_input(x, min_columns=1)

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    bounds = resolve_scale_bounds(x_array, scale_min=scale_min, scale_max=scale_max)
    if bounds is None:
        return np.full(len(x_array), np.nan)
    scale_min, scale_max = bounds

    return _valid_proportion(
        x_array,
        _midpoint_matcher(x_array.dtype, scale_min, scale_max, tolerance),
    )


def response_pattern(
    x: MatrixLike,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate multiple response pattern indices.

    Returns a dictionary with various response style indicators.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are items.
    - scale_min: Minimum value of the response scale.
    - scale_max: Maximum value of the response scale.

    Returns:
    - Dictionary with:
        - "extreme": proportion of extreme responses (U3)
        - "midpoint": proportion of midpoint responses
        - "acquiescence": mean response (higher = more agreement bias)
        - "variability": response variability (SD)

    Example:
        >>> data = [[1, 2, 5, 4, 3], [3, 3, 3, 3, 3], [5, 5, 5, 5, 5]]
        >>> patterns = response_pattern(data, scale_min=1, scale_max=5)
    """
    x_array = validate_matrix_input(x, min_columns=1)

    bounds = resolve_scale_bounds(x_array, scale_min=scale_min, scale_max=scale_max)
    scores = {
        name: np.full(len(x_array), np.nan)
        for name in ("extreme", "midpoint", "acquiescence", "variability")
    }
    if bounds is not None:
        extreme_matches = _endpoint_matcher(x_array.dtype, *bounds)
        midpoint_matches = _midpoint_matcher(x_array.dtype, *bounds)
    # Four retained outputs leave less room for temporaries than a single reduction.
    for start, stop in row_slices(len(x_array), 3 * x_array.shape[1]):
        block = x_array[start:stop]
        means, deviations, counts = _row_mean_std_counts_block(block)
        scores["acquiescence"][start:stop] = means
        scores["variability"][start:stop] = deviations
        if bounds is not None:
            scores["extreme"][start:stop] = _proportion(extreme_matches(block), counts)
            scores["midpoint"][start:stop] = _proportion(midpoint_matches(block), counts)
    return scores
