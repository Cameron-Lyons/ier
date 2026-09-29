"""
Carelessness onset detection via changepoint analysis.

Detects the item index at which a respondent's behavior shifts from attentive
to careless responding, using running intra-individual response variability (IRV)
and the Shao & Zhang self-normalized cumulative sum changepoint test.

References:
- Shao, X., & Zhang, X. (2010). Testing for change points in time series.
  Journal of the American Statistical Association, 105(491), 1228-1240.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

from operator import index

import numpy as np

from ier._row_statistics import _integer_offsets, _integer_reduction_parameters, row_slices
from ier._validation import MatrixLike, validate_matrix_input

_SHAO_ZHANG_CRITICAL_VALUE = 1.358
_MIN_VARIANCE = 1e-10
_SAFE_RESPONSE_MAGNITUDE = 0.25 * np.sqrt(np.finfo(float).max * _MIN_VARIANCE)


def onset(
    x: MatrixLike,
    window_size: int = 10,
    min_items: int = 20,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Detect the item index at which carelessness begins for each respondent.

    Computes running IRV over sliding windows, then applies a self-normalized
    cumulative sum changepoint test to identify the transition point from
    attentive to careless responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - window_size: Integer sliding-window size for running IRV (default 10).
    - min_items: Integer minimum number of observed responses (default 20).
    - na_rm: If True, removes missing responses while preserving sequence order.
             If False, missing responses raise ValueError.

    Returns:
    - A numpy array of onset item indices per respondent. NaN if no changepoint
      is detected, the respondent has fewer than min_items valid responses, or
      fewer than three running windows are available. Positions are zero-based
      within the observed response sequence when missing responses are removed.
      Rows containing infinite responses remain unavailable.

    Raises:
    - ValueError: If sizes are not integers, window_size < 2, or min_items < window_size.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 15))
        >>> careless = np.full((1, 15), 3)
        >>> data = np.hstack([attentive, careless])
        >>> onset(data, window_size=5, min_items=10)
    """
    x_array = validate_matrix_input(x, check_type=False)

    window_size = _validate_item_count(window_size, name="window_size")
    min_items = _validate_item_count(min_items, name="min_items")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    if min_items < window_size:
        raise ValueError("min_items must be at least as large as window_size")

    required = max(min_items, window_size + 2)
    result = np.full(len(x_array), np.nan)
    if na_rm and x_array.shape[1] < required:
        return result

    for start, stop in row_slices(*x_array.shape):
        block = x_array[start:stop]
        if block.dtype.kind in "iub":
            if block.shape[1] >= required:
                result[start:stop] = _bounded_onsets(block, window_size, check_overflow=False)
            continue
        valid = ~np.isnan(block)
        complete = bool(np.all(valid))
        if not na_rm and not complete:
            raise ValueError("data contains missing values. Set na_rm=True to handle them")
        if block.shape[1] < required:
            continue
        largest = np.fmax.reduce(block, axis=None)
        smallest = np.fmin.reduce(block, axis=None)
        eligible = (
            ~np.isinf(block).any(axis=1)
            if np.isinf(largest) or np.isinf(smallest)
            else np.ones(len(block), dtype=bool)
        )
        safe = _SAFE_RESPONSE_MAGNITUDE / block.shape[1]
        check_overflow = bool(largest > safe or smallest < -safe)
        if complete:
            if np.all(eligible):
                result[start:stop] = _bounded_onsets(
                    block, window_size, check_overflow=check_overflow
                )
            elif np.any(eligible):
                result[start:stop][eligible] = _bounded_onsets(
                    block[eligible], window_size, check_overflow=check_overflow
                )
        else:
            counts = valid.sum(axis=1, dtype=np.intp)
            eligible &= counts >= required
            for raw_count in np.unique(counts[eligible]):
                matching = (counts == raw_count) & eligible
                selected = block[matching]
                packed = selected[valid[matching]].reshape(len(selected), int(raw_count))
                result[start:stop][matching] = _bounded_onsets(
                    packed, window_size, check_overflow=check_overflow
                )
    return result


def _validate_item_count(value: int, *, name: str) -> int:
    """Normalize Python/NumPy integers before size comparisons and negative slices."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _bounded_onsets(x: np.ndarray, window_size: int, *, check_overflow: bool) -> np.ndarray:
    """Bound the multiple rolling and cumulative-sum workspaces per respondent."""
    result = np.empty(len(x))
    for start, stop in row_slices(len(x), 4 * x.shape[1]):
        result[start:stop] = _complete_onsets(
            x[start:stop], window_size, check_overflow=check_overflow
        )
    return result


def _complete_onsets(x: np.ndarray, window_size: int, *, check_overflow: bool) -> np.ndarray:
    """Score finite rows, rescaling only those with unstable rolling moments."""
    if not check_overflow:
        running_irv = _running_inconsistency_complete(x, window_size)
        return _shao_zhang_changepoints(running_irv) + window_size - 1
    with np.errstate(over="ignore", invalid="ignore"):
        running_irv = _running_inconsistency_complete(x, window_size)
    largest = np.max(running_irv, axis=1)
    safe = 2 * _SAFE_RESPONSE_MAGNITUDE / running_irv.shape[1]
    unstable = ~np.isfinite(largest) | (largest > safe)
    if not np.any(unstable):
        return _shao_zhang_changepoints(running_irv) + window_size - 1

    result = np.full(len(x), np.nan)
    if not np.all(unstable):
        result[~unstable] = _shao_zhang_changepoints(running_irv[~unstable])
    scaled = np.asarray(x[unstable], dtype=float)
    _, exponents = np.frexp(np.max(np.abs(scaled), axis=1))
    np.ldexp(scaled, -exponents[:, None], out=scaled)
    running_irv = _running_inconsistency_complete(scaled, window_size)
    result[unstable] = _shao_zhang_changepoints(running_irv, log_scale=exponents * np.log(2.0))
    return result + window_size - 1


def onset_flag(
    x: MatrixLike,
    window_size: int = 10,
    min_items: int = 20,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Flag respondents for whom a carelessness onset was detected.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - window_size: Size of the sliding window for running IRV.
    - min_items: Minimum number of items required for onset detection.
    - na_rm: If True, handles missing values.

    Returns:
    - Boolean array where True indicates a carelessness onset was detected.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 15))
        >>> careless = np.full((1, 15), 3)
        >>> data = np.hstack([attentive, careless])
        >>> onset_flag(data, window_size=5, min_items=10)
    """
    onset_indices = onset(x, window_size=window_size, min_items=min_items, na_rm=na_rm)
    result: np.ndarray = ~np.isnan(onset_indices)
    return result


def _running_inconsistency_complete(x: np.ndarray, window_size: int) -> np.ndarray:
    """Compute complete-row window deviations in bounded rolling workspaces."""
    large, _ = _integer_reduction_parameters(x)
    centered = _integer_offsets(x) if large is not None else x.astype(float, copy=True)
    centered -= centered[:, :1]

    # Integer-valued responses permit exact cumulative moments when every sum,
    # square, and product stays within float64's exact integer range. This avoids
    # cancellation even in constant windows and eliminates the window-size loop.
    if window_size >= 16:
        limit = np.floor(np.sqrt(2**52 / (x.shape[1] * window_size)))
        if (
            np.max(centered) <= limit
            and np.min(centered) >= -limit
            and (x.dtype.kind in "iub" or np.all(centered == np.trunc(centered)))
        ):
            totals = _window_sums(centered, window_size)
            np.square(centered, out=centered)
            variances = _window_sums(centered, window_size)
            variances *= window_size
            np.square(totals, out=totals)
            variances -= totals
            variances /= window_size * window_size
            np.sqrt(variances, out=variances)
            return variances

    window_means = _window_sums(centered, window_size)
    window_means /= window_size

    squared_deviations = np.zeros(window_means.shape)
    scratch = np.empty(window_means.shape)
    for offset in range(window_size):
        np.subtract(
            centered[:, offset : offset + window_means.shape[1]],
            window_means,
            out=scratch,
        )
        np.square(scratch, out=scratch)
        squared_deviations += scratch

    squared_deviations /= window_size
    np.sqrt(squared_deviations, out=squared_deviations)
    return squared_deviations


def _window_sums(x: np.ndarray, window_size: int) -> np.ndarray:
    """Reduce complete rolling windows using one cumulative-sum workspace."""
    prefix = np.cumsum(x, axis=1)
    totals = prefix[:, window_size - 1 :].copy()
    if totals.shape[1] > 1:
        totals[:, 1:] -= prefix[:, :-window_size]
    return totals


def _shao_zhang_changepoints(
    series: np.ndarray, *, log_scale: np.ndarray | None = None
) -> np.ndarray:
    """Apply the changepoint test, consuming its internal series workspace."""
    n_rows, n_observations = series.shape
    result = np.full(n_rows, np.nan)
    if n_observations < 3:
        return result

    # Translation preserves the statistic while removing cancellation from a
    # common baseline, including exactly constant running-variability series.
    series -= series[:, :1]
    prefix_sum = np.cumsum(series, axis=1)
    np.square(series, out=series)
    prefix_square_sum = np.cumsum(series, axis=1)

    trim = max(1, n_observations // 10)
    candidate_positions = np.arange(trim, n_observations - trim)
    prefix_counts = candidate_positions.astype(float)
    prefix_values = prefix_sum[:, candidate_positions - 1]
    variances = prefix_square_sum[:, candidate_positions - 1]
    np.square(prefix_values, out=prefix_values)
    prefix_values /= prefix_counts
    variances -= prefix_values
    if log_scale is None:
        np.maximum(variances, _MIN_VARIANCE, out=variances)
    else:
        # Preserve the absolute variance floor in the original response units.
        # Logarithms retain candidate ordering even when ratios exceed float range.
        np.maximum(variances, 0.0, out=variances)
        with np.errstate(divide="ignore"):
            np.log(variances, out=variances)
        np.maximum(variances, np.log(_MIN_VARIANCE) - 2 * log_scale[:, None], out=variances)

    centered_candidates = prefix_sum[:, candidate_positions]
    prefix_values[:] = prefix_sum[:, -1, np.newaxis]
    prefix_values *= candidate_positions + 1
    prefix_values /= n_observations
    centered_candidates -= prefix_values
    if log_scale is None:
        np.square(centered_candidates, out=centered_candidates)
        centered_candidates /= variances
        threshold = _SHAO_ZHANG_CRITICAL_VALUE
    else:
        np.abs(centered_candidates, out=centered_candidates)
        with np.errstate(divide="ignore"):
            np.log(centered_candidates, out=centered_candidates)
        centered_candidates *= 2
        centered_candidates -= variances
        threshold = np.log(_SHAO_ZHANG_CRITICAL_VALUE)

    offsets = np.argmax(centered_candidates, axis=1)
    max_stats = np.take_along_axis(centered_candidates, offsets[:, None], axis=1)[:, 0]
    detected = max_stats > threshold
    result[detected] = (trim + offsets[detected]).astype(float)
    return result
