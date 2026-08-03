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

import numpy as np

from ier._row_statistics import row_slices
from ier._validation import MatrixLike, validate_matrix_input

_SHAO_ZHANG_CRITICAL_VALUE = 1.358


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
    - window_size: Size of the sliding window for running IRV (default 10).
    - min_items: Minimum number of items required for onset detection (default 20).
    - na_rm: If True, handles missing values.

    Returns:
    - A numpy array of onset item indices per respondent. NaN if no changepoint
      is detected or if the respondent has fewer than min_items valid responses.

    Raises:
    - ValueError: If window_size < 2 or min_items < window_size.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(42)
        >>> attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 15))
        >>> careless = np.full((1, 15), 3)
        >>> data = np.hstack([attentive, careless])
        >>> onset(data, window_size=5, min_items=10)
    """
    x_array = validate_matrix_input(x, check_type=False)

    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    if min_items < window_size:
        raise ValueError("min_items must be at least as large as window_size")

    n_rows, n_items = x_array.shape
    has_missing = any(
        np.isnan(x_array[start:stop]).any() for start, stop in row_slices(n_rows, n_items)
    )
    if not na_rm and has_missing:
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    result = np.full(n_rows, np.nan)
    if n_items < min_items:
        return result

    if not has_missing:
        return _onset_complete(x_array, window_size)

    return _onset_missing(x_array, window_size, min_items)


def _onset_missing(
    x: np.ndarray,
    window_size: int,
    min_items: int,
) -> np.ndarray:
    """Compress and score missing-response rows in bounded equal-length groups."""
    n_rows, n_items = x.shape
    result = np.full(n_rows, np.nan)

    for start, stop in row_slices(n_rows, n_items):
        block = x[start:stop]
        valid = ~np.isnan(block)
        valid_counts = np.asarray(np.sum(valid, axis=1, dtype=np.intp))
        eligible_counts = np.unique(valid_counts[valid_counts >= min_items])

        for raw_count in eligible_counts:
            valid_count = int(raw_count)
            local_rows = np.flatnonzero(valid_counts == valid_count)
            selected = block[local_rows]
            compressed = selected[valid[local_rows]].reshape(len(local_rows), valid_count)
            result[start + local_rows] = _onset_complete(compressed, window_size)

    return result


def _onset_complete(x: np.ndarray, window_size: int) -> np.ndarray:
    """Detect onset for complete rows in bounded vectorized batches."""
    n_rows, n_items = x.shape
    n_windows = n_items - window_size + 1
    result = np.full(n_rows, np.nan)
    if n_windows < 3:
        return result

    for start, stop in row_slices(n_rows, n_items):
        running_irv = _running_inconsistency_complete(x[start:stop], window_size)
        changepoints = _shao_zhang_changepoints(running_irv)
        result[start:stop] = changepoints + window_size - 1

    return result


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
    centered = x.astype(float, copy=True)
    centered -= centered[:, :1]
    prefix_sum = np.cumsum(centered, axis=1)
    window_means = prefix_sum[:, window_size - 1 :].copy()
    if window_means.shape[1] > 1:
        window_means[:, 1:] -= prefix_sum[:, :-window_size]
    window_means /= window_size
    del prefix_sum

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


def _shao_zhang_changepoints(series: np.ndarray) -> np.ndarray:
    """Apply the changepoint test, consuming its internal series workspace."""
    n_rows, n_observations = series.shape
    result = np.full(n_rows, np.nan)
    if n_observations < 3:
        return result

    prefix_sum = np.cumsum(series, axis=1)
    np.square(series, out=series)
    prefix_square_sum = np.cumsum(series, axis=1)

    trim = max(1, n_observations // 10)
    candidate_positions = np.arange(trim, n_observations - trim)
    if len(candidate_positions) == 0:
        return result

    prefix_counts = candidate_positions.astype(float)
    prefix_values = prefix_sum[:, candidate_positions - 1]
    variances = prefix_square_sum[:, candidate_positions - 1]
    np.square(prefix_values, out=prefix_values)
    prefix_values /= prefix_counts
    variances -= prefix_values
    np.maximum(variances, 1e-10, out=variances)

    centered_candidates = prefix_sum[:, candidate_positions]
    prefix_values[:] = prefix_sum[:, -1, np.newaxis]
    prefix_values *= candidate_positions + 1
    prefix_values /= n_observations
    centered_candidates -= prefix_values
    np.square(centered_candidates, out=centered_candidates)
    centered_candidates /= variances

    offsets = np.argmax(centered_candidates, axis=1)
    max_stats = np.take_along_axis(centered_candidates, offsets[:, None], axis=1)[:, 0]
    detected = max_stats > _SHAO_ZHANG_CRITICAL_VALUE
    result[detected] = (trim + offsets[detected]).astype(float)
    return result
