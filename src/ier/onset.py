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

from ier._validation import MatrixLike, validate_matrix_input

_ONSET_BATCH_WORKSPACE_BYTES = 64 * 1024 * 1024
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

    has_missing = bool(np.isnan(x_array).any())
    if not na_rm and has_missing:
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    n_rows = x_array.shape[0]
    result = np.full(n_rows, np.nan)
    if x_array.shape[1] < min_items:
        return result

    if not has_missing:
        return _onset_complete(x_array, window_size)

    for i in range(n_rows):
        row = x_array[i, :]
        if na_rm:
            row = row[~np.isnan(row)]

        if len(row) < min_items:
            continue

        running_irv = _running_inconsistency(row, window_size)

        if len(running_irv) < 3:
            continue

        cp = _shao_zhang_changepoint(running_irv)
        if cp is not None:
            result[i] = float(cp + window_size - 1)

    return result


def _onset_complete(x: np.ndarray, window_size: int) -> np.ndarray:
    """Detect onset for complete rows in bounded vectorized batches."""
    n_rows, n_items = x.shape
    n_windows = n_items - window_size + 1
    result = np.full(n_rows, np.nan)
    if n_windows < 3:
        return result

    # ``np.std`` may materialize a window-sized temporary in addition to the
    # prefix/candidate arrays, so include both in the row-size estimate.
    bytes_per_row = np.dtype(float).itemsize * n_windows * (window_size + 6)
    batch_rows = max(1, _ONSET_BATCH_WORKSPACE_BYTES // bytes_per_row)

    for start in range(0, n_rows, batch_rows):
        stop = min(start + batch_rows, n_rows)
        windows = np.lib.stride_tricks.sliding_window_view(
            x[start:stop],
            window_size,
            axis=1,
        )
        running_irv = np.std(windows, axis=2)
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


def _running_inconsistency(row: np.ndarray, window_size: int) -> np.ndarray:
    """Compute running standard deviation over sliding windows."""
    if len(row) < window_size:
        return np.array([])

    windows = np.lib.stride_tricks.sliding_window_view(row, window_size)
    result: np.ndarray = np.std(windows, axis=1)
    return result


def _shao_zhang_changepoint(series: np.ndarray) -> int | None:
    """
    Apply the Shao & Zhang self-normalized cumulative sum test to detect a changepoint.

    Returns the index of the changepoint, or None if the test statistic does not
    exceed the critical value.
    """
    n = len(series)
    if n < 3:
        return None

    changepoint = _shao_zhang_changepoints(series[None, :])[0]
    return None if np.isnan(changepoint) else int(changepoint)


def _shao_zhang_changepoints(series: np.ndarray) -> np.ndarray:
    """Apply the changepoint test to a complete batch of running IRV rows."""
    n_rows, n_observations = series.shape
    result = np.full(n_rows, np.nan)
    if n_observations < 3:
        return result

    prefix_sum = np.cumsum(series, axis=1)
    prefix_square_sum = np.cumsum(series * series, axis=1)
    centered_prefix = np.cumsum(series - np.mean(series, axis=1, keepdims=True), axis=1)

    trim = max(1, n_observations // 10)
    candidate_positions = np.arange(trim, n_observations - trim)
    if len(candidate_positions) == 0:
        return result

    prefix_positions = candidate_positions - 1
    prefix_counts = candidate_positions.astype(float)
    variances = (
        prefix_square_sum[:, prefix_positions]
        - prefix_sum[:, prefix_positions] ** 2 / prefix_counts
    )
    variances = np.maximum(variances, 1e-10)
    test_stats = centered_prefix[:, candidate_positions] ** 2 / variances

    offsets = np.argmax(test_stats, axis=1)
    max_stats = np.take_along_axis(test_stats, offsets[:, None], axis=1)[:, 0]
    detected = max_stats > _SHAO_ZHANG_CRITICAL_VALUE
    result[detected] = (trim + offsets[detected]).astype(float)
    return result
