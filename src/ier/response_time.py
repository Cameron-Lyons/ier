"""
Response time indices for detecting careless responding.

Extremely fast or unusually consistent response times may indicate
careless or inattentive responding.
"""

import warnings
from typing import Any

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input

SCIPY_AVAILABLE = False
_norm: Any = None
try:
    from scipy.stats import norm as _scipy_norm

    _norm = _scipy_norm
    SCIPY_AVAILABLE = True
except ImportError:
    pass  # scipy is optional; response_time_flagging falls back gracefully


def response_time(
    times: MatrixLike,
    metric: str = "median",
) -> np.ndarray:
    """
    Calculate response time summary statistics for each individual.

    Very low response times may indicate careless responding where
    participants rush through items without reading them.

    Parameters:
    - times: A matrix of response times where rows are individuals and
             columns are items. Times should be in consistent units (e.g., seconds).
    - metric: Summary statistic to compute. Options:
              "mean" - average response time per item
              "median" - median response time per item
              "sd" - standard deviation of response times
              "min" - minimum response time

    Returns:
    - A numpy array of response time statistics for each individual.

    Raises:
    - ValueError: If inputs are invalid or metric is unknown

    Example:
        >>> times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]
        >>> avg_times = response_time(times, metric="mean")
        >>> print(avg_times)  # Second person has suspiciously fast times
    """
    times_array = validate_matrix_input(times, min_columns=1)

    result: np.ndarray
    if metric == "mean":
        result = np.nanmean(times_array, axis=1)
    elif metric == "median":
        result = np.nanmedian(times_array, axis=1)
    elif metric == "sd":
        result = np.nanstd(times_array, axis=1)
    elif metric == "min":
        result = np.nanmin(times_array, axis=1)
    else:
        raise ValueError(f"unknown metric: {metric}. Use 'mean', 'median', 'sd', or 'min'")
    return result


def response_time_flag(
    times: MatrixLike,
    threshold: float | None = None,
    method: str = "median",
    cutoff_percentile: float = 5.0,
) -> np.ndarray:
    """
    Flag individuals with suspiciously fast response times.

    Parameters:
    - times: A matrix of response times.
    - threshold: Absolute threshold for flagging (in same units as times).
                 If None, uses cutoff_percentile to determine threshold.
    - method: Method for computing per-person response time ("mean" or "median").
    - cutoff_percentile: Percentile below which to flag (default 5th percentile).
                         Only used if threshold is None.

    Returns:
    - Boolean array where True indicates potentially careless responding.

    Example:
        >>> times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]
        >>> flags = response_time_flag(times, threshold=1.0)
    """
    person_times = response_time(times, metric=method)

    if threshold is None:
        valid_times = person_times[~np.isnan(person_times)]
        if len(valid_times) == 0:
            return np.full(len(person_times), False, dtype=bool)
        threshold = np.percentile(valid_times, cutoff_percentile)

    return person_times < threshold


def response_time_consistency(
    times: MatrixLike,
) -> np.ndarray:
    """
    Calculate response time consistency (coefficient of variation).

    Very low consistency (uniform times) may indicate "clicking through"
    behavior where the person isn't reading items.

    Parameters:
    - times: A matrix of response times.

    Returns:
    - A numpy array of coefficient of variation values for each individual.
      Lower values indicate more uniform (potentially suspicious) timing.

    Example:
        >>> times = [[2.1, 3.4, 2.8], [1.0, 1.0, 1.0], [2.5, 2.3, 2.7]]
        >>> cv = response_time_consistency(times)
        >>> print(cv)  # Second person has very consistent (suspicious) times
    """
    times_array = validate_matrix_input(times, min_columns=2)

    means = np.nanmean(times_array, axis=1)
    stds = np.nanstd(times_array, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cv: np.ndarray = stds / means

    return cv


def response_time_mixture(
    times: MatrixLike,
    n_components: int = 2,
    log_transform: bool = True,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Fit a Gaussian mixture model to per-person response times and return
    the posterior probability of belonging to the fast (careless) component.

    Computes per-person median response time, optionally log-transforms,
    then fits a k-component Gaussian mixture via EM. The component with
    the lowest mean is identified as the "fast" (careless) component.

    Parameters:
    - times: A matrix of response times where rows are individuals and columns
             are items.
    - n_components: Number of mixture components (default 2).
    - log_transform: If True (default), log-transform median times before fitting.
    - random_seed: Optional seed for reproducibility of EM initialization.

    Returns:
    - A numpy array of posterior probabilities of belonging to the fast component,
      one per respondent. Higher values indicate greater likelihood of careless
      (fast) responding.

    Raises:
    - RuntimeError: If scipy is not available.
    - ValueError: If n_components < 2 or data is insufficient.

    Example:
        >>> times = [[5.0, 6.0, 4.0], [0.5, 0.6, 0.4], [4.5, 5.5, 5.0]]
        >>> probs = response_time_mixture(times, random_seed=42)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy is required for response_time_mixture. Install with: pip install scipy"
        )

    if n_components < 2:
        raise ValueError("n_components must be at least 2")

    times_array = validate_matrix_input(times, min_columns=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(times_array, axis=1)

    valid_mask = ~np.isnan(medians) & (medians > 0)
    if np.sum(valid_mask) < n_components:
        raise ValueError(
            f"insufficient valid observations ({int(np.sum(valid_mask))}) "
            f"for {n_components} components"
        )

    data = medians[valid_mask].copy()

    if log_transform:
        data = np.log(data)

    rng = np.random.default_rng(random_seed)

    posteriors_valid = _em_gaussian_mixture(data, n_components, rng)

    result = np.full(len(medians), np.nan)
    result[valid_mask] = posteriors_valid

    return result


def _em_gaussian_mixture(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit k-component Gaussian mixture via EM; return posterior P(fast component)."""
    n = len(data)

    sorted_data = np.sort(data)
    split_points = np.array_split(sorted_data, k)
    means = np.array([np.mean(s) for s in split_points])
    variances = np.full(k, np.var(data) / k)
    variances = np.maximum(variances, 1e-10)
    weights = np.full(k, 1.0 / k)

    means += rng.normal(0, 0.01, size=k)

    resp = np.zeros((n, k))
    prev_ll = -np.inf

    for _ in range(max_iter):
        for j in range(k):
            resp[:, j] = weights[j] * _norm.pdf(data, loc=means[j], scale=np.sqrt(variances[j]))

        row_sums = resp.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        resp /= row_sums

        ll = np.sum(np.log(np.maximum(resp.sum(axis=1), 1e-300)))

        for j in range(k):
            nj = resp[:, j].sum()
            if nj < 1e-10:
                continue
            weights[j] = nj / n
            means[j] = (resp[:, j] @ data) / nj
            diff = data - means[j]
            variances[j] = (resp[:, j] @ (diff**2)) / nj
            variances[j] = max(variances[j], 1e-10)

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    for j in range(k):
        resp[:, j] = weights[j] * _norm.pdf(data, loc=means[j], scale=np.sqrt(variances[j]))
    row_sums = resp.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-300)
    resp /= row_sums

    fast_component = int(np.argmin(means))
    result: np.ndarray = resp[:, fast_component]
    return result
