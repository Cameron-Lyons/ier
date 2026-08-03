"""
Response time indices for detecting careless responding.

Extremely fast or unusually consistent response times may indicate
careless or inattentive responding.
"""

import math
import warnings

import numpy as np

from ier._flagging import threshold_flags
from ier._validation import MatrixLike, validate_matrix_input

_LOG_TWO_PI = math.log(2.0 * math.pi)
_MIN_COMPONENT_MASS = 1e-10
_MIN_VARIANCE = 1e-10


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
    - threshold: Absolute threshold at or below which to flag (in the same units as times).
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

    return threshold_flags(
        person_times,
        threshold=threshold,
        percentile=cutoff_percentile,
        direction="low",
    )


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
    - ValueError: If n_components < 2 or data is insufficient.

    Example:
        >>> times = [[5.0, 6.0, 4.0], [0.5, 0.6, 0.4], [4.5, 5.5, 5.0]]
        >>> probs = response_time_mixture(times, random_seed=42)
    """
    if n_components < 2:
        raise ValueError("n_components must be at least 2")

    times_array = validate_matrix_input(times, min_columns=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(times_array, axis=1)

    valid_mask = np.isfinite(medians) & (medians > 0)
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
    variances = np.maximum(variances, _MIN_VARIANCE)
    weights = np.full(k, 1.0 / k)

    means += rng.normal(0, 0.01, size=k)

    resp = np.empty((n, k))
    scratch = np.empty(n)
    prev_ll = -np.inf

    for _ in range(max_iter):
        ll = _mixture_expectation(data, weights, means, variances, resp, scratch)

        for j in range(k):
            nj = resp[:, j].sum()
            if nj < _MIN_COMPONENT_MASS:
                continue
            weights[j] = nj / n
            means[j] = (resp[:, j] @ data) / nj
            np.subtract(data, means[j], out=scratch)
            np.square(scratch, out=scratch)
            variances[j] = (resp[:, j] @ scratch) / nj
            variances[j] = max(variances[j], _MIN_VARIANCE)

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    _mixture_expectation(data, weights, means, variances, resp, scratch)

    fast_component = int(np.argmin(means))
    result: np.ndarray = resp[:, fast_component]
    return result


def _mixture_expectation(
    data: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    responsibilities: np.ndarray,
    scratch: np.ndarray,
) -> float:
    """Fill normalized responsibilities and return their log-likelihood."""
    for component in range(len(weights)):
        np.subtract(data, means[component], out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, -0.5 / variances[component], out=scratch)
        np.exp(scratch, out=scratch)
        scale = weights[component] / math.sqrt(2.0 * math.pi * variances[component])
        np.multiply(scratch, scale, out=responsibilities[:, component])

    row_sums = np.sum(responsibilities, axis=1)
    regular = np.isfinite(row_sums) & (row_sums > 0.0)
    if np.all(regular):
        log_likelihood = float(np.sum(np.log(row_sums)))
        responsibilities /= row_sums[:, None]
        return log_likelihood

    log_likelihood = float(np.sum(np.log(row_sums[regular])))
    np.divide(
        responsibilities,
        row_sums[:, None],
        out=responsibilities,
        where=regular[:, None],
    )

    underflow = ~regular
    underflow_data = data[underflow]
    log_joint = np.empty((len(underflow_data), len(weights)))
    for component in range(len(weights)):
        component_values = log_joint[:, component]
        np.subtract(underflow_data, means[component], out=component_values)
        np.square(component_values, out=component_values)
        np.multiply(component_values, -0.5 / variances[component], out=component_values)
        component_values += math.log(weights[component]) - 0.5 * (
            _LOG_TWO_PI + math.log(variances[component])
        )

    row_maximum = np.max(log_joint, axis=1)
    log_joint -= row_maximum[:, None]
    np.exp(log_joint, out=log_joint)
    normalizers = np.sum(log_joint, axis=1)
    log_joint /= normalizers[:, None]
    responsibilities[underflow] = log_joint
    log_likelihood += float(np.sum(row_maximum + np.log(normalizers)))
    return log_likelihood
