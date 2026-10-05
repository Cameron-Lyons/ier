"""
Response time indices for detecting careless responding.

Extremely fast or unusually consistent response times may indicate
careless or inattentive responding.
"""

from __future__ import annotations

import math
import numbers
import warnings
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from ier._column_statistics import column_mean
from ier._flagging import threshold_flags
from ier._row_statistics import (
    _scaled_subnormal_moment_rows,
    row_mean,
    row_mean_std,
    row_median,
    row_slices,
    row_std,
)
from ier._validation import (
    MatrixLike,
    validate_integer,
    validate_matrix_input,
    validate_probability,
    validate_score_array,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ier.types import ResponseTimeFlagDirection

_LOG_TWO_PI = math.log(2.0 * math.pi)
_MIN_COMPONENT_MASS = 1e-10
_MIN_VARIANCE = 1e-10
_MIN_DEVIATION = math.sqrt(_MIN_VARIANCE)


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
        >>> import numpy as np
        >>> times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]
        >>> avg_times = response_time(times, metric="mean")
        >>> print(np.round(avg_times, 2).tolist())  # Second person has suspiciously fast times
        [2.77, 0.5, 2.5]
    """
    times_array = validate_matrix_input(times, min_columns=1)

    result: np.ndarray
    if metric == "mean":
        result = row_mean(times_array, ignore_nan=True)
    elif metric == "median":
        result = row_median(times_array, ignore_nan=True)
    elif metric == "sd":
        result = row_std(times_array, ignore_nan=True)
    elif metric == "min":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.nanmin(times_array, axis=1)
    else:
        raise ValueError(f"unknown metric: {metric}. Use 'mean', 'median', 'sd', or 'min'")
    return result


def response_time_score_flags(
    scores: ArrayLike,
    threshold: float | None = None,
    cutoff_percentile: float | None = None,
    direction: ResponseTimeFlagDirection = "low",
) -> np.ndarray:
    """
    Flag a retained one-dimensional response-time score vector.

    Use low-tail flagging for direct timing summaries and consistency scores,
    or high-tail flagging for fast-component mixture probabilities. Fixed
    thresholds include equality; percentile-derived cutoffs exclude ties.
    When ``cutoff_percentile`` is omitted, the low tail defaults to the 5th
    percentile and the high tail to the 95th percentile.

    Parameters:
    - scores: Retained per-respondent response-time scores.
    - threshold: Optional fixed cutoff in the score's units.
    - cutoff_percentile: Optional sample-relative cutoff percentile.
    - direction: Suspicious tail, ``"low"`` or ``"high"``.

    Returns:
    - Boolean array where ``True`` indicates a suspicious score.

    Example:
        >>> times = [[2.1, 3.4, 2.8], [0.5, 0.4, 0.6], [2.5, 2.3, 2.7]]
        >>> medians = response_time(times, metric="median")
        >>> strict = response_time_score_flags(medians, cutoff_percentile=1)
        >>> print(strict.tolist())
        [False, True, False]
        >>> mixture = response_time_mixture(times, random_seed=42)
        >>> likely_fast = response_time_score_flags(mixture, direction="high")
        >>> print(likely_fast.tolist())
        [False, True, False]
    """
    if not isinstance(direction, str) or direction not in {"high", "low"}:
        raise ValueError("direction must be 'high' or 'low'")
    validated_scores = validate_score_array(scores, name="response time scores")
    percentile = cutoff_percentile
    if percentile is None:
        percentile = 95.0 if direction == "high" else 5.0
    return threshold_flags(
        validated_scores,
        threshold=threshold,
        percentile=percentile,
        direction=direction,
    )


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

    return response_time_score_flags(
        person_times,
        threshold=threshold,
        cutoff_percentile=cutoff_percentile,
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
        >>> import numpy as np
        >>> times = [[2.1, 3.4, 2.8], [1.0, 1.0, 1.0], [2.5, 2.3, 2.7]]
        >>> cv = response_time_consistency(times)
        >>> print(np.round(cv, 2).tolist())  # Second person has very consistent (suspicious) times
        [0.19, 0.0, 0.07]
    """
    times_array = validate_matrix_input(times, min_columns=2)

    means, stds = row_mean_std(times_array, ignore_nan=True)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        cv: np.ndarray = stds / means

    for start, stop in row_slices(len(times_array), times_array.shape[1]):
        scaled = _scaled_subnormal_moment_rows(
            times_array[start:stop], means[start:stop], stds[start:stop], repair_mean=True
        )
        if scaled is None:
            continue
        positions, values, _ = scaled
        scaled_means, scaled_stds = row_mean_std(values, ignore_nan=True)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            cv[start + positions] = scaled_stds / scaled_means

    return cv


@overload
def response_time_effort(
    times: MatrixLike,
    thresholds: ArrayLike | None = None,
    *,
    normative_fraction: float = 0.10,
    max_threshold: float | None = None,
    return_item_flags: Literal[False] = False,
) -> np.ndarray: ...
@overload
def response_time_effort(
    times: MatrixLike,
    thresholds: ArrayLike | None = None,
    *,
    normative_fraction: float = 0.10,
    max_threshold: float | None = None,
    return_item_flags: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def response_time_effort(
    times: MatrixLike,
    thresholds: ArrayLike | None = None,
    *,
    normative_fraction: float = 0.10,
    max_threshold: float | None = None,
    return_item_flags: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]: ...


def response_time_effort(
    times: MatrixLike,
    thresholds: ArrayLike | None = None,
    *,
    normative_fraction: float = 0.10,
    max_threshold: float | None = None,
    return_item_flags: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Calculate response time effort: the share of answered items not rapidly answered.

    Response time effort (RTE; Wise & Kong, 2005) compares every response time
    with a threshold for that item, so long and short items are judged by their
    own expected time. A response faster than the item threshold (strictly
    below it) is a rapid response. By default each item threshold is the
    normative NT10 threshold (Wise & Ma, 2012): 10% of the item's mean response
    time over the respondents who answered it, optionally capped (commonly at
    10 seconds) with ``max_threshold``.

    Parameters:
    - times: A matrix of response times where rows are individuals and
             columns are items. Missing times (``NaN``) are unanswered items.
    - thresholds: Optional item thresholds in the units of ``times``: one value
                  for every item or a vector with one value per item. Missing
                  (``NaN``), infinite, and nonpositive values exclude their
                  items, but ``max_threshold`` caps positive infinity like any
                  other value. If None, normative thresholds are used.
    - normative_fraction: Fraction of each item's mean time used as its
                          normative threshold, greater than 0 and at most 1
                          (default 0.10).
    - max_threshold: Optional positive cap applied to every item threshold.
    - return_item_flags: If True, also return the Boolean matrix of rapid
                         responses, for example for effort-moderated scoring.

    Returns:
    - A numpy array of RTE values in ``[0, 1]``: the proportion of each
      respondent's answered items with a response time at or above the item
      threshold. Lower values indicate more rapid responding. Items whose
      threshold, after any ``max_threshold`` cap, is missing, infinite, or not
      positive are excluded, and respondents without an answered eligible
      item are unavailable (``NaN``).
      With ``return_item_flags``, a tuple of (scores, rapid) where ``rapid`` is
      True for answered eligible items below their threshold.

    Raises:
    - ValueError: If inputs are invalid, thresholds do not match the items, or
                  ``normative_fraction`` or ``max_threshold`` is out of range.

    Example:
        >>> times = [[12, 30, 18], [0.5, 1.5, 20], [14, 28, np.nan], [13.5, 20.5, 22]]
        >>> np.round(response_time_effort(times), 2).tolist()  # thresholds 1, 2, and 2
        [1.0, 0.33, 1.0, 1.0]
        >>> scores, rapid = response_time_effort(times, return_item_flags=True)
        >>> rapid[1].tolist()
        [True, True, False]
        >>> response_time_effort(times, thresholds=15).tolist()
        [0.6666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666]
    """
    if not isinstance(return_item_flags, bool):
        raise ValueError("return_item_flags must be a boolean")
    fraction = validate_probability(normative_fraction, name="normative_fraction")
    if fraction == 0.0:
        raise ValueError("normative_fraction must be greater than 0")
    cap = None if max_threshold is None else _positive_time(max_threshold, name="max_threshold")
    times_array = validate_matrix_input(times, min_columns=1)
    n_items = times_array.shape[1]

    if thresholds is None:
        item_thresholds = fraction * column_mean(times_array, ignore_nan=True)
    else:
        item_thresholds = _item_thresholds(thresholds, n_items)
    if cap is not None:
        item_thresholds = np.minimum(item_thresholds, cap)

    eligible = np.isfinite(item_thresholds) & (item_thresholds > 0)
    # No time is below negative infinity, so excluded items are never rapid.
    cutoffs = np.where(eligible, item_thresholds, -np.inf)
    n_eligible = int(np.count_nonzero(eligible))
    timed = times_array.dtype.kind == "f"

    scores = np.full(len(times_array), np.nan)
    rapid_items = np.zeros(times_array.shape, dtype=bool) if return_item_flags else None
    for start, stop in row_slices(*times_array.shape):
        block = times_array[start:stop]
        rapid = block < cutoffs
        if timed:
            answered = ~np.isnan(block)
            if n_eligible < n_items:
                answered &= eligible
            n_answered = np.count_nonzero(answered, axis=1)
        else:
            n_answered = np.full(len(block), n_eligible)
        n_effortful = n_answered - np.count_nonzero(rapid, axis=1)
        np.divide(n_effortful, n_answered, out=scores[start:stop], where=n_answered > 0)
        if rapid_items is not None:
            rapid_items[start:stop] = rapid

    if rapid_items is not None:
        return scores, rapid_items
    return scores


def response_time_effort_flag(
    times: MatrixLike,
    threshold: float = 0.90,
    *,
    thresholds: ArrayLike | None = None,
    normative_fraction: float = 0.10,
    max_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate response time effort and flag respondents below an RTE cutoff.

    Parameters:
    - times: A matrix of response times.
    - threshold: RTE value between 0 and 1 below which (strictly) to flag
                 (default 0.90, a common rapid-guessing screening rule).
    - thresholds: Optional item thresholds passed to ``response_time_effort()``.
    - normative_fraction: Normative threshold fraction passed to
                          ``response_time_effort()`` (default 0.10).
    - max_threshold: Optional item-threshold cap passed to
                     ``response_time_effort()``.

    Returns:
    - Tuple of (scores, flags) where flags is True for respondents whose RTE is
      below ``threshold``. Unavailable (``NaN``) scores are never flagged.

    Example:
        >>> times = [[12, 30, 18], [0.5, 1.5, 20], [14, 28, np.nan], [13.5, 20.5, 22]]
        >>> scores, flags = response_time_effort_flag(times)
        >>> flags.tolist()
        [False, True, False, False]
    """
    cutoff = validate_probability(threshold, name="threshold")
    scores = response_time_effort(
        times,
        thresholds,
        normative_fraction=normative_fraction,
        max_threshold=max_threshold,
    )
    # Comparisons with unavailable scores are false, so NaN is never flagged.
    flags: np.ndarray = scores < cutoff
    return scores, flags


def _positive_time(value: object, *, name: str) -> float:
    """Return a positive finite real number as a Python float."""
    message = f"{name} must be a positive finite number"
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    rejected = (bool, np.bool_, np.datetime64, np.timedelta64)
    if isinstance(value, rejected) or not isinstance(value, (numbers.Real, Decimal)):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not (math.isfinite(result) and result > 0):
        raise ValueError(message)
    return result


def _item_thresholds(thresholds: ArrayLike, n_items: int) -> np.ndarray:
    """Validate one shared or one-per-item threshold in the units of the times.

    Infinite thresholds are kept, so ``max_threshold`` caps them like any other
    threshold; uncapped, they exclude their items.
    """
    try:
        values = np.asarray(thresholds)
    except (TypeError, ValueError) as error:
        raise ValueError("thresholds must be a real number or a one-dimensional array") from error
    shared = values.ndim == 0
    if shared:
        values = values.reshape(1)
    infinite = np.isinf(values) if values.dtype.kind == "f" else None
    if infinite is not None and infinite.any():
        # Validate the remaining thresholds, then restore the infinite ones.
        item_thresholds = validate_score_array(
            np.where(infinite, np.nan, values), name="thresholds"
        )
        item_thresholds[infinite] = values[infinite]
    else:
        item_thresholds = validate_score_array(values, name="thresholds")
    if shared:
        return np.full(n_items, item_thresholds[0])
    if len(item_thresholds) != n_items:
        raise ValueError(
            f"thresholds must contain one value per item ({n_items}), got {len(item_thresholds)}"
        )
    return item_thresholds


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
    - n_components: Integer number of mixture components, at least 2 (default 2).
    - log_transform: If True (default), log-transform median times before fitting.
    - random_seed: Optional seed for reproducibility of EM initialization.

    Returns:
    - A numpy array of posterior probabilities of belonging to the fast component,
      one per respondent. Higher values indicate greater likelihood of careless
      (fast) responding. Equal usable medians return ``1 / n_components``;
      the data cannot distinguish a fast group in that case. Nonpositive or
      nonfinite medians remain unavailable (``NaN``).

    Raises:
    - ValueError: If n_components < 2 or data is insufficient.

    Example:
        >>> times = [[5.0, 6.0, 4.0], [0.5, 0.6, 0.4], [4.5, 5.5, 5.0]]
        >>> probs = response_time_mixture(times, random_seed=42)
    """
    n_components = validate_integer(
        n_components,
        message="n_components must be an integer of at least 2",
        minimum=2,
        minimum_message="n_components must be at least 2",
    )

    times_array = validate_matrix_input(times, min_columns=1)

    medians = row_median(times_array, ignore_nan=True)

    valid_mask = np.isfinite(medians) & (medians > 0)
    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid < n_components:
        raise ValueError(
            f"insufficient valid observations ({n_valid}) for {n_components} components"
        )

    data = medians[valid_mask]

    if log_transform:
        np.log(data, out=data)

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
    lower, upper = float(np.min(data)), float(np.max(data))
    if lower == upper:
        return np.full(n, 1.0 / k)

    magnitude = max(abs(lower), abs(upper))
    exponent = 0
    if magnitude > math.sqrt(np.finfo(float).max / n) / 4:
        _, exponent = math.frexp(magnitude)
        with np.errstate(under="ignore"):
            data = np.ldexp(data, -exponent)
        lower, upper = float(np.min(data)), float(np.max(data))
        magnitude = max(abs(lower), abs(upper))
    if upper - lower < magnitude * math.sqrt(np.finfo(float).eps):
        # Preserve small differences from a large common timing baseline.
        data = data - data[0]
    minimum_deviation = math.ldexp(_MIN_DEVIATION, -exponent)
    jitter = math.ldexp(0.01, -exponent)

    sorted_data = np.sort(data)
    split_points = np.array_split(sorted_data, k)
    means = np.array([np.mean(s) for s in split_points])
    del sorted_data, split_points
    _, spread = row_mean_std(data[None, :], ignore_nan=False)
    deviations = np.full(k, max(float(spread[0]) / math.sqrt(k), minimum_deviation))
    weights = np.full(k, 1.0 / k)

    means += rng.normal(0, min(jitter, float(spread[0])), size=k)

    # Component updates traverse columns; contiguous columns also speed row sums.
    resp = np.empty((n, k), order="F")
    scratch = np.empty(n)
    prev_ll = -np.inf

    for _ in range(max_iter):
        ll = _mixture_expectation(data, weights, means, deviations, resp, scratch)

        for j in range(k):
            nj = resp[:, j].sum()
            if nj < _MIN_COMPONENT_MASS:
                weights[j] = 0.0
                continue
            weights[j] = nj / n
            means[j] = (resp[:, j] @ data) / nj
            deviations[j] = max(
                _weighted_deviation(data, means[j], resp[:, j], nj, scratch), minimum_deviation
            )
        weights /= weights.sum()

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    _mixture_expectation(data, weights, means, deviations, resp, scratch)

    fast_component = int(np.argmin(np.where(weights > 0, means, np.inf)))
    # Retain only the returned probabilities, not every component's workspace.
    result: np.ndarray = resp[:, fast_component].copy()
    return result


def _weighted_deviation(
    data: np.ndarray, mean: float, weights: np.ndarray, mass: float, scratch: np.ndarray
) -> float:
    """Reduce weighted residuals, rescaling only when their squares lose range."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        np.subtract(data, mean, out=scratch)
        np.square(scratch, out=scratch)
        variance = float(weights @ scratch) / mass
        if variance >= np.finfo(float).tiny and math.isfinite(variance):
            return math.sqrt(variance)
        np.subtract(data, mean, out=scratch)
        scratch *= np.sqrt(weights)
        scale = float(np.max(np.abs(scratch)))
        if scale == 0:
            return 0.0
        scratch /= scale
        return scale * math.sqrt(float(scratch @ scratch) / mass)


def _mixture_expectation(
    data: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    deviations: np.ndarray,
    responsibilities: np.ndarray,
    scratch: np.ndarray,
) -> float:
    """Fill responsibilities using standard deviations to preserve extreme scales."""
    # Very narrow components can amplify an underflowed exponential into a
    # meaningful density. Combine their exponent and normalization in log space.
    log_only = np.any(deviations < weights * (np.finfo(float).eps / math.sqrt(2.0 * math.pi)))
    if log_only:
        row_sums = np.zeros(len(data))
    else:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for component in range(len(weights)):
                if weights[component] == 0:
                    responsibilities[:, component] = 0.0
                    continue
                np.subtract(data, means[component], out=scratch)
                scratch /= deviations[component]
                np.square(scratch, out=scratch)
                scratch *= -0.5
                np.exp(scratch, out=scratch)
                scale = (weights[component] / deviations[component]) / math.sqrt(2.0 * math.pi)
                np.multiply(scratch, scale, out=responsibilities[:, component])

            row_sums = np.sum(responsibilities, axis=1)
    regular = np.isfinite(row_sums) & (row_sums >= np.finfo(float).tiny)
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
    log_joint = np.empty((len(underflow_data), len(weights)), order="F")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for component in range(len(weights)):
            component_values = log_joint[:, component]
            if weights[component] == 0:
                component_values.fill(-np.inf)
                continue
            np.subtract(underflow_data, means[component], out=component_values)
            component_values /= deviations[component]
            np.square(component_values, out=component_values)
            component_values *= -0.5
            component_values += (
                math.log(weights[component]) - 0.5 * _LOG_TWO_PI - math.log(deviations[component])
            )

    row_maximum = np.max(log_joint, axis=1)
    extreme = ~np.isfinite(row_maximum)
    close_densities = np.count_nonzero(log_joint >= row_maximum[:, None] - 32.0, axis=1) > 1
    extreme |= (row_maximum < -1024.0) & close_densities
    if np.any(extreme):
        # Huge or nearly tied log densities can lose their differences even
        # before overflow. Resolve only those exceptional rows in high precision.
        for row in np.flatnonzero(extreme):
            log_joint[row], row_maximum[row] = _extreme_log_joint(
                float(underflow_data[row]), weights, means, deviations
            )
        # The helper already removes the common log density for extreme rows.
        log_joint[~extreme] -= row_maximum[~extreme, None]
    else:
        log_joint -= row_maximum[:, None]
    with np.errstate(under="ignore"):
        np.exp(log_joint, out=log_joint)
    normalizers = np.sum(log_joint, axis=1)
    log_joint /= normalizers[:, None]
    responsibilities[underflow] = log_joint
    with np.errstate(over="ignore"):
        log_likelihood += float(np.sum(row_maximum + np.log(normalizers)))
    return log_likelihood


def _extreme_log_joint(
    value: float, weights: np.ndarray, means: np.ndarray, deviations: np.ndarray
) -> tuple[np.ndarray, float]:
    """Resolve unrepresentable Gaussian exponents only for exceptional observations."""
    with localcontext() as context:
        context.prec = 800
        observation = Decimal.from_float(value)
        joint = []
        for weight, mean, deviation in zip(weights, means, deviations, strict=True):
            if weight == 0:
                joint.append(Decimal("-Infinity"))
                continue
            distance = (observation - Decimal.from_float(float(mean))) / Decimal.from_float(
                float(deviation)
            )
            log_scale = math.log(weight) - math.log(deviation) - 0.5 * _LOG_TWO_PI
            joint.append(Decimal.from_float(log_scale) - distance * distance / 2)
        maximum = max(joint)
        return np.array([float(value - maximum) for value in joint]), float(maximum)
