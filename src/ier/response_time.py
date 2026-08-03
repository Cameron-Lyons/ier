"""
Response time indices for detecting careless responding.

Extremely fast or unusually consistent response times may indicate
careless or inattentive responding.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ier._flagging import threshold_flags
from ier._row_statistics import row_mean, row_mean_std, row_median, row_std
from ier._validation import MatrixLike, validate_matrix_input, validate_score_array

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ier.types import ResponseTimeFlagDirection

_LOG_TWO_PI = math.log(2.0 * math.pi)
_MIN_COMPONENT_MASS = 1e-10
_MIN_VARIANCE = 1e-10


def _readonly_model_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return a validated, independently owned mixture-model vector."""
    try:
        vector = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from error
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class ResponseTimeMixtureModel:
    """Reusable Gaussian-mixture calibration for response-time scoring.

    Arrays are copied and made read-only during construction so a fitted model
    can be shared safely across repeated scoring calls.
    """

    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    log_transform: bool = True

    def __post_init__(self) -> None:
        weights = _readonly_model_vector(self.weights, "weights")
        means = _readonly_model_vector(self.means, "means")
        variances = _readonly_model_vector(self.variances, "variances")
        if len(weights) < 2:
            raise ValueError("a response-time mixture model requires at least two components")
        if len(means) != len(weights) or len(variances) != len(weights):
            raise ValueError("weights, means, and variances must have the same length")
        if np.any(weights <= 0.0):
            raise ValueError("weights must be positive")
        if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("weights must sum to one")
        if np.any(variances <= 0.0):
            raise ValueError("variances must be positive")
        if not isinstance(self.log_transform, bool):
            raise ValueError("log_transform must be a boolean")

        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "variances", variances)

    @property
    def n_components(self) -> int:
        """Number of calibrated mixture components."""
        return len(self.weights)

    @property
    def fast_component(self) -> int:
        """Position of the component with the lowest calibrated mean."""
        return int(np.argmin(self.means))


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
        >>> medians = response_time(times, metric="median")
        >>> strict = response_time_score_flags(medians, cutoff_percentile=1)
        >>> mixture = response_time_mixture(times, random_seed=42)
        >>> likely_fast = response_time_score_flags(mixture, direction="high")
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
        >>> times = [[2.1, 3.4, 2.8], [1.0, 1.0, 1.0], [2.5, 2.3, 2.7]]
        >>> cv = response_time_consistency(times)
        >>> print(cv)  # Second person has very consistent (suspicious) times
    """
    times_array = validate_matrix_input(times, min_columns=2)

    means, stds = row_mean_std(times_array, ignore_nan=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        cv: np.ndarray = stds / means

    return cv


def _validate_n_components(n_components: int) -> int:
    """Return a validated Gaussian-mixture component count."""
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise ValueError("n_components must be an integer")
    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    return n_components


def _prepare_mixture_data(
    times: MatrixLike,
    *,
    log_transform: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return medians, their validity mask, and prepared finite model values."""
    if not isinstance(log_transform, bool):
        raise ValueError("log_transform must be a boolean")
    times_array = validate_matrix_input(times, min_columns=1)
    medians = row_median(times_array, ignore_nan=True)
    valid_mask = np.isfinite(medians) & (medians > 0)
    data = medians[valid_mask].copy()
    if log_transform:
        np.log(data, out=data)
    return medians, valid_mask, data


def _restore_mixture_scores(
    medians: np.ndarray,
    valid_mask: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Restore finite mixture scores to respondent order with unavailable NaNs."""
    result = np.full(len(medians), np.nan)
    result[valid_mask] = scores
    return result


def fit_response_time_mixture(
    times: MatrixLike,
    n_components: int = 2,
    log_transform: bool = True,
    random_seed: int | None = None,
) -> ResponseTimeMixtureModel:
    """Fit a reusable Gaussian-mixture calibration to response-time medians.

    The returned model can score later cohorts with
    :func:`response_time_mixture_scores` without repeating EM fitting. This is
    useful for applying a fixed reference-cohort calibration and for high-volume
    workflows that score multiple batches.

    Parameters:
    - times: Reference response-time matrix with respondents in rows.
    - n_components: Number of Gaussian mixture components.
    - log_transform: Whether to fit on log median response times.
    - random_seed: Optional seed for reproducible EM initialization.

    Returns:
    - An immutable response-time mixture model.
    """
    n_components = _validate_n_components(n_components)
    _, _, data = _prepare_mixture_data(times, log_transform=log_transform)
    if len(data) < n_components:
        raise ValueError(
            f"insufficient valid observations ({len(data)}) for {n_components} components"
        )
    return _fit_gaussian_mixture_model(
        data,
        n_components,
        np.random.default_rng(random_seed),
        log_transform=log_transform,
    )


def response_time_mixture_scores(
    times: MatrixLike,
    model: ResponseTimeMixtureModel,
) -> np.ndarray:
    """Score response times against a fitted Gaussian-mixture calibration.

    Per-person medians receive the posterior probability of membership in the
    model's fastest component. Non-finite and non-positive medians remain
    unavailable as ``NaN``. Unlike :func:`response_time_mixture`, this function
    never refits the mixture and can score cohorts smaller than the component
    count.

    Parameters:
    - times: Response-time matrix to score.
    - model: Calibration returned by :func:`fit_response_time_mixture`.

    Returns:
    - Fast-component posterior probability for each respondent.
    """
    if not isinstance(model, ResponseTimeMixtureModel):
        raise TypeError("model must be a ResponseTimeMixtureModel")
    medians, valid_mask, data = _prepare_mixture_data(
        times,
        log_transform=model.log_transform,
    )
    scores = _score_gaussian_mixture_data(data, model)
    return _restore_mixture_scores(medians, valid_mask, scores)


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
    the lowest mean is identified as the "fast" (careless) component. Use
    :func:`fit_response_time_mixture` and :func:`response_time_mixture_scores`
    when one calibration should be reused across later cohorts or batches.

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
    n_components = _validate_n_components(n_components)
    medians, valid_mask, data = _prepare_mixture_data(
        times,
        log_transform=log_transform,
    )
    if len(data) < n_components:
        raise ValueError(
            f"insufficient valid observations ({len(data)}) for {n_components} components"
        )
    model = _fit_gaussian_mixture_model(
        data,
        n_components,
        np.random.default_rng(random_seed),
        log_transform=log_transform,
    )
    scores = _score_gaussian_mixture_data(data, model)
    return _restore_mixture_scores(medians, valid_mask, scores)


def _em_gaussian_mixture(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Fit k-component Gaussian mixture via EM; return posterior P(fast component)."""
    model = _fit_gaussian_mixture_model(
        data,
        k,
        rng,
        log_transform=False,
        max_iter=max_iter,
        tol=tol,
    )
    return _score_gaussian_mixture_data(data, model)


def _fit_gaussian_mixture_model(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    log_transform: bool,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> ResponseTimeMixtureModel:
    """Fit Gaussian-mixture parameters to prepared one-dimensional data."""
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

    return ResponseTimeMixtureModel(
        weights=weights,
        means=means,
        variances=variances,
        log_transform=log_transform,
    )


def _score_gaussian_mixture_data(
    data: np.ndarray,
    model: ResponseTimeMixtureModel,
) -> np.ndarray:
    """Return fast-component posteriors without a respondent-by-component matrix."""
    if len(data) == 0:
        return np.empty(0)

    scores = np.empty(len(data))
    normalizers = np.zeros(len(data))
    scratch = np.empty(len(data))
    fast_component = model.fast_component
    for component in range(model.n_components):
        _gaussian_joint_density(
            data,
            model.weights[component],
            model.means[component],
            model.variances[component],
            scratch,
        )
        np.add(normalizers, scratch, out=normalizers)
        if component == fast_component:
            np.copyto(scores, scratch)

    regular = np.isfinite(normalizers) & (normalizers > 0.0)
    np.divide(scores, normalizers, out=scores, where=regular)
    if np.all(regular):
        return scores

    underflow = ~regular
    underflow_data = data[underflow]
    underflow_count = len(underflow_data)
    row_maximum = normalizers[:underflow_count]
    row_maximum.fill(-np.inf)
    underflow_scratch = scratch[:underflow_count]

    for component in range(model.n_components):
        _gaussian_log_joint(
            underflow_data,
            model.weights[component],
            model.means[component],
            model.variances[component],
            underflow_scratch,
        )
        np.maximum(row_maximum, underflow_scratch, out=row_maximum)

    log_normalizers = np.zeros(underflow_count)
    for component in range(model.n_components):
        _gaussian_log_joint(
            underflow_data,
            model.weights[component],
            model.means[component],
            model.variances[component],
            underflow_scratch,
        )
        np.subtract(underflow_scratch, row_maximum, out=underflow_scratch)
        np.exp(underflow_scratch, out=underflow_scratch)
        np.add(log_normalizers, underflow_scratch, out=log_normalizers)

    _gaussian_log_joint(
        underflow_data,
        model.weights[fast_component],
        model.means[fast_component],
        model.variances[fast_component],
        underflow_scratch,
    )
    np.subtract(underflow_scratch, row_maximum, out=underflow_scratch)
    np.exp(underflow_scratch, out=underflow_scratch)
    np.divide(underflow_scratch, log_normalizers, out=underflow_scratch)
    scores[underflow] = underflow_scratch
    return scores


def _gaussian_joint_density(
    data: np.ndarray,
    weight: float,
    mean: float,
    variance: float,
    out: np.ndarray,
) -> None:
    """Write one weighted Gaussian density vector."""
    np.subtract(data, mean, out=out)
    np.square(out, out=out)
    np.multiply(out, -0.5 / variance, out=out)
    np.exp(out, out=out)
    scale = weight / math.sqrt(2.0 * math.pi * variance)
    np.multiply(out, scale, out=out)


def _gaussian_log_joint(
    data: np.ndarray,
    weight: float,
    mean: float,
    variance: float,
    out: np.ndarray,
) -> None:
    """Write one weighted Gaussian log-density vector."""
    np.subtract(data, mean, out=out)
    np.square(out, out=out)
    np.multiply(out, -0.5 / variance, out=out)
    out += math.log(weight) - 0.5 * (_LOG_TWO_PI + math.log(variance))


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
        _gaussian_log_joint(
            underflow_data,
            weights[component],
            means[component],
            variances[component],
            log_joint[:, component],
        )

    row_maximum = np.max(log_joint, axis=1)
    log_joint -= row_maximum[:, None]
    np.exp(log_joint, out=log_joint)
    normalizers = np.sum(log_joint, axis=1)
    log_joint /= normalizers[:, None]
    responsibilities[underflow] = log_joint
    log_likelihood += float(np.sum(row_maximum + np.log(normalizers)))
    return log_likelihood
