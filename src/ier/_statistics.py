"""Small statistical primitives used internally by IER.

The implementations here intentionally cover only the narrow operations the
package needs.  Keeping them local avoids making a large general-purpose
statistics library a runtime dependency.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

_GAMMA_EPSILON = 1e-14
_GAMMA_MAX_ITERATIONS = 10_000
_CONTINUED_FRACTION_FLOOR = 1e-300
_QUANTILE_MAX_ITERATIONS = 128
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def normal_quantile(probability: float) -> float:
    """Return the standard-normal quantile for a probability in ``[0, 1]``."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite value between 0 and 1")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    return NormalDist().inv_cdf(probability)


def normal_pdf(
    values: np.ndarray,
    *,
    loc: float | np.ndarray = 0.0,
    scale: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Evaluate normal densities with NumPy broadcasting."""
    scale_array = np.asarray(scale, dtype=float)
    if np.any(scale_array <= 0.0):
        raise ValueError("scale must be positive")
    standardized = (values - loc) / scale_array
    result: np.ndarray = np.exp(-0.5 * standardized**2) / (scale_array * _SQRT_TWO_PI)
    return result


def logistic_transform(values: np.ndarray) -> np.ndarray:
    """Apply the logistic transform without overflowing at either extreme."""
    value_array = np.asarray(values, dtype=float)
    result = np.empty_like(value_array)
    nonnegative = value_array >= 0.0

    np.negative(value_array, out=result)
    np.exp(result, out=result, where=nonnegative)
    np.logical_not(nonnegative, out=nonnegative)
    np.exp(value_array, out=result, where=nonnegative)
    denominator = 1.0 + result
    np.divide(result, denominator, out=result, where=nonnegative)
    np.logical_not(nonnegative, out=nonnegative)
    np.reciprocal(denominator, out=result, where=nonnegative)
    return result


def _regularized_gamma_pair(shape: float, value: float) -> tuple[float, float]:
    """Return regularized lower/upper incomplete gamma values ``P`` and ``Q``.

    A power series is used below ``shape + 1`` and a modified-Lentz continued
    fraction above it.  Returning both tails lets the quantile solver avoid
    cancellation when probabilities are close to one.
    """
    if shape <= 0.0 or value < 0.0 or not math.isfinite(shape):
        raise ValueError("shape must be positive and value must be non-negative")
    if value == 0.0:
        return 0.0, 1.0
    if math.isinf(value):
        return 1.0, 0.0

    log_scale = -value + shape * math.log(value) - math.lgamma(shape)

    if value < shape + 1.0:
        term = 1.0 / shape
        series = term
        denominator = shape
        for _ in range(_GAMMA_MAX_ITERATIONS):
            denominator += 1.0
            term *= value / denominator
            series += term
            if abs(term) <= abs(series) * _GAMMA_EPSILON:
                lower = series * math.exp(log_scale)
                lower = min(max(lower, 0.0), 1.0)
                return lower, 1.0 - lower
        raise ArithmeticError("regularized gamma series did not converge")

    denominator = value + 1.0 - shape
    if abs(denominator) < _CONTINUED_FRACTION_FLOOR:
        denominator = _CONTINUED_FRACTION_FLOOR
    reciprocal_previous = 1.0 / _CONTINUED_FRACTION_FLOOR
    reciprocal_current = 1.0 / denominator
    fraction = reciprocal_current

    for iteration in range(1, _GAMMA_MAX_ITERATIONS + 1):
        coefficient = -float(iteration) * (float(iteration) - shape)
        denominator += 2.0
        reciprocal_current = coefficient * reciprocal_current + denominator
        if abs(reciprocal_current) < _CONTINUED_FRACTION_FLOOR:
            reciprocal_current = _CONTINUED_FRACTION_FLOOR
        reciprocal_previous = denominator + coefficient / reciprocal_previous
        if abs(reciprocal_previous) < _CONTINUED_FRACTION_FLOOR:
            reciprocal_previous = _CONTINUED_FRACTION_FLOOR
        reciprocal_current = 1.0 / reciprocal_current
        change = reciprocal_previous * reciprocal_current
        fraction *= change
        if abs(change - 1.0) <= _GAMMA_EPSILON:
            upper = math.exp(log_scale) * fraction
            upper = min(max(upper, 0.0), 1.0)
            return 1.0 - upper, upper

    raise ArithmeticError("regularized gamma continued fraction did not converge")


def _chi_square_tail_pair(value: float, degrees_of_freedom: int) -> tuple[float, float]:
    """Return chi-square CDF and survival function at ``value``."""
    return _regularized_gamma_pair(degrees_of_freedom / 2.0, value / 2.0)


def _chi_square_density(value: float, degrees_of_freedom: int) -> float:
    """Return the chi-square probability density."""
    if value <= 0.0:
        return math.inf if degrees_of_freedom < 2 else float(degrees_of_freedom == 2) / 2.0
    half_df = degrees_of_freedom / 2.0
    log_density = (
        (half_df - 1.0) * math.log(value)
        - value / 2.0
        - half_df * math.log(2.0)
        - math.lgamma(half_df)
    )
    return math.exp(log_density)


def chi_square_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Return a chi-square quantile without an external statistics dependency.

    The inverse is solved with safeguarded Newton iterations.  Wilson-Hilferty
    supplies the usual starting point, while the lower-tail gamma asymptotic
    handles small degrees of freedom where that approximation becomes negative.
    """
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite value between 0 and 1")
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return math.inf

    df = float(degrees_of_freedom)
    normal = normal_quantile(probability)
    correction = 1.0 - 2.0 / (9.0 * df) + normal * math.sqrt(2.0 / (9.0 * df))
    wilson_hilferty = df * correction**3 if correction > 0.0 else 0.0

    half_df = df / 2.0
    lower_tail_guess = 2.0 * math.exp(
        (math.log(probability) + math.lgamma(half_df + 1.0)) / half_df
    )
    estimate = wilson_hilferty if wilson_hilferty > 0.0 else lower_tail_guess
    estimate = max(estimate, np.finfo(float).tiny)

    lower = 0.0
    upper = max(df, estimate, 1.0)
    while True:
        cdf, survival = _chi_square_tail_pair(upper, degrees_of_freedom)
        below_target = cdf < probability if probability <= 0.5 else survival > 1.0 - probability
        if not below_target:
            break
        upper *= 2.0
        if math.isinf(upper):
            raise ArithmeticError("could not bracket chi-square quantile")

    value = min(max(estimate, np.finfo(float).tiny), upper)
    target_tail = probability if probability <= 0.5 else 1.0 - probability

    for _ in range(_QUANTILE_MAX_ITERATIONS):
        cdf, survival = _chi_square_tail_pair(value, degrees_of_freedom)
        if probability <= 0.5:
            residual = cdf - probability
            below_target = residual < 0.0
            derivative = _chi_square_density(value, degrees_of_freedom)
        else:
            residual = survival - target_tail
            below_target = residual > 0.0
            derivative = -_chi_square_density(value, degrees_of_freedom)

        if abs(residual) <= max(target_tail * 5e-14, 1e-300):
            return value

        if below_target:
            lower = value
        else:
            upper = value

        candidate = (
            value - residual / derivative if derivative > 0.0 or derivative < 0.0 else math.nan
        )
        if not math.isfinite(candidate) or not lower < candidate < upper:
            candidate = (lower + upper) / 2.0

        if abs(candidate - value) <= max(abs(value) * 5e-14, np.finfo(float).tiny):
            return candidate
        value = candidate

    raise ArithmeticError("chi-square quantile did not converge")


def chi_square_quantiles(probabilities: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Vectorized wrapper around :func:`chi_square_quantile`."""
    probability_array = np.asarray(probabilities, dtype=float)
    flat_result = np.fromiter(
        (chi_square_quantile(float(item), degrees_of_freedom) for item in probability_array.flat),
        dtype=float,
        count=probability_array.size,
    )
    return flat_result.reshape(probability_array.shape)
