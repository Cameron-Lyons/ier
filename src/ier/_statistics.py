"""Small statistical primitives used internally by IER.

The implementations here intentionally cover only the narrow operations the
package needs.  Keeping them local avoids making a large general-purpose
statistics library a runtime dependency.
"""

from __future__ import annotations

import math
from operator import index
from statistics import NormalDist

import numpy as np

_GAMMA_EPSILON = 1e-14
_GAMMA_MAX_ITERATIONS = 10_000
_CONTINUED_FRACTION_FLOOR = 1e-300
_QUANTILE_MAX_ITERATIONS = 128


def _gamma_lower_series(shape: float, value: float) -> float:
    """Return the convergent lower-gamma series before applying its scale."""
    term = 1.0 / shape
    series = term
    denominator = shape
    for _ in range(_GAMMA_MAX_ITERATIONS):
        denominator += 1.0
        term *= value / denominator
        series += term
        if abs(term) <= abs(series) * _GAMMA_EPSILON:
            return series
    raise ArithmeticError("regularized gamma series did not converge")


def normal_quantile(probability: float) -> float:
    """Return the standard-normal quantile for a probability in ``[0, 1]``."""
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite value between 0 and 1")
    if probability == 0.0:
        return -math.inf
    if probability == 1.0:
        return math.inf
    return NormalDist().inv_cdf(probability)


def logistic_transform(values: np.ndarray) -> np.ndarray:
    """Apply the logistic transform, retaining subnormal tails and saturated limits."""
    value_array = np.asarray(values, dtype=float)
    result = np.empty_like(value_array)
    nonnegative = value_array >= 0.0

    np.negative(value_array, out=result)
    # Tiny exponential tails may underflow to zero. Some platform math libraries
    # also signal underflow for near-zero arguments whose exponential rounds to
    # one. Both are valid here; retain the caller's other floating-point checks.
    with np.errstate(under="ignore"):
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
    if shape <= 0.0 or value < 0.0 or math.isnan(value) or not math.isfinite(shape):
        raise ValueError("shape must be positive and value must be non-negative")
    if value == 0.0:
        return 0.0, 1.0
    if math.isinf(value):
        return 1.0, 0.0

    log_scale = -value + shape * math.log(value) - math.lgamma(shape)

    if value < shape + 1.0:
        lower = _gamma_lower_series(shape, value) * math.exp(log_scale)
        lower = min(max(lower, 0.0), 1.0)
        return lower, 1.0 - lower

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

    One- and two-degree special cases avoid iteration where they are accurate.
    General cases use safeguarded Newton iterations, with very small lower tails
    solved in logarithmic coordinates to preserve subnormal probabilities.
    """
    degrees_of_freedom = _validate_degrees_of_freedom(degrees_of_freedom)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite value between 0 and 1")
    return _chi_square_quantile(float(probability), degrees_of_freedom)


def _validate_degrees_of_freedom(value: int) -> int:
    """Use the same positive integer domain for scalar and array quantiles."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("degrees_of_freedom must be a positive integer")
    try:
        value = index(value)
    except TypeError as error:
        raise ValueError("degrees_of_freedom must be a positive integer") from error
    if value < 1:
        raise ValueError("degrees_of_freedom must be a positive integer")
    return value


def _chi_square_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Solve an already validated probability and degree count."""
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return math.inf
    # Gamma special values: https://dlmf.nist.gov/8.4.E5 and 8.4.E6.
    if degrees_of_freedom == 2:
        return -2.0 * math.log1p(-probability)
    if degrees_of_freedom == 1:
        if probability <= 1e-8:
            # The leading omitted relative correction is pi*p**2/6. Multiply
            # in this order to retain representable subnormal quantiles.
            return probability * (math.pi / 2.0 * probability)
        if probability >= 0.1:
            normal = normal_quantile((1.0 - probability) / 2.0)
            return normal * normal
    if probability <= 1e-50:
        return _chi_square_lower_quantile(probability, degrees_of_freedom)

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

        if abs(residual) <= target_tail * 5e-14:
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


def _chi_square_lower_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Invert tiny tails for df >= 3 without underflowing CDFs or Newton steps."""
    shape = degrees_of_freedom / 2.0
    log_target = math.log(probability)
    log_gamma = math.lgamma(shape)
    log_two = math.log(2.0)
    estimate = log_two + (log_target + math.lgamma(shape + 1.0)) / shape
    lower = math.log(math.ulp(0.0))
    upper = math.log(degrees_of_freedom)

    for _ in range(_QUANTILE_MAX_ITERATIONS):
        log_half = estimate - log_two
        half_value = math.exp(log_half)
        series = _gamma_lower_series(shape, half_value)
        log_cdf = -half_value + shape * log_half - log_gamma + math.log(series)
        residual = log_cdf - log_target
        # d(log CDF) / d(log quantile) = 1 / series.
        step = residual * series
        if abs(step) <= 5e-14:
            return math.exp(estimate)
        if residual < 0.0:
            lower = estimate
        else:
            upper = estimate
        candidate = estimate - step
        if not lower < candidate < upper:
            candidate = (lower + upper) / 2.0
        if candidate == estimate:
            return math.exp(estimate)
        estimate = candidate
    raise ArithmeticError("chi-square lower-tail quantile did not converge")


def chi_square_quantiles(probabilities: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Evaluate array quantiles, validating once and vectorizing the exact df=2 case."""
    degrees_of_freedom = _validate_degrees_of_freedom(degrees_of_freedom)
    probability_array = np.asarray(probabilities, dtype=float)
    if not (
        np.min(probability_array, initial=0.0) >= 0.0
        and np.max(probability_array, initial=1.0) <= 1.0
    ):
        raise ValueError("probability must be a finite value between 0 and 1")
    if degrees_of_freedom == 2:
        result = np.empty_like(probability_array)
        np.negative(probability_array, out=result)
        with np.errstate(divide="ignore"):
            np.log1p(result, out=result)
        result *= -2.0
        return result
    flat_result = np.fromiter(
        (_chi_square_quantile(float(item), degrees_of_freedom) for item in probability_array.flat),
        dtype=float,
        count=probability_array.size,
    )
    return flat_result.reshape(probability_array.shape)
