"""Small statistical primitives used internally by IER.

The implementations here intentionally cover only the narrow operations the
package needs.  Keeping them local avoids making a large general-purpose
statistics library a runtime dependency.
"""

from __future__ import annotations

import math
from operator import index
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

_GAMMA_EPSILON = 1e-14
_GAMMA_MAX_ITERATIONS = 10_000
_CONTINUED_FRACTION_FLOOR = 1e-300
_QUANTILE_MAX_ITERATIONS = 128
_QUANTILE_BATCH_ELEMENTS = 8_192
# Array Newton iterations overtake scalar solves near 100-250 general
# probabilities (NumPy 2.x sooner, NumPy 1.26 later); smaller sets solve singly.
_QUANTILE_ARRAY_MIN_ELEMENTS = 192
_TINY = float(np.finfo(float).tiny)


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
    estimate = max(estimate, _TINY)

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

    value = min(max(estimate, _TINY), upper)
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

        if abs(candidate - value) <= max(abs(value) * 5e-14, _TINY):
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
    """Evaluate array quantiles, validating once and solving each batch together.

    Results equal ``chi_square_quantile`` element by element. Bounded batches
    take the same special cases, and the general Newton solve runs on arrays
    unless a batch has too few general probabilities to repay array overhead.
    """
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
    flat_result = np.empty(probability_array.size)
    # Each result is independent of its batch, so batches only bound workspace.
    for start in range(0, probability_array.size, _QUANTILE_BATCH_ELEMENTS):
        stop = start + _QUANTILE_BATCH_ELEMENTS
        flat_result[start:stop] = _chi_square_quantile_batch(
            probability_array.flat[start:stop], degrees_of_freedom
        )
    return flat_result.reshape(probability_array.shape)


def _chi_square_quantile_batch(probabilities: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Route validated probabilities through the scalar special cases for df != 2."""
    result = np.zeros(probabilities.shape)
    result[probabilities == 1.0] = math.inf
    general = (probabilities > 0.0) & (probabilities < 1.0)
    if degrees_of_freedom == 1:
        small = general & (probabilities <= 1e-8)
        selected = probabilities[small]
        with np.errstate(under="ignore"):
            result[small] = selected * (math.pi / 2.0 * selected)
        large = general & (probabilities >= 0.1)
        normal = _libm(NormalDist().inv_cdf, (1.0 - probabilities[large]) / 2.0)
        result[large] = normal * normal
        general &= ~(small | large)
    else:
        tiny = general & (probabilities <= 1e-50)
        result[tiny] = [
            _chi_square_lower_quantile(probability, degrees_of_freedom)
            for probability in probabilities[tiny].tolist()
        ]
        general &= ~tiny
    selected = probabilities[general]
    if selected.size < _QUANTILE_ARRAY_MIN_ELEMENTS:
        # Per-iteration array overhead outweighs scalar solves for few values.
        result[general] = [
            _chi_square_quantile(probability, degrees_of_freedom)
            for probability in selected.tolist()
        ]
    else:
        result[general] = _chi_square_quantiles_general(selected, degrees_of_freedom)
    return result


def _chi_square_quantiles_general(probabilities: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Apply ``_chi_square_quantile``'s safeguarded Newton solve to an array.

    Each element repeats the scalar operations: a Wilson-Hilferty or lower-tail
    start, upper-bracket doubling, then Newton steps with a bisection fallback.
    Finished elements leave the working set, so a result never depends on the
    other probabilities solved with it.
    """
    count = probabilities.size
    result = np.empty(count)
    if not count:
        return result
    df = float(degrees_of_freedom)
    half_df = df / 2.0
    lower_tail = probabilities <= 0.5
    complement = 1.0 - probabilities

    # Python float arithmetic overflows, underflows and propagates NaN silently.
    # Match it so a caller's NumPy error policy cannot change any result.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normal = _libm(NormalDist().inv_cdf, probabilities)
        correction = 1.0 - 2.0 / (9.0 * df) + normal * math.sqrt(2.0 / (9.0 * df))
        # Nonpositive corrections give nonpositive Wilson-Hilferty estimates, so
        # one test selects the lower-tail start exactly where the scalar does.
        estimate = df * _libm(_cube, correction)
        nonpositive = ~(estimate > 0.0)
        if nonpositive.any():
            log_probability = _libm(math.log, probabilities[nonpositive])
            estimate[nonpositive] = 2.0 * _libm(
                math.exp, (log_probability + math.lgamma(half_df + 1.0)) / half_df
            )
        np.maximum(estimate, _TINY, out=estimate)

        # The scalar max(df, estimate, 1.0) and its second _TINY clamp reduce to
        # these because df >= 1 and the estimate is already at least _TINY.
        upper = np.maximum(estimate, df)
        pending = np.arange(count)
        while pending.size:
            cdf, survival = _regularized_gamma_pair_array(half_df, upper[pending] / 2.0)
            below_target = np.where(
                lower_tail[pending], cdf < probabilities[pending], survival > complement[pending]
            )
            pending = pending[below_target]
            doubled = upper[pending] * 2.0
            if np.isinf(doubled).any():
                raise ArithmeticError("could not bracket chi-square quantile")
            upper[pending] = doubled

        positions = np.arange(count)
        value = np.minimum(estimate, upper)
        lower = np.zeros(count)
        target_tail = np.where(lower_tail, probabilities, complement)
        for _ in range(_QUANTILE_MAX_ITERATIONS):
            cdf, survival = _regularized_gamma_pair_array(half_df, value / 2.0)
            residual = np.where(lower_tail, cdf, survival)
            residual -= target_tail
            below_target = np.where(lower_tail, residual < 0.0, residual > 0.0)
            derivative = _chi_square_density_array(value, degrees_of_freedom)
            np.negative(derivative, out=derivative, where=~lower_tail)
            converged = np.abs(residual) <= target_tail * 5e-14

            lower = np.where(below_target, value, lower)
            upper = np.where(below_target, upper, value)
            candidate = np.full_like(value, math.nan)
            usable = (derivative > 0.0) | (derivative < 0.0)
            np.divide(residual, derivative, out=candidate, where=usable)
            np.subtract(value, candidate, out=candidate, where=usable)
            bisect = ~(np.isfinite(candidate) & (lower < candidate) & (candidate < upper))
            candidate[bisect] = (lower[bisect] + upper[bisect]) / 2.0
            stalled = np.abs(candidate - value) <= np.maximum(np.abs(value) * 5e-14, _TINY)

            result[positions[converged]] = value[converged]
            stalled &= ~converged
            result[positions[stalled]] = candidate[stalled]
            remaining = ~(converged | stalled)
            if not remaining.any():
                return result
            positions = positions[remaining]
            value = candidate[remaining]
            lower = lower[remaining]
            upper = upper[remaining]
            lower_tail = lower_tail[remaining]
            target_tail = target_tail[remaining]

    raise ArithmeticError("chi-square quantile did not converge")


def _regularized_gamma_pair_array(
    shape: float, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``_regularized_gamma_pair`` for an array of finite non-negative values.

    Each element repeats the scalar recurrence and operation order. Callers
    suppress floating-point warnings as Python float arithmetic would.
    """
    lower = np.zeros_like(values)
    upper = np.ones_like(values)
    # Zero values (for example halved subnormals) keep the scalar P = 0, Q = 1.
    positive = values > 0.0
    values = values[positive]
    scale = _libm(math.exp, -values + shape * _libm(math.log, values) - math.lgamma(shape))
    positive_lower = np.empty_like(values)
    positive_upper = np.empty_like(values)

    series = values < shape + 1.0
    tail = _gamma_lower_series_array(shape, values[series])
    tail *= scale[series]
    np.clip(tail, 0.0, 1.0, out=tail)
    positive_lower[series] = tail
    positive_upper[series] = 1.0 - tail

    fraction = ~series
    tail = _gamma_upper_fraction_array(shape, values[fraction])
    tail *= scale[fraction]
    np.clip(tail, 0.0, 1.0, out=tail)
    positive_upper[fraction] = tail
    positive_lower[fraction] = 1.0 - tail

    lower[positive] = positive_lower
    upper[positive] = positive_upper
    return lower, upper


def _gamma_lower_series_array(shape: float, values: np.ndarray) -> np.ndarray:
    """Evaluate ``_gamma_lower_series`` elementwise, retiring converged elements."""
    result = np.empty_like(values)
    if not values.size:
        return result
    positions = np.arange(values.size)
    term = np.full_like(values, 1.0 / shape)
    series = term.copy()
    denominator = shape
    for _ in range(_GAMMA_MAX_ITERATIONS):
        denominator += 1.0
        term *= values / denominator
        series += term
        # Terms are non-negative and the series is positive for non-negative
        # values, so the scalar convergence test needs no absolute values here.
        converged = term <= series * _GAMMA_EPSILON
        finished = np.count_nonzero(converged)
        if finished:
            result[positions[converged]] = series[converged]
            if finished == converged.size:
                return result
            remaining = ~converged
            positions = positions[remaining]
            values = values[remaining]
            term = term[remaining]
            series = series[remaining]
    raise ArithmeticError("regularized gamma series did not converge")


def _gamma_upper_fraction_array(shape: float, values: np.ndarray) -> np.ndarray:
    """Evaluate the scalar modified-Lentz upper-gamma fraction elementwise."""
    result = np.empty_like(values)
    if not values.size:
        return result
    positions = np.arange(values.size)
    denominator = values + 1.0 - shape
    np.copyto(
        denominator,
        _CONTINUED_FRACTION_FLOOR,
        where=np.abs(denominator) < _CONTINUED_FRACTION_FLOOR,
    )
    reciprocal_previous = np.full_like(values, 1.0 / _CONTINUED_FRACTION_FLOOR)
    reciprocal_current = 1.0 / denominator
    fraction = reciprocal_current.copy()

    for iteration in range(1, _GAMMA_MAX_ITERATIONS + 1):
        coefficient = -float(iteration) * (float(iteration) - shape)
        denominator += 2.0
        reciprocal_current *= coefficient
        reciprocal_current += denominator
        np.copyto(
            reciprocal_current,
            _CONTINUED_FRACTION_FLOOR,
            where=np.abs(reciprocal_current) < _CONTINUED_FRACTION_FLOOR,
        )
        reciprocal_previous = denominator + coefficient / reciprocal_previous
        np.copyto(
            reciprocal_previous,
            _CONTINUED_FRACTION_FLOOR,
            where=np.abs(reciprocal_previous) < _CONTINUED_FRACTION_FLOOR,
        )
        np.divide(1.0, reciprocal_current, out=reciprocal_current)
        change = reciprocal_previous * reciprocal_current
        fraction *= change
        converged = np.abs(change - 1.0) <= _GAMMA_EPSILON
        finished = np.count_nonzero(converged)
        if finished:
            result[positions[converged]] = fraction[converged]
            if finished == converged.size:
                return result
            remaining = ~converged
            positions = positions[remaining]
            denominator = denominator[remaining]
            reciprocal_previous = reciprocal_previous[remaining]
            reciprocal_current = reciprocal_current[remaining]
            fraction = fraction[remaining]

    raise ArithmeticError("regularized gamma continued fraction did not converge")


def _chi_square_density_array(values: np.ndarray, degrees_of_freedom: int) -> np.ndarray:
    """Return ``_chi_square_density`` for an array of positive finite values."""
    half_df = degrees_of_freedom / 2.0
    log_density = (
        (half_df - 1.0) * _libm(math.log, values)
        - values / 2.0
        - half_df * math.log(2.0)
        - math.lgamma(half_df)
    )
    return _libm(math.exp, log_density)


def _libm(function: Callable[[float], float], values: np.ndarray) -> np.ndarray:
    """Apply a scalar float function elementwise with the scalar path's rounding.

    NumPy's vectorized exponentials and logarithms may differ from the C
    library by an ulp on some platforms. Elementwise arithmetic rounds the same
    in NumPy and Python, so evaluating only these calls through the scalar
    functions keeps array and scalar quantiles identical everywhere.
    """
    return np.fromiter(map(function, values.tolist()), dtype=float, count=values.size)


def _cube(value: float) -> float:
    """Return the Wilson-Hilferty cube with the scalar power operation."""
    return value**3
