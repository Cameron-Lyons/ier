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
_BATCH_QUANTILE_MAX_ITERATIONS = 128
_QUANTILE_BATCH_ELEMENTS = 8_192


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


def _regularized_gamma_pairs(shape: float, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate regularized incomplete-gamma tails for one bounded value batch."""
    lower = np.empty_like(values)
    upper = np.empty_like(values)
    log_scale = -values + shape * np.log(values) - math.lgamma(shape)
    series_mask = values < shape + 1.0

    if np.any(series_mask):
        series_values = values[series_mask]
        term = np.full(len(series_values), 1.0 / shape)
        series = term.copy()
        denominator = shape
        for _ in range(_GAMMA_MAX_ITERATIONS):
            denominator += 1.0
            term *= series_values / denominator
            series += term
            if np.all(np.abs(term) <= np.abs(series) * _GAMMA_EPSILON):
                series_lower = np.clip(
                    series * np.exp(log_scale[series_mask]),
                    0.0,
                    1.0,
                )
                lower[series_mask] = series_lower
                upper[series_mask] = 1.0 - series_lower
                break
        else:
            raise ArithmeticError("regularized gamma series did not converge")

    fraction_mask = ~series_mask
    if np.any(fraction_mask):
        fraction_values = values[fraction_mask]
        denominator_values = fraction_values + 1.0 - shape
        denominator_values = np.where(
            np.abs(denominator_values) < _CONTINUED_FRACTION_FLOOR,
            _CONTINUED_FRACTION_FLOOR,
            denominator_values,
        )
        reciprocal_previous = np.full(
            len(fraction_values),
            1.0 / _CONTINUED_FRACTION_FLOOR,
        )
        reciprocal_current = 1.0 / denominator_values
        fraction = reciprocal_current.copy()

        for iteration in range(1, _GAMMA_MAX_ITERATIONS + 1):
            coefficient = -float(iteration) * (float(iteration) - shape)
            denominator_values += 2.0
            reciprocal_current = coefficient * reciprocal_current + denominator_values
            reciprocal_current = np.where(
                np.abs(reciprocal_current) < _CONTINUED_FRACTION_FLOOR,
                _CONTINUED_FRACTION_FLOOR,
                reciprocal_current,
            )
            reciprocal_previous = denominator_values + coefficient / reciprocal_previous
            reciprocal_previous = np.where(
                np.abs(reciprocal_previous) < _CONTINUED_FRACTION_FLOOR,
                _CONTINUED_FRACTION_FLOOR,
                reciprocal_previous,
            )
            reciprocal_current = 1.0 / reciprocal_current
            change = reciprocal_previous * reciprocal_current
            fraction *= change
            if np.all(np.abs(change - 1.0) <= _GAMMA_EPSILON):
                fraction_upper = np.clip(
                    np.exp(log_scale[fraction_mask]) * fraction,
                    0.0,
                    1.0,
                )
                upper[fraction_mask] = fraction_upper
                lower[fraction_mask] = 1.0 - fraction_upper
                break
        else:
            raise ArithmeticError("regularized gamma continued fraction did not converge")

    return lower, upper


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
    """Return chi-square quantiles in bounded vectorized solver batches."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")

    probability_array = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probability_array).all() or np.any(
        (probability_array < 0.0) | (probability_array > 1.0)
    ):
        raise ValueError("probability must be a finite value between 0 and 1")

    flat_probabilities = probability_array.ravel()
    flat_result = np.empty_like(flat_probabilities)
    for start in range(0, len(flat_probabilities), _QUANTILE_BATCH_ELEMENTS):
        stop = min(start + _QUANTILE_BATCH_ELEMENTS, len(flat_probabilities))
        probability_batch = flat_probabilities[start:stop]
        result_batch = flat_result[start:stop]
        at_zero = probability_batch == 0.0
        at_one = probability_batch == 1.0
        interior = ~(at_zero | at_one)
        result_batch[at_zero] = 0.0
        result_batch[at_one] = math.inf
        if np.any(interior):
            interior_probabilities = probability_batch[interior]
            try:
                result_batch[interior] = _chi_square_quantile_batch(
                    interior_probabilities,
                    degrees_of_freedom,
                )
            except ArithmeticError:
                result_batch[interior] = np.fromiter(
                    (
                        chi_square_quantile(float(probability), degrees_of_freedom)
                        for probability in interior_probabilities
                    ),
                    dtype=float,
                    count=len(interior_probabilities),
                )
    return flat_result.reshape(probability_array.shape)


def _chi_square_quantile_batch(
    probabilities: np.ndarray,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Solve one interior-probability batch with safeguarded array operations."""
    df = float(degrees_of_freedom)
    normal = np.fromiter(
        (normal_quantile(float(item)) for item in probabilities),
        dtype=float,
        count=len(probabilities),
    )
    correction = 1.0 - 2.0 / (9.0 * df) + normal * math.sqrt(2.0 / (9.0 * df))
    wilson_hilferty = np.where(correction > 0.0, df * correction**3, 0.0)
    half_df = df / 2.0
    lower_tail_guess = 2.0 * np.exp((np.log(probabilities) + math.lgamma(half_df + 1.0)) / half_df)
    estimates = np.where(wilson_hilferty > 0.0, wilson_hilferty, lower_tail_guess)
    estimates = np.maximum(estimates, np.finfo(float).tiny)

    lower = np.zeros(len(probabilities))
    upper = np.maximum(np.maximum(df, estimates), 1.0)
    lower_tail = probabilities <= 0.5
    target_tail = np.where(lower_tail, probabilities, 1.0 - probabilities)

    while True:
        cdf, survival = _regularized_gamma_pairs(half_df, upper / 2.0)
        below_target = np.where(
            lower_tail,
            cdf < probabilities,
            survival > target_tail,
        )
        if not np.any(below_target):
            break
        upper[below_target] *= 2.0
        if np.isinf(upper).any():
            raise ArithmeticError("could not bracket chi-square quantile")

    values: np.ndarray = np.minimum(estimates, upper)
    converged = np.zeros(len(probabilities), dtype=bool)
    density_log_constant = -half_df * math.log(2.0) - math.lgamma(half_df)

    for _ in range(_BATCH_QUANTILE_MAX_ITERATIONS):
        active = np.flatnonzero(~converged)
        if active.size == 0:
            return values

        active_values = values[active]
        cdf, survival = _regularized_gamma_pairs(half_df, active_values / 2.0)
        active_lower_tail = lower_tail[active]
        residual = np.where(
            active_lower_tail,
            cdf - probabilities[active],
            survival - target_tail[active],
        )
        residual_converged = np.abs(residual) <= np.maximum(
            target_tail[active] * 5e-14,
            1e-300,
        )
        if np.any(residual_converged):
            converged[active[residual_converged]] = True

        working = ~residual_converged
        if not np.any(working):
            continue
        working_indices = active[working]
        working_values = active_values[working]
        working_residual = residual[working]
        below_target = np.where(
            active_lower_tail[working],
            working_residual < 0.0,
            working_residual > 0.0,
        )
        lower[working_indices] = np.where(
            below_target,
            working_values,
            lower[working_indices],
        )
        upper[working_indices] = np.where(
            below_target,
            upper[working_indices],
            working_values,
        )

        density = np.exp(
            (half_df - 1.0) * np.log(working_values) - working_values / 2.0 + density_log_constant
        )
        derivative = np.where(active_lower_tail[working], density, -density)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            candidate = working_values - working_residual / derivative
        midpoint = (lower[working_indices] + upper[working_indices]) / 2.0
        invalid_candidate = (
            ~np.isfinite(candidate)
            | (candidate <= lower[working_indices])
            | (candidate >= upper[working_indices])
        )
        candidate = np.where(invalid_candidate, midpoint, candidate)
        step_converged = np.abs(candidate - working_values) <= np.maximum(
            np.abs(working_values) * 5e-14,
            np.finfo(float).tiny,
        )
        values[working_indices] = candidate
        if np.any(step_converged):
            converged[working_indices[step_converged]] = True

    for index in np.flatnonzero(~converged):
        values[index] = chi_square_quantile(float(probabilities[index]), degrees_of_freedom)
    return values
