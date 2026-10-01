"""Stable component calibration and bounded weighted score reductions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

_REDUCTION_BATCH_ROWS = 8192
_MIN_EXPONENT = -4096
_ROUNDING_TOLERANCE = 8 * np.finfo(float).eps


@dataclass(frozen=True)
class _Calibration:
    exponent: int
    anchor: float
    center: float
    deviation: float

    def apply(self, scores: np.ndarray) -> np.ndarray:
        values = np.array(scores, dtype=float, copy=True)
        if self.deviation == 0:
            values[~np.isnan(values)] = 0.0
        else:
            with np.errstate(under="ignore"):
                np.ldexp(values, -self.exponent, out=values)
            if self.anchor:
                values -= self.anchor
            values -= self.center
            values /= self.deviation
        return values

    def scalar(self, value: float) -> float:
        if self.deviation == 0:
            return 0.0
        return ((math.ldexp(value, -self.exponent) - self.anchor) - self.center) / self.deviation


def _score_availability(scores: np.ndarray) -> np.ndarray | None:
    """Return one observed-score mask, or None when every score is available."""
    missing: np.ndarray = np.isnan(scores)
    if not np.any(missing):
        return None
    np.logical_not(missing, out=missing)
    return missing


def _prepare_scores(
    scores: np.ndarray,
) -> tuple[np.ndarray, _Calibration | None, float, np.ndarray | None]:
    available = _score_availability(scores)
    n_valid = len(scores) if available is None else int(np.count_nonzero(available))
    if n_valid <= 1:
        observed = scores if available is None else scores[available]
        return scores, None, abs(float(observed[0])) if n_valid else 0.0, available

    complete = available is None
    observed = (
        np.array(scores, dtype=float, copy=True)
        if complete
        else np.asarray(scores[available], dtype=float)
    )
    lower, upper = float(np.min(observed)), float(np.max(observed))
    if lower == upper:
        calibration = _Calibration(0, 0.0, 0.0, 0.0)
        observed.fill(0.0)
        bound = 0.0
    else:
        _, exponent = math.frexp(max(abs(lower), abs(upper)))
        with np.errstate(under="ignore"):
            np.ldexp(observed, -exponent, out=observed)
        nearby = (lower > 0 and lower >= upper / 2) or (upper < 0 and upper <= lower / 2)
        anchor = float(observed[0]) if nearby else 0.0
        if anchor:
            observed -= anchor
        center = float(np.mean(observed))
        observed -= center
        deviation = math.sqrt(float(np.einsum("i,i->", observed, observed)) / n_valid)
        observed /= deviation
        calibration = _Calibration(exponent, anchor, center, deviation)
        bound = max(abs(calibration.scalar(lower)), abs(calibration.scalar(upper)))
    if complete:
        prepared = observed
    else:
        prepared = np.full(len(scores), np.nan)
        prepared[available] = observed
    if not math.isfinite(bound):
        # Scorers can return an unrepresentable magnitude; calibration may omit it.
        available = _score_availability(prepared)
    return prepared, calibration, bound, available


def standardize_index_scores(scores: np.ndarray) -> np.ndarray:
    """Standardize observations without mutating inputs or changing sparse policy."""
    return _prepare_scores(scores)[0]


def _cancellation_mask(result: np.ndarray, magnitude: float) -> np.ndarray | None:
    """Find near-zero totals using boolean buffers instead of a full float copy."""
    if magnitude == 0:
        return None
    limit = _ROUNDING_TOLERANCE * magnitude
    cancellation = result <= limit
    cancellation &= result >= -limit
    return cancellation if np.any(cancellation) else None


@dataclass(frozen=True)
class _ScoreComponent:
    scores: np.ndarray
    calibration: _Calibration | None
    weight: float
    multiplier: float

    def values(self, selection: slice | np.ndarray) -> np.ndarray:
        selected = self.scores[selection]
        values = self.calibration.apply(selected) if self.calibration is not None else selected
        return values if self.multiplier == 1.0 else values * self.multiplier

    def scalar(self, row: int) -> float:
        value = float(self.scores[row])
        if math.isnan(value):
            return value
        if self.calibration is not None:
            value = self.calibration.scalar(value)
        return value * self.multiplier


def _exact_mean(components: list[_ScoreComponent], row: int) -> float:
    numerator = Fraction(0)
    denominator = Fraction(0)
    for component in components:
        value = component.scalar(row)
        if math.isnan(value):
            continue
        weight = Fraction(component.weight)
        numerator += Fraction(value) * weight
        denominator += weight
    return float(numerator / denominator)


def _range_error(method: str, row: int) -> ValueError:
    return ValueError(
        f"weighted composite {method} is outside the finite float range at respondent index {row}; "
        "reduce weights or use method='mean'"
    )


def _exact_reduction(
    components: list[_ScoreComponent], row: int, method: Literal["mean", "sum", "max"]
) -> float:
    if method == "mean":
        return _exact_mean(components, row)
    terms = [
        Fraction(value) * Fraction(component.weight)
        for component in components
        if not math.isnan(value := component.scalar(row))
    ]
    try:
        return float(sum(terms) if method == "sum" else max(terms))
    except OverflowError as error:
        raise _range_error(method, row) from error


def _scaled_linear_block(
    components: list[_ScoreComponent],
    selection: slice | np.ndarray,
    size: int,
    method: Literal["mean", "sum"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale numerator products and positive weight totals independently."""
    product_exponents = np.full(size, _MIN_EXPONENT, dtype=np.int32)
    weight_exponents = product_exponents.copy()
    counts = np.zeros(size, dtype=np.int_)
    lower = np.full(size, np.nan)
    upper = lower.copy()
    for component in components:
        values = component.values(selection)
        available = ~np.isnan(values)
        counts += available
        _, exponent = math.frexp(component.weight)
        np.maximum(weight_exponents, exponent, out=weight_exponents, where=available)
        _, value_exponents = np.frexp(values)
        np.maximum(
            product_exponents,
            value_exponents + exponent,
            out=product_exponents,
            where=available & (values != 0),
        )
        np.fmin(lower, values, out=lower)
        np.fmax(upper, values, out=upper)

    numerator = np.zeros(size)
    denominator = numerator.copy()
    magnitude = numerator.copy()
    with np.errstate(under="ignore"):
        for component in components:
            values = component.values(selection)
            available = ~np.isnan(values)
            mantissa, exponent = math.frexp(component.weight)
            value_mantissas, value_exponents = np.frexp(values)
            value_mantissas *= mantissa
            np.ldexp(
                value_mantissas,
                value_exponents + exponent - product_exponents,
                out=value_mantissas,
            )
            np.add(numerator, value_mantissas, out=numerator, where=available)
            np.abs(value_mantissas, out=value_mantissas)
            np.add(magnitude, value_mantissas, out=magnitude, where=available)
            mass = np.zeros(size)
            np.ldexp(mantissa, exponent - weight_exponents, out=mass, where=available)
            denominator += mass
        result = numerator.copy()
        if method == "mean":
            np.divide(numerator, denominator, out=result, where=counts > 0)
        exponents = product_exponents - weight_exponents if method == "mean" else product_exponents
        with np.errstate(over="ignore"):
            np.ldexp(result, exponents, out=result)

    nonempty = counts > 0
    constant = nonempty & ((lower == upper) if method == "mean" else (magnitude == 0))
    if method == "mean":
        result[constant] = lower[constant]
    repair = (
        nonempty
        & ~constant
        & (
            ~np.isfinite(result)
            | (np.abs(result) < np.finfo(float).tiny)
            | (np.abs(numerator) <= _ROUNDING_TOLERANCE * magnitude)
        )
    )
    if len(components) == 2 and components[0].weight == components[1].weight:
        # Equal and opposite contributions cancel exactly, including decimal weights.
        opposite = components[0].values(selection) == -components[1].values(selection)
        exact_zero = nonempty & opposite
        result[exact_zero] = 0.0
        repair[exact_zero] = False
    result[~nonempty] = np.nan if method == "mean" else 0.0
    return result, counts, repair


def _scaled_max_block(
    components: list[_ScoreComponent], selection: slice | np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare product signs and exponents before converting the winning value."""
    best_signs = np.full(size, -2, dtype=np.int8)
    best_mantissas = np.zeros(size)
    best_exponents = np.zeros(size, dtype=np.int32)
    counts = np.zeros(size, dtype=np.int_)
    for component in components:
        values = component.values(selection)
        available = ~np.isnan(values)
        counts += available
        weight_mantissa, weight_exponent = math.frexp(component.weight)
        mantissas, exponents = np.frexp(values)
        mantissas *= weight_mantissa
        mantissas[~available] = 0.0
        mantissas, shifts = np.frexp(mantissas)
        exponents += weight_exponent + shifts
        signs = np.sign(mantissas).astype(np.int8)
        greater_exponent = np.where(
            signs > 0, exponents > best_exponents, exponents < best_exponents
        )
        better = available & (
            (signs > best_signs)
            | (
                (signs == best_signs)
                & (signs != 0)
                & (
                    greater_exponent
                    | ((exponents == best_exponents) & (mantissas > best_mantissas))
                )
            )
        )
        best_signs[better] = signs[better]
        best_mantissas[better] = mantissas[better]
        best_exponents[better] = exponents[better]
    with np.errstate(over="ignore", under="ignore"):
        result = np.ldexp(best_mantissas, best_exponents)
    nonempty = counts > 0
    repair = nonempty & (
        ~np.isfinite(result) | ((np.abs(result) < np.finfo(float).tiny) & (best_mantissas != 0))
    )
    result[~nonempty] = np.nan
    return result, counts, repair


def _repair_reductions(
    components: list[_ScoreComponent],
    result: np.ndarray,
    counts_out: np.ndarray | None,
    mask: np.ndarray | None = None,
    *,
    method: Literal["mean", "sum", "max"] = "mean",
    min_valid_indices: int | None = None,
) -> None:
    for start in range(0, len(result), _REDUCTION_BATCH_ROWS):
        stop = min(start + _REDUCTION_BATCH_ROWS, len(result))
        selection: slice | np.ndarray = slice(start, stop)
        if mask is not None:
            selection = np.flatnonzero(mask[start:stop]) + start
            if not len(selection):
                continue
        size = stop - start if isinstance(selection, slice) else len(selection)
        values, counts, repair = (
            _scaled_max_block(components, selection, size)
            if method == "max"
            else _scaled_linear_block(components, selection, size, method)
        )
        if min_valid_indices is not None:
            eligible = counts >= min_valid_indices
            repair &= eligible
            values[~eligible] = np.nan
        for position in np.flatnonzero(repair):
            row = (
                start + int(position) if isinstance(selection, slice) else int(selection[position])
            )
            values[position] = _exact_reduction(components, row, method)
        result[selection] = values
        if counts_out is not None:
            counts_out[selection] = counts


def combine_mean_scores(
    scores: dict[str, np.ndarray],
    standardize: bool,
    weights: Mapping[str, float] | None,
    min_valid_indices: int | None,
    valid_counts_out: np.ndarray | None,
    multipliers: Mapping[str, float] | None,
) -> np.ndarray:
    """Retain the fast ordinary mean and repair unstable reductions in bounded blocks."""
    if len(scores) == 1:
        name, original = next(iter(scores.items()))
        values = standardize_index_scores(original) if standardize else original
        result = values if values is not original else np.array(values, dtype=float, copy=True)
        multiplier = multipliers.get(name, 1.0) if multipliers is not None else 1.0
        if multiplier != 1.0:
            result *= multiplier
        if valid_counts_out is not None:
            valid_counts_out[:] = ~np.isnan(original)
        if min_valid_indices is not None and min_valid_indices > 1:
            result.fill(np.nan)
        return result

    n_rows = len(next(iter(scores.values())))
    counts = valid_counts_out
    if counts is None and min_valid_indices is not None:
        counts = np.zeros(n_rows, dtype=np.int_)
    denominator = (
        counts
        if weights is None and counts is not None
        else np.zeros(n_rows, dtype=float if weights is not None else np.int_)
    )
    result = np.zeros(n_rows)
    weighted = result
    components: list[_ScoreComponent] = []
    weight_scale = max(weights.get(name, 1.0) for name in scores) if weights is not None else 1.0
    failed = weights is not None and any(
        weights.get(name, 1.0) / weight_scale < np.finfo(float).tiny for name in scores
    )
    magnitude = 0.0
    for name, original in scores.items():
        if standardize:
            values, calibration, bound, available = _prepare_scores(original)
        else:
            values, calibration = original, None
            available = _score_availability(values)
            lower, upper = float(np.fmin.reduce(values)), float(np.fmax.reduce(values))
            bound = 0.0 if math.isnan(lower) else max(abs(lower), abs(upper))
        weight = weights.get(name, 1.0) if weights is not None else 1.0
        multiplier = multipliers.get(name, 1.0) if multipliers is not None else 1.0
        components.append(_ScoreComponent(original, calibration, weight, multiplier))
        normalized_weight = weight / weight_scale
        magnitude += bound * normalized_weight * abs(multiplier)
        if failed:
            continue
        weighted = values
        with np.errstate(over="raise", under="raise", invalid="raise"):
            try:
                scale = normalized_weight * multiplier
                if scale != 1.0:
                    if calibration is not None:
                        np.multiply(values, scale, out=values)
                    else:
                        weighted = values * scale
                where = True if available is None else available
                np.add(result, weighted, out=result, where=where)
                if weights is None:
                    denominator += 1 if available is None else available
                else:
                    np.add(denominator, normalized_weight, out=denominator, where=where)
                if counts is not None and counts is not denominator:
                    counts += 1 if available is None else available
            except FloatingPointError:
                failed = True
    del values, weighted, available

    if failed:
        _repair_reductions(components, result, denominator if weights is None else counts)
    else:
        cancellation = _cancellation_mask(result, magnitude)
        if cancellation is not None:
            cancellation &= denominator > 0
        with np.errstate(over="raise", under="raise", invalid="raise"):
            try:
                np.divide(result, denominator, out=result, where=denominator > 0)
            except FloatingPointError:
                failed = True
        if failed:
            _repair_reductions(components, result, denominator if weights is None else counts)
        else:
            result[denominator == 0] = np.nan
            if cancellation is not None and np.any(cancellation):
                _repair_reductions(components, result, counts, cancellation)

    if min_valid_indices is not None:
        available_counts = denominator if weights is None else counts
        assert available_counts is not None
        result[available_counts < min_valid_indices] = np.nan
    return result


def _single_weighted_score(
    scores: dict[str, np.ndarray],
    method: Literal["sum", "max"],
    standardize: bool,
    weights: Mapping[str, float] | None,
    min_valid_indices: int | None,
    valid_counts_out: np.ndarray | None,
    multipliers: Mapping[str, float] | None,
) -> np.ndarray:
    name, original = next(iter(scores.items()))
    if valid_counts_out is not None:
        valid_counts_out[:] = ~np.isnan(original)
    if min_valid_indices is not None and min_valid_indices > 1:
        return np.full(len(original), np.nan)
    values = standardize_index_scores(original) if standardize else original
    result = values if values is not original else np.array(values, dtype=float, copy=True)
    weight = weights.get(name, 1.0) if weights is not None else 1.0
    multiplier = multipliers.get(name, 1.0) if multipliers is not None else 1.0
    with np.errstate(over="raise", under="ignore", invalid="raise"):
        try:
            if weight * multiplier != 1.0:
                result *= weight * multiplier
        except FloatingPointError as error:
            row = int(np.flatnonzero(np.isinf(result))[0])
            raise _range_error(method, row) from error
    if method == "sum" and min_valid_indices is None:
        result[np.isnan(result)] = 0.0
    return result


def combine_sum_max_scores(
    scores: dict[str, np.ndarray],
    method: Literal["sum", "max"],
    standardize: bool,
    weights: Mapping[str, float] | None,
    min_valid_indices: int | None,
    valid_counts_out: np.ndarray | None,
    multipliers: Mapping[str, float] | None,
) -> np.ndarray:
    """Reduce ordinary scores directly and repair intermediate overflow without clamping."""
    if len(scores) == 1:
        return _single_weighted_score(
            scores, method, standardize, weights, min_valid_indices, valid_counts_out, multipliers
        )
    n_rows = len(next(iter(scores.values())))
    counts = valid_counts_out
    if counts is None and min_valid_indices is not None:
        counts = np.zeros(n_rows, dtype=np.int_)
    result = np.zeros(n_rows) if method == "sum" else np.full(n_rows, np.nan)
    presence: bool | np.ndarray = False
    weight_scale = (
        max(weights.get(name, 1.0) for name in scores)
        if method == "sum" and weights is not None
        else 1.0
    )
    failed = (
        method == "sum"
        and weights is not None
        and any(weights.get(name, 1.0) / weight_scale < np.finfo(float).tiny for name in scores)
    )
    components: list[_ScoreComponent] = []
    magnitude = 0.0
    weighted = result
    for name, original in scores.items():
        if standardize:
            values, calibration, bound, available = _prepare_scores(original)
        else:
            values, calibration, bound = original, None, 0.0
            available = (
                _score_availability(values) if method == "sum" or counts is not None else None
            )
            if method == "sum" and not failed:
                lower, upper = float(np.fmin.reduce(values)), float(np.fmax.reduce(values))
                bound = 0.0 if math.isnan(lower) else max(abs(lower), abs(upper))
        weight = weights.get(name, 1.0) if weights is not None else 1.0
        multiplier = multipliers.get(name, 1.0) if multipliers is not None else 1.0
        components.append(_ScoreComponent(original, calibration, weight, multiplier))
        normalized_weight = weight / weight_scale
        magnitude += bound * normalized_weight * abs(multiplier)
        if failed:
            continue
        weighted = values
        with np.errstate(
            over="raise", under="raise" if method == "sum" else "ignore", invalid="raise"
        ):
            try:
                scale = normalized_weight * multiplier
                if scale != 1.0:
                    if calibration is not None:
                        np.multiply(values, scale, out=values)
                    else:
                        weighted = values * scale
                if method == "sum":
                    np.add(
                        result, weighted, out=result, where=True if available is None else available
                    )
                else:
                    np.fmax(result, weighted, out=result)
                if counts is not None:
                    counts += 1 if available is None else available
                elif method == "sum" and presence is not True:
                    if available is None:
                        presence = True
                    elif isinstance(presence, np.ndarray):
                        np.logical_or(presence, available, out=presence)
                    else:
                        presence = available.copy()
            except FloatingPointError:
                failed = True
    del values, weighted, available

    if failed:
        _repair_reductions(
            components, result, counts, method=method, min_valid_indices=min_valid_indices
        )
    elif method == "sum":
        cancellation = _cancellation_mask(result, magnitude)
        if cancellation is not None:
            if counts is not None:
                cancellation &= counts > 0
            elif presence is not True:
                cancellation &= presence
        with np.errstate(over="raise", under="raise", invalid="raise"):
            try:
                if weight_scale != 1.0:
                    result *= weight_scale
            except FloatingPointError:
                failed = True
        if failed:
            _repair_reductions(
                components, result, counts, method=method, min_valid_indices=min_valid_indices
            )
        elif cancellation is not None and np.any(cancellation):
            _repair_reductions(
                components,
                result,
                counts,
                cancellation,
                method=method,
                min_valid_indices=min_valid_indices,
            )
    if min_valid_indices is not None:
        assert counts is not None
        result[counts < min_valid_indices] = np.nan
    return result
