"""Stable component calibration and bounded weighted-mean reductions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

_MEAN_BATCH_ROWS = 8192
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


def _prepare_scores(scores: np.ndarray) -> tuple[np.ndarray, _Calibration | None, float]:
    available = ~np.isnan(scores)
    n_valid = int(np.count_nonzero(available))
    if n_valid <= 1:
        observed = scores[available]
        return scores, None, abs(float(observed[0])) if n_valid else 0.0

    complete = n_valid == len(scores)
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
        return observed, calibration, bound
    prepared = np.full(len(scores), np.nan)
    prepared[available] = observed
    return prepared, calibration, bound


def standardize_index_scores(scores: np.ndarray) -> np.ndarray:
    """Standardize observations without mutating inputs or changing sparse policy."""
    return _prepare_scores(scores)[0]


@dataclass(frozen=True)
class _MeanComponent:
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


def _exact_mean(components: list[_MeanComponent], row: int) -> float:
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


def _scaled_mean_block(
    components: list[_MeanComponent], selection: slice | np.ndarray, size: int
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
        result = np.zeros(size)
        np.divide(numerator, denominator, out=result, where=counts > 0)
        with np.errstate(over="ignore"):
            np.ldexp(result, product_exponents - weight_exponents, out=result)

    nonempty = counts > 0
    constant = nonempty & (lower == upper)
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
    result[~nonempty] = np.nan
    return result, counts, repair


def _repair_means(
    components: list[_MeanComponent],
    result: np.ndarray,
    counts_out: np.ndarray | None,
    mask: np.ndarray | None = None,
) -> None:
    for start in range(0, len(result), _MEAN_BATCH_ROWS):
        stop = min(start + _MEAN_BATCH_ROWS, len(result))
        selection: slice | np.ndarray = slice(start, stop)
        if mask is not None:
            selection = np.flatnonzero(mask[start:stop]) + start
            if not len(selection):
                continue
        size = stop - start if isinstance(selection, slice) else len(selection)
        values, counts, repair = _scaled_mean_block(components, selection, size)
        for position in np.flatnonzero(repair):
            row = (
                start + int(position) if isinstance(selection, slice) else int(selection[position])
            )
            values[position] = _exact_mean(components, row)
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
    components: list[_MeanComponent] = []
    weight_scale = max(weights.get(name, 1.0) for name in scores) if weights is not None else 1.0
    failed = weights is not None and any(
        weights.get(name, 1.0) / weight_scale < np.finfo(float).tiny for name in scores
    )
    magnitude = 0.0
    for name, original in scores.items():
        if standardize:
            values, calibration, bound = _prepare_scores(original)
        else:
            values, calibration = original, None
            lower, upper = float(np.fmin.reduce(values)), float(np.fmax.reduce(values))
            bound = 0.0 if math.isnan(lower) else max(abs(lower), abs(upper))
        weight = weights.get(name, 1.0) if weights is not None else 1.0
        multiplier = multipliers.get(name, 1.0) if multipliers is not None else 1.0
        components.append(_MeanComponent(original, calibration, weight, multiplier))
        normalized_weight = weight / weight_scale
        magnitude += bound * normalized_weight * abs(multiplier)
        if failed:
            continue
        available = ~np.isnan(values)
        weighted = values
        with np.errstate(over="raise", under="raise", invalid="raise"):
            try:
                scale = normalized_weight * multiplier
                if scale != 1.0:
                    if calibration is not None:
                        np.multiply(values, scale, out=values)
                    else:
                        weighted = values * scale
                np.add(result, weighted, out=result, where=available)
                if weights is None:
                    denominator += available
                else:
                    np.add(denominator, normalized_weight, out=denominator, where=available)
                if counts is not None and counts is not denominator:
                    counts += available
            except FloatingPointError:
                failed = True
    del values, weighted

    if failed:
        _repair_means(components, result, denominator if weights is None else counts)
    else:
        cancellation = (np.abs(result) <= _ROUNDING_TOLERANCE * magnitude) & (denominator > 0)
        with np.errstate(over="raise", under="raise", invalid="raise"):
            try:
                np.divide(result, denominator, out=result, where=denominator > 0)
            except FloatingPointError:
                failed = True
        if failed:
            _repair_means(components, result, denominator if weights is None else counts)
        else:
            result[denominator == 0] = np.nan
            if magnitude > 0 and np.any(cancellation):
                _repair_means(components, result, counts, cancellation)

    if min_valid_indices is not None:
        available_counts = denominator if weights is None else counts
        assert available_counts is not None
        result[available_counts < min_valid_indices] = np.nan
    return result
