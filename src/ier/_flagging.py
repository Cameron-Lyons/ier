"""Shared helpers for percentile/threshold-based flagging."""

import math
from fractions import Fraction
from typing import Any

import numpy as np

from ier.types import FlagDirection

_MIN_NORMAL = np.finfo(float).tiny


def _finite_number(value: Any, message: str) -> float:
    """Return one finite numeric option after rejecting Boolean and array lookalikes."""
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value[()]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    """Select a linear sample percentile in one owned observation buffer."""
    if scores.size == 0:
        return 0.0
    if percentile in (0.0, 100.0):
        endpoint = float(
            np.fmin.reduce(scores, axis=None)
            if percentile == 0.0
            else np.fmax.reduce(scores, axis=None)
        )
        return 0.0 if math.isnan(endpoint) else endpoint

    missing = np.isnan(scores)
    n_missing = int(np.count_nonzero(missing))
    if n_missing == scores.size:
        return 0.0
    if n_missing:
        np.logical_not(missing, out=missing)
        observed: np.ndarray = scores[missing]
        del missing
    else:
        del missing
        observed = scores.flatten()
    position = (observed.size - 1) * (percentile / 100.0)
    lower_index = math.floor(position)
    fraction = position - lower_index
    if fraction == 0.0:
        observed.partition(lower_index)
        return float(observed[lower_index])

    observed.partition((lower_index, lower_index + 1))
    lower, upper = observed[lower_index : lower_index + 2]
    if observed.dtype.kind in "iu":
        # Preserve integer differences and round only the interpolated cutoff.
        return float(Fraction(int(lower)) + (int(upper) - int(lower)) * Fraction(fraction))
    left, right = float(lower), float(upper)
    difference = right - left
    if not math.isfinite(difference) or 0 < difference < _MIN_NORMAL:
        # Exact scalar arithmetic repairs overflowing spans and subnormal ties.
        return float(Fraction(left) + (Fraction(right) - Fraction(left)) * Fraction(fraction))
    return (
        right - difference * (1.0 - fraction) if fraction >= 0.5 else left + difference * fraction
    )


def validate_percentile(percentile: float) -> float:
    """Return a finite percentile in ``[0, 100]`` or raise ``ValueError``."""
    message = "percentile must be a finite number between 0 and 100"
    result = _finite_number(percentile, message)
    if not 0.0 <= result <= 100.0:
        raise ValueError(message)
    return result


def validate_threshold(threshold: float | None) -> float | None:
    """Return a finite optional threshold or raise ``ValueError``."""
    if threshold is None:
        return None
    return _finite_number(threshold, "threshold must be a finite number")


def resolve_threshold(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float,
) -> float:
    """Resolve an explicit threshold or derive one from valid scores."""
    validated_percentile = validate_percentile(percentile)
    validated_threshold = validate_threshold(threshold)
    if validated_threshold is not None:
        return validated_threshold

    return _percentile_threshold(scores, validated_percentile)


def threshold_flags(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float,
    direction: FlagDirection,
    inclusive: bool | None = None,
) -> np.ndarray:
    """Create flags, including fixed-cutoff equality but excluding percentile ties."""
    if inclusive is None:
        inclusive = threshold is not None
    cutoff = resolve_threshold(scores, threshold, percentile)
    comparison_cutoff: float | np.ndarray = cutoff
    if scores.dtype.kind in "iu":
        # Compare integers to an exact integer boundary instead of rounding scores.
        round_up = inclusive if direction == "high" else not inclusive
        boundary = math.ceil(cutoff) if round_up else math.floor(cutoff)
        limits = np.iinfo(scores.dtype)
        if boundary < int(limits.min):
            return np.full(scores.shape, direction == "high", dtype=bool)
        if boundary > int(limits.max):
            return np.full(scores.shape, direction == "low", dtype=bool)
        comparison_cutoff = np.array([boundary], dtype=scores.dtype)
    elif scores.dtype.kind == "f" and scores.dtype.itemsize < 8:
        # A vector cutoff promotes float32 comparisons consistently in NumPy 1/2.
        comparison_cutoff = np.array([cutoff])

    if direction == "high":
        return scores >= comparison_cutoff if inclusive else scores > comparison_cutoff
    return scores <= comparison_cutoff if inclusive else scores < comparison_cutoff
