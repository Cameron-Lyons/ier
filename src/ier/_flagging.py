"""Shared helpers for percentile/threshold-based flagging."""

from typing import Literal

import numpy as np


def validate_percentile(percentile: float) -> float:
    """Return a finite percentile in ``[0, 100]`` or raise ``ValueError``."""
    if isinstance(percentile, bool):
        raise ValueError("percentile must be a finite number between 0 and 100")
    try:
        result = float(percentile)
    except (TypeError, ValueError) as error:
        raise ValueError("percentile must be a finite number between 0 and 100") from error
    if not np.isfinite(result) or not 0.0 <= result <= 100.0:
        raise ValueError("percentile must be a finite number between 0 and 100")
    return result


def validate_threshold(threshold: float | None) -> float | None:
    """Return a finite optional threshold or raise ``ValueError``."""
    if threshold is None:
        return None
    if isinstance(threshold, bool):
        raise ValueError("threshold must be a finite number")
    try:
        result = float(threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("threshold must be a finite number") from error
    if not np.isfinite(result):
        raise ValueError("threshold must be a finite number")
    return result


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

    if np.isnan(scores).all():
        return 0.0
    return float(np.nanpercentile(scores, validated_percentile))


def threshold_flags(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float,
    direction: Literal["high", "low"],
    inclusive: bool | None = None,
) -> np.ndarray:
    """Create flags, including fixed-cutoff equality but excluding percentile ties."""
    if inclusive is None:
        inclusive = threshold is not None
    cutoff = resolve_threshold(scores, threshold, percentile)

    if direction == "high":
        return scores >= cutoff if inclusive else scores > cutoff
    return scores <= cutoff if inclusive else scores < cutoff
