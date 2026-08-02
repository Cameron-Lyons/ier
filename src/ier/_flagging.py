"""Shared helpers for percentile/threshold-based flagging."""

from typing import Literal

import numpy as np


def resolve_threshold(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float,
) -> float:
    """Resolve an explicit threshold or derive one from valid scores."""
    if threshold is not None:
        return threshold

    valid_scores = scores[~np.isnan(scores)]
    return 0.0 if len(valid_scores) == 0 else float(np.percentile(valid_scores, percentile))


def threshold_flags(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float,
    direction: Literal["high", "low"],
    inclusive: bool = False,
) -> np.ndarray:
    """Create boolean flags from scores using explicit or percentile thresholding."""
    cutoff = resolve_threshold(scores, threshold, percentile)

    flags = np.zeros(len(scores), dtype=bool)
    valid_mask = ~np.isnan(scores)
    valid_values = scores[valid_mask]

    if direction == "high":
        flags[valid_mask] = valid_values >= cutoff if inclusive else valid_values > cutoff
    else:
        flags[valid_mask] = valid_values <= cutoff if inclusive else valid_values < cutoff

    return flags
