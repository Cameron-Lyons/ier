"""Shared helpers for percentile/threshold-based flagging."""

from typing import Literal

import numpy as np


def threshold_flags(
    scores: np.ndarray,
    threshold: float | None,
    percentile: float,
    direction: Literal["high", "low"],
    inclusive: bool = False,
) -> np.ndarray:
    """Create boolean flags from scores using explicit or percentile thresholding."""
    valid_scores = scores[~np.isnan(scores)]

    cutoff = threshold
    if cutoff is None:
        cutoff = 0.0 if len(valid_scores) == 0 else float(np.percentile(valid_scores, percentile))

    flags = np.zeros(len(scores), dtype=bool)
    valid_mask = ~np.isnan(scores)
    valid_values = scores[valid_mask]

    if direction == "high":
        flags[valid_mask] = valid_values >= cutoff if inclusive else valid_values > cutoff
    else:
        flags[valid_mask] = valid_values <= cutoff if inclusive else valid_values < cutoff

    return flags
