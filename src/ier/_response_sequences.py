"""Stable missing-response compaction for bounded sequence-scoring batches."""

import numpy as np


def compact_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-align observed responses, preserving order and padding with NaN.

    Callers bound the input batch with ``row_slices``. Counts distinguish real
    sequences from padding, including rows with no observed responses.
    """
    valid = ~np.isnan(x)
    counts = np.asarray(np.sum(valid, axis=1, dtype=np.intp))
    width = int(np.max(counts))
    packed = np.full((len(x), width), np.nan, dtype=x.dtype)
    destinations = np.arange(width) < counts[:, None]
    packed[destinations] = x[valid]
    return packed, counts
