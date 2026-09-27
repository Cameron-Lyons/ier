"""Bounded response-sequence preparation and run-length kernels."""

from collections.abc import Iterator

import numpy as np

from ier._row_statistics import row_slices


def sequence_batches(
    x: np.ndarray, *, na_rm: bool
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray | None]]:
    """Yield complete or NaN-padded batches and their optional observed lengths."""
    for start, stop in row_slices(*x.shape):
        block = x[start:stop]
        counts = None
        if np.isnan(block).any():
            if not na_rm:
                raise ValueError("data contains missing values. Set na_rm=True to handle them")
            block, counts = compact_rows(block)
        yield start, stop, block, counts


def true_run_lengths(matches: np.ndarray) -> np.ndarray:
    """Count consecutive true values ending at each position in bounded rows."""
    width = matches.shape[1]
    positions = np.arange(1, width + 1, dtype=np.min_scalar_type(width))
    lengths = np.where(matches, 0, positions)
    np.maximum.accumulate(lengths, axis=1, out=lengths)
    np.subtract(positions, lengths, out=lengths)
    return lengths


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
