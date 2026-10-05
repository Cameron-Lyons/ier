"""
Markov chain index for detecting patterned insufficient effort responding.

Builds a first-order transition matrix from each respondent's response sequence and
computes the Shannon entropy of transitions. Low entropy indicates highly predictable
(patterned) responding, which may reflect careless strategies such as alternating
or cycling through response options.

References:
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

import math
from typing import Any

import numpy as np

from ier._flagging import threshold_flags
from ier._response_sequences import sequence_batches, true_run_lengths
from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input

_MAX_DENSE_STATES = 64
# Dense tables cost a state square per row; sorting a row of k items costs about
# k log2 k steps. Up to 64 states, sort only once the square exceeds this many
# cells per sort step. On an AVX2 machine, sorting overtook dense tables at about
# 0.2-0.4 cells per step with NumPy 2's vectorized sorts and 0.55-1.3 with NumPy
# 1.26's scalar sorts. At these constants, sorting measured 10-45% faster at the
# boundary (about even only for 3-5 items with NumPy 2), with or without missing
# responses. Rows of at least 843 items (NumPy 2) or 327 items (NumPy 1.x) keep
# dense tables for every state count up to 64. Faster sorts, such as NumPy 1.26
# with AVX-512, only widen the margin.
_DENSE_CELLS_PER_SORT_STEP = 0.5 if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else 1.5
_TRANSITION_BATCH_WORKSPACE_BYTES = 64 * 1024 * 1024


def markov(
    x: MatrixLike,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Compute Markov chain transition entropy for each respondent.

    Builds a first-order transition matrix from each respondent's response sequence
    and computes the Shannon entropy of the transition probabilities, weighted by
    row marginals. Low entropy indicates predictable, patterned responding.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - A numpy array of transition entropy values per respondent.
      Lower values indicate more predictable (potentially careless) patterns.

    Raises:
    - ValueError: If data has fewer than 3 columns.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 3, 1, 2]]
        >>> np.round(markov(data), 2).tolist()
        [0.0, 0.55]
    """
    x_array = validate_matrix_input(x, min_columns=3)

    result = np.full(len(x_array), np.nan)
    for start, stop, block, counts in sequence_batches(x_array, na_rm=na_rm):
        if block.shape[1] < 2:
            continue
        if counts is not None:
            # Compaction owns this block. Reuse an observed category for padding;
            # counts exclude padded transitions without adding a synthetic state.
            first_value = block[np.flatnonzero(counts)[0], 0]
            np.copyto(block, first_value, where=np.isnan(block))
        result[start:stop] = _markov_complete(block, counts=counts)

    return result


def _markov_complete(x: np.ndarray, *, counts: np.ndarray | None = None) -> np.ndarray:
    """Score complete or padded rows with bounded dense batches or sorted transitions."""
    encoded, n_states = _encode_states(x)
    n_items = x.shape[1]
    sort_cells = _DENSE_CELLS_PER_SORT_STEP * n_items * math.log2(n_items)
    if n_states > _MAX_DENSE_STATES or n_states * n_states > sort_cells:
        return _transition_entropies_from_codes(encoded, n_states, counts=counts)

    integer_bytes = np.dtype(np.intp).itemsize
    float_bytes = np.dtype(float).itemsize
    bytes_per_row = integer_bytes * (2 * n_items + n_states * n_states + n_states) + float_bytes * (
        n_states * n_states + n_states
    )
    if counts is not None:
        bytes_per_row += n_items * (integer_bytes + np.dtype(bool).itemsize)
    batch_rows = max(1, _TRANSITION_BATCH_WORKSPACE_BYTES // bytes_per_row)

    result = np.empty(len(x), dtype=float)
    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        batch_counts = None if counts is None else counts[start:stop]
        transition_counts = _dense_transition_counts(
            encoded[start:stop], n_states, counts=batch_counts
        )
        result[start:stop] = _transition_entropy_batch(transition_counts)
    if counts is not None:
        result[counts < 2] = np.nan
    return result


def _encode_states(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Encode one bounded sequence block as consecutive category codes."""
    minimum = np.min(x)
    maximum = np.max(x)
    integral = x.dtype.kind in "iu"
    span = float("inf")
    if integral:
        # Keep integer labels exact, including neighboring values beyond 2**53.
        span = int(maximum) - int(minimum)
    elif x.dtype.kind == "f" and np.isfinite(minimum) and np.isfinite(maximum):
        span = float(maximum) - float(minimum)

    if span < _MAX_DENSE_STATES and (integral or np.all(x == np.floor(x))):
        encoded = np.empty(x.shape, dtype=np.intp)
        np.subtract(x, minimum, out=encoded, casting="unsafe")
        present = np.bincount(encoded.ravel(), minlength=int(span) + 1) > 0
        n_states = int(np.count_nonzero(present))
        if n_states != len(present):
            # Empty categories need no rows or columns in the transition table.
            mapping = np.cumsum(present, dtype=np.intp) - 1
            np.take(mapping, encoded, out=encoded)
        return encoded, n_states

    categories, inverse = np.unique(x, return_inverse=True)
    # NumPy 2 returns the inverse in the input shape; NumPy 1.26 flattens it.
    encoded = inverse.reshape(x.shape).astype(np.intp, copy=False)
    return encoded, len(categories)


def _dense_transition_counts(
    encoded: np.ndarray, n_states: int, *, counts: np.ndarray | None = None
) -> np.ndarray:
    """Count transition pairs for one encoded batch without repeated row IDs."""
    n_rows, n_items = encoded.shape
    pair_ids = np.empty((n_rows, n_items - 1), dtype=np.intp)
    np.multiply(encoded[:, :-1], n_states, out=pair_ids)
    np.add(pair_ids, encoded[:, 1:], out=pair_ids)
    row_offsets = np.arange(n_rows, dtype=np.intp) * (n_states * n_states)
    pair_ids += row_offsets[:, None]

    if counts is None:
        observed_pairs = pair_ids.ravel()
    else:
        observed_pairs = pair_ids[np.arange(n_items - 1) < counts[:, None] - 1]
    transitions = np.bincount(observed_pairs, minlength=n_rows * n_states * n_states)
    return transitions.reshape(n_rows, n_states, n_states)


def _transition_entropies_from_codes(
    codes: np.ndarray, n_categories: int, *, counts: np.ndarray | None = None
) -> np.ndarray:
    """Score rows from sorted source-state and transition multiplicities.

    With ``n_s`` transitions leaving state ``s``, ``n_st`` transitions from ``s``
    to ``t`` and ``T`` transitions in total, the conditional entropy equals
    ``(sum n_s log2 n_s - sum n_st log2 n_st) / T``. Only observed states and
    pairs contribute, so the workspace grows with the items, not the states.
    """
    n_rows, n_items = codes.shape
    from_ids = codes[:, :-1].astype(np.int64)
    # Codes are bounded by the block size, so every pair ID fits in 64 bits.
    pair_ids = np.multiply(codes[:, :-1], n_categories, dtype=np.int64)
    pair_ids += codes[:, 1:]
    if counts is None:
        totals = np.full(n_rows, n_items - 1, dtype=np.intp)
    else:
        # Distinct sentinels above every pair ID form singleton runs, and
        # 1 * log2(1) = 0, so padded transitions contribute nothing.
        positions = np.arange(n_items - 1, dtype=np.int64)
        padded = positions >= counts[:, None] - 1
        sentinels = n_categories * n_categories + positions
        np.copyto(from_ids, sentinels, where=padded)
        np.copyto(pair_ids, sentinels, where=padded)
        totals = np.maximum(counts - 1, 0)

    numerators = _sum_xlogx_multiplicities(from_ids)
    numerators -= _sum_xlogx_multiplicities(pair_ids)
    result = np.full(n_rows, np.nan)
    np.divide(numerators, totals, out=result, where=totals > 0)
    return result


def _sum_xlogx_multiplicities(ids: np.ndarray) -> np.ndarray:
    """Sum ``c * log2(c)`` over the multiplicity ``c`` of each distinct row ID."""
    ordered = np.sort(ids, axis=1)
    width = ordered.shape[1]
    run_ends = np.empty(ordered.shape, dtype=bool)
    np.not_equal(ordered[:, 1:], ordered[:, :-1], out=run_ends[:, :-1])
    run_ends[:, -1] = True
    # Each sorted position stores the length of the run of equal IDs ending there.
    lengths = np.empty(ordered.shape, dtype=np.min_scalar_type(width))
    lengths[:, 0] = 1
    np.add(true_run_lengths(~run_ends[:, :-1]), 1, out=lengths[:, 1:], dtype=lengths.dtype)
    # Interior run positions select terms[0] == 0. An unmasked sum keeps NumPy's
    # pairwise summation; a sparse where= mask would add one term at a time.
    lengths *= run_ends
    multiplicities = np.arange(2, width + 1, dtype=float)
    terms = np.zeros(width + 1)
    np.multiply(multiplicities, np.log2(multiplicities), out=terms[2:])
    sums: np.ndarray = np.sum(terms[lengths], axis=1)
    return sums


def markov_flag(
    x: MatrixLike,
    threshold: float | None = None,
    percentile: float = 5.0,
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Markov chain entropy and flag respondents with low entropy.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - threshold: Absolute entropy threshold at or below which to flag. If None, uses percentile.
    - percentile: Percentile below which to flag (default 5th percentile).
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - Tuple of (entropy_scores, flags) where flags is True for flagged respondents.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 3, 1, 2]]
        >>> scores, flags = markov_flag(data, threshold=0.1)
        >>> flags.tolist()
        [True, False]
    """
    scores = markov(x, na_rm=na_rm)

    flags = threshold_flags(scores, threshold=threshold, percentile=percentile, direction="low")

    return scores, flags


def markov_summary(
    x: MatrixLike,
    na_rm: bool = True,
) -> dict[str, Any]:
    """
    Calculate summary statistics for Markov chain entropy scores.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - na_rm: If True, removes NaN values before analysis.

    Returns:
    - Dictionary with summary statistics.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 2, 1, 3, 1, 2]]
        >>> summary = markov_summary(data)
        >>> summary["n_valid"], round(summary["max"], 2)
        (2, 0.55)
    """
    scores = markov(x, na_rm=na_rm)

    summary = calculate_summary_stats(scores)
    summary.update(
        {
            "n_total": len(scores),
            "n_valid": int(np.sum(~np.isnan(scores))),
            "n_missing": int(np.sum(np.isnan(scores))),
        }
    )
    return summary


def _transition_entropy_batch(transitions: np.ndarray) -> np.ndarray:
    """Vectorized Shannon entropy for a batch of transition matrices."""
    row_sums = transitions.sum(axis=2)
    totals = row_sums.sum(axis=1)

    transition_terms = np.zeros(transitions.shape, dtype=float)
    np.log2(transitions, where=transitions > 0, out=transition_terms)
    transition_terms *= transitions

    row_terms = np.zeros(row_sums.shape, dtype=float)
    np.log2(row_sums, where=row_sums > 0, out=row_terms)
    row_terms *= row_sums

    numerators = np.sum(row_terms, axis=1) - np.sum(transition_terms, axis=(1, 2))
    result: np.ndarray = np.divide(
        numerators,
        totals,
        out=np.zeros(len(transitions), dtype=float),
        where=totals > 0,
    )
    return result
