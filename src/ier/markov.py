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

from typing import Any

import numpy as np

from ier._flagging import threshold_flags
from ier._response_sequences import sequence_batches
from ier._summary import calculate_summary_stats
from ier._validation import MatrixLike, validate_matrix_input

_MAX_DENSE_STATES = 64
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
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> markov(data)
        array([0.  , 1.56])
    """
    x_array = validate_matrix_input(x, min_columns=3, check_type=False)

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
    """Score complete or padded rows with bounded dense batches or a sparse fallback."""
    state_batch = _encode_states(x)
    if state_batch is None:
        return _transition_entropies_sparse(x, counts=counts)

    encoded, n_states = state_batch
    n_items = x.shape[1]
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


def _encode_states(x: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Encode one bounded sequence block, or use sparse scoring above 64 states."""
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

    categories = np.unique(x)
    if len(categories) > _MAX_DENSE_STATES:
        return None
    return np.searchsorted(categories, x), len(categories)


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


def _transition_entropies_sparse(x: np.ndarray, *, counts: np.ndarray | None = None) -> np.ndarray:
    """Score high-cardinality complete rows without dense state-square arrays."""
    result = np.empty(len(x), dtype=float)
    for row_index, raw_row in enumerate(x):
        row = raw_row if counts is None else raw_row[: counts[row_index]]
        if len(row) < 2:
            result[row_index] = np.nan
            continue
        result[row_index] = _transition_entropy_row(row)
    return result


def _transition_entropy_row(row: np.ndarray) -> float:
    """Compute transition entropy from the observed counts in one row."""
    _, encoded = np.unique(row, return_inverse=True)
    n_states = int(np.max(encoded)) + 1
    from_counts = np.bincount(encoded[:-1])
    pair_ids = encoded[:-1] * n_states + encoded[1:]
    _, pair_counts = np.unique(pair_ids, return_counts=True)
    return _conditional_entropy_from_counts(from_counts, pair_counts, len(row) - 1)


def _conditional_entropy_from_counts(
    from_counts: np.ndarray,
    pair_counts: np.ndarray,
    total: int | float,
) -> float:
    """Compute conditional entropy using only positive transition counts."""
    if total == 0:
        return 0.0

    positive_from = from_counts[from_counts > 0]
    positive_pairs = pair_counts[pair_counts > 0]
    from_terms = positive_from @ np.log2(positive_from)
    pair_terms = positive_pairs @ np.log2(positive_pairs)
    return float((from_terms - pair_terms) / total)


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
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> scores, flags = markov_flag(data)
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
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1]]
        >>> markov_summary(data)
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


def _transition_entropy(trans: np.ndarray) -> float:
    """Compute Shannon entropy of one transition matrix, weighted by row marginals."""
    row_sums = trans.sum(axis=1)
    total = float(row_sums.sum())
    return _conditional_entropy_from_counts(row_sums, trans.ravel(), total)


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
