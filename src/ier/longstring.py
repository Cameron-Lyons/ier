"""
Identifies the longest string or average length of identical consecutive responses
for each observation.

This module provides functions to analyze patterns in response data, particularly useful for
detecting careless responding patterns such as straightlining (repeating the same response).
"""

from itertools import groupby
from typing import Literal, overload

import numpy as np

from ier._response_sequences import sequence_batches, true_run_lengths
from ier._validation import MatrixLike, validate_integer, validate_matrix_input


def _run_length_encode(message: str) -> list[tuple[str, int]]:
    """Run-length encode a string into ``(character, count)`` runs."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    return [(char, len(list(group))) for char, group in groupby(message)]


def _longstr_message(message: str) -> tuple[str, int] | None:
    """Return ``(character, length)`` for the longest identical run, or None."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    if not message:
        return None

    encoded = _run_length_encode(message)
    longest_run = max(encoded, key=lambda x: x[1])
    return longest_run


def _avgstr_message(message: str) -> float:
    """Return average length of uninterrupted identical-character runs."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    if not message:
        return 0.0

    rle_list = _run_length_encode(message)
    total_len = sum(count for _, count in rle_list)
    return total_len / len(rle_list)


@overload
def longstring(messages: str, avg: Literal[False] = False) -> tuple[str, int] | None: ...
@overload
def longstring(messages: str, avg: Literal[True]) -> float: ...
@overload
def longstring(
    messages: list[str], avg: Literal[False] = False
) -> list[tuple[str, int] | None]: ...
@overload
def longstring(messages: list[str], avg: Literal[True]) -> list[float]: ...
@overload
def longstring(
    messages: np.ndarray, avg: Literal[False] = False
) -> list[tuple[str, int] | None]: ...
@overload
def longstring(messages: np.ndarray, avg: Literal[True]) -> list[float]: ...


def longstring(
    messages: str | list[str] | np.ndarray, avg: bool = False
) -> tuple[str, int] | None | list[tuple[str, int] | None] | float | list[float]:
    """
    Analyze strings for patterns of identical consecutive characters.

    This function is useful for detecting careless responding patterns in survey data.
    It can identify either the longest sequence of identical responses or the average
    length of consecutive identical responses.

    Parameters:
    - messages: Input string(s) to analyze. Can be a single string, list of strings,
               or numpy array of strings.
    - avg: If True, return average length of consecutive identical characters.
           If False, return the longest sequence of identical characters.

    Returns:
    - If avg=False: Tuple (character, length) for longest run, or None if no runs found
    - If avg=True: Float representing average length of consecutive runs
    - For multiple messages: List of results for each message

    Raises:
    - TypeError: If input is not a string or contains non-string values
    - ValueError: If input is empty or a numpy array is not one-dimensional

    Example:
        >>> longstring("aaabbbcc")
        ('a', 3)

        >>> round(longstring("aaabbbcc", avg=True), 2)
        2.67

        >>> data = ["aaabbb", "cccc", "abc"]
        >>> longstring(data)
        [('a', 3), ('c', 4), ('a', 1)]

        >>> import numpy as np
        >>> arr = np.array(["aaabbb", "cccc", "abc"])
        >>> longstring(arr, avg=True)
        [3.0, 4.0, 1.0]
    """

    if messages is None:
        raise ValueError("messages cannot be None")

    if isinstance(messages, str):
        if avg:
            return _avgstr_message(messages)
        else:
            return _longstr_message(messages)

    if isinstance(messages, list):
        if not messages:
            raise ValueError("messages list cannot be empty")

        if not all(isinstance(msg, str) for msg in messages):
            raise TypeError("all elements in messages list must be strings")

        if avg:
            return [_avgstr_message(msg) for msg in messages]
        else:
            return [_longstr_message(msg) for msg in messages]

    elif isinstance(messages, np.ndarray):
        if messages.size == 0:
            raise ValueError("messages array cannot be empty")
        if messages.ndim != 1:
            raise ValueError("messages array must be one-dimensional")

        messages_list: list[str] = []
        for message in messages.tolist():
            if not isinstance(message, str):
                raise TypeError("all elements in messages array must be strings")
            messages_list.append(message)

        if avg:
            return [_avgstr_message(msg) for msg in messages_list]
        else:
            return [_longstr_message(msg) for msg in messages_list]

    else:
        raise TypeError("messages must be a string, list of strings, or numpy array")


def longstring_pattern(
    x: MatrixLike,
    max_pattern_length: int = 5,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Detect repeating sub-patterns in numeric response sequences.

    For each respondent, searches for repeating sub-patterns of length 2..k
    in their response vector. Returns the longest consecutive repeating
    pattern length found. Detects seesaw (1-2-1-2), cycling (1-2-3-1-2-3),
    and similar patterned responding.

    Parameters:
    - x: A matrix of numeric data where rows are individuals and columns are
         item responses.
    - max_pattern_length: Maximum sub-pattern length to search for (default 5).
                          Must be an integer of at least 2, the shortest
                          repeating pattern.
    - na_rm: If True, removes NaN values before analysis. If False, raises
             error if NaN values are present.

    Returns:
    - A numpy array with the longest repeating pattern length per respondent.
      Returns 0 if no repeating pattern is found.

    Raises:
    - ValueError: If inputs are invalid, including a Boolean, non-integer, or
                  smaller ``max_pattern_length``.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6]]
        >>> longstring_pattern(data).tolist()
        [6.0, 0.0]
    """
    max_pattern_length = validate_integer(
        max_pattern_length, message="max_pattern_length must be an integer of at least 2", minimum=2
    )
    x_array = validate_matrix_input(x, min_columns=2)

    result = np.empty(len(x_array))
    for start, stop, block, counts in sequence_batches(x_array, na_rm=na_rm):
        result[start:stop] = _longest_repeating_patterns(block, max_pattern_length, counts=counts)
    return result


def longstring_scores(
    x: MatrixLike,
    na_rm: bool = True,
    avg: bool = False,
) -> np.ndarray:
    """
    Compute longest or average run-length scores directly from matrix rows.

    This avoids value-collisions from string casting (e.g., 1 vs 1.0 vs 1.00)
    and preserves non-integer response values.

    Parameters:
    - x: A matrix of numeric data where rows are individuals and columns are
         item responses.
    - na_rm: If True, removes NaN values before analysis, so a run can continue
             across a skipped item. R's ``rle()`` instead ends a run at ``NA``.
             If False, raises an error if NaN values are present.
    - avg: If True, return the average length of uninterrupted identical runs
           (``avgstr``, the mean of ``rle(x)$lengths`` in R's ``careless``):
           observed responses divided by the number of runs. If False (default),
           return the longest run.

    Returns:
    - A numpy array of run lengths per respondent. Rows without observed
      responses score 0 for the longest run and NaN for the average.

    Raises:
    - ValueError: If inputs are invalid or ``avg`` is not a boolean.

    Example:
        >>> data = [[1, 1, 2, 2, 2, 3], [4, 4, 4, 4, 4, 4]]
        >>> longstring_scores(data).tolist()
        [3.0, 6.0]
        >>> longstring_scores(data, avg=True).tolist()
        [2.0, 6.0]
    """
    if not isinstance(avg, bool):
        raise ValueError("avg must be a boolean")
    x_array = validate_matrix_input(x, min_columns=1)

    scores = np.empty(len(x_array))
    for start, stop, block, counts in sequence_batches(x_array, na_rm=na_rm):
        if avg:
            changes = block[:, 1:] != block[:, :-1]
            if counts is not None:
                # NaN padding never equals itself; count only changes between
                # observed responses, at positions j < counts - 1.
                changes &= np.arange(1, block.shape[1]) < counts[:, None]
            n_runs = np.count_nonzero(changes, axis=1) + 1
            block_scores = (block.shape[1] if counts is None else counts) / n_runs
            if counts is not None:
                block_scores[counts == 0] = np.nan
        else:
            lengths = true_run_lengths(block[:, 1:] == block[:, :-1])
            block_scores = np.max(lengths, axis=1, initial=0).astype(float) + 1.0
            if counts is not None:
                block_scores[counts == 0] = 0.0
        scores[start:stop] = block_scores
    return scores


def _longest_repeating_patterns(
    x: np.ndarray, max_k: int, *, counts: np.ndarray | None = None
) -> np.ndarray:
    """Find repeating sub-patterns in one bounded complete or NaN-padded batch."""
    n_rows, n_columns = x.shape
    best = np.zeros(n_rows)
    if n_columns < 4:
        return best

    change_prefix = np.empty(x.shape, dtype=np.min_scalar_type(n_columns))
    change_prefix[:, 0] = 0
    np.cumsum(
        x[:, 1:] != x[:, :-1],
        axis=1,
        dtype=change_prefix.dtype,
        out=change_prefix[:, 1:],
    )

    for k in range(2, min(max_k, n_columns // 2) + 1):
        lengths = true_run_lengths(x[:, k:] == x[:, :-k])
        # Each lag-k match extends a repeating pattern by one response. Its
        # final k responses are a rotation of the first k, so checking that
        # final window excludes constant patterns regardless of run length.
        constant = change_prefix[:, k:] == change_prefix[:, 1 : n_columns - k + 1]
        np.copyto(lengths, 0, where=constant)
        longest = np.max(lengths, axis=1).astype(float)
        candidates = np.where(longest > 0, longest + k, 0.0)
        if counts is not None:
            candidates[counts < 2 * k] = 0.0
        np.maximum(best, candidates, out=best)

    return best
