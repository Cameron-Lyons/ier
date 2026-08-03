"""
Identifies the longest string or average length of identical consecutive responses
for each observation.

This module provides functions to analyze patterns in response data, particularly useful for
detecting careless responding patterns such as straightlining (repeating the same response).
"""

from itertools import groupby
from typing import Literal, overload

import numpy as np

from ier._row_statistics import compressed_row_groups, row_slices
from ier._validation import MatrixLike, validate_matrix_input

_MISSING_COMPRESSION_BATCH_ELEMENTS = 131_072


def _has_missing(x: np.ndarray) -> bool:
    """Check for missing responses without allocating a full-size mask."""
    return any(np.isnan(x[start:stop]).any() for start, stop in row_slices(len(x), x.shape[1]))


def _run_length_encode(message: str) -> list[tuple[str, int]]:
    """Run-length encode a string into ``(character, count)`` runs."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    return [(char, len(list(group))) for char, group in groupby(message)]


def _run_length_decode(encoded_data: list[tuple[str, int]]) -> str:
    """Decode run-length encoded ``(character, count)`` runs back to a string."""
    if not isinstance(encoded_data, list):
        raise TypeError("encoded_data must be a list")

    return "".join([char * count for char, count in encoded_data])


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
    if not rle_list:
        return 0.0

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

        >>> longstring("aaabbbcc", avg=True)
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
    - na_rm: If True, removes NaN values before analysis. If False, raises
             error if NaN values are present.

    Returns:
    - A numpy array with the longest repeating pattern length per respondent.
      Returns 0 if no repeating pattern is found.

    Raises:
    - ValueError: If inputs are invalid.

    Example:
        >>> data = [[1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6]]
        >>> longstring_pattern(data)
        array([6., 0.])
    """
    x_array = validate_matrix_input(x, min_columns=2, check_type=False)

    has_missing = _has_missing(x_array)
    if not na_rm and has_missing:
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    n_rows = x_array.shape[0]
    if not has_missing:
        return _longest_repeating_patterns(x_array, max_pattern_length)

    result = np.zeros(n_rows, dtype=float)
    for rows, compressed in compressed_row_groups(
        x_array,
        min_columns=4,
        max_elements=_MISSING_COMPRESSION_BATCH_ELEMENTS,
    ):
        result[rows] = _longest_repeating_patterns(compressed, max_pattern_length)

    return result


def longstring_scores(
    x: MatrixLike,
    na_rm: bool = True,
) -> np.ndarray:
    """
    Compute longest run-length scores directly from matrix rows.

    This avoids value-collisions from string casting (e.g., 1 vs 1.0 vs 1.00)
    and preserves non-integer response values.
    """
    x_array = validate_matrix_input(x, min_columns=1, check_type=False)

    has_missing = _has_missing(x_array)
    if not na_rm and has_missing:
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    if not has_missing:
        return _longstring_scores_complete(x_array)

    scores = np.zeros(x_array.shape[0], dtype=float)
    for rows, compressed in compressed_row_groups(
        x_array,
        min_columns=1,
        max_elements=_MISSING_COMPRESSION_BATCH_ELEMENTS,
    ):
        scores[rows] = _longstring_scores_complete(compressed)

    return scores


def _longstring_scores_complete(x: np.ndarray) -> np.ndarray:
    """Compute longest runs across complete matrix rows."""
    current = np.ones(x.shape[0], dtype=float)
    longest = current.copy()

    for column in range(1, x.shape[1]):
        current = np.where(x[:, column] == x[:, column - 1], current + 1.0, 1.0)
        longest = np.maximum(longest, current)

    return longest


def _longest_repeating_patterns(x: np.ndarray, max_k: int) -> np.ndarray:
    """Find longest consecutive repeating sub-patterns for complete matrix rows."""
    n_rows, n_columns = x.shape
    count_dtype = np.min_scalar_type(n_columns)
    best = np.zeros(n_rows, dtype=count_dtype)
    changes = x[:, 1:] != x[:, :-1]
    change_prefix = np.concatenate(
        (
            np.zeros((n_rows, 1), dtype=count_dtype),
            np.cumsum(changes, axis=1, dtype=count_dtype),
        ),
        axis=1,
    )

    for k in range(2, min(max_k, n_columns // 2) + 1):
        match_lengths = np.zeros(n_rows, dtype=count_dtype)
        for position in range(n_columns - k - 1, -1, -1):
            match_lengths = np.where(
                x[:, position + k] == x[:, position],
                match_lengths + 1,
                0,
            )
            nonconstant = change_prefix[:, position + k - 1] != change_prefix[:, position]
            candidates = np.where(
                nonconstant & (match_lengths > 0),
                k + match_lengths,
                0,
            )
            best = np.maximum(best, candidates)

    return best.astype(float)


def _longest_repeating_pattern(row: np.ndarray, max_k: int) -> float:
    """Find the longest consecutive repeating sub-pattern in a numeric sequence.

    Only counts patterns where the sub-pattern contains at least 2 distinct values
    (i.e., excludes straight-line / constant sequences which are detected by longstring).
    """
    n = len(row)
    best = 0
    changes = row[1:] != row[:-1]
    change_prefix = np.concatenate(([0], np.cumsum(changes, dtype=np.intp)))

    for k in range(2, min(max_k, n // 2) + 1):
        matches = row[k:] == row[:-k]
        match_indices = np.arange(matches.size)
        next_mismatch = np.minimum.accumulate(
            np.where(~matches, match_indices, matches.size)[::-1]
        )[::-1]
        match_lengths = next_mismatch - match_indices

        start = 0
        while start + k <= n:
            if change_prefix[start + k - 1] == change_prefix[start]:
                start += 1
                continue

            repeat_extension = int(match_lengths[start]) if start < matches.size else 0
            repeat_len = k + repeat_extension
            if repeat_extension > 0 and repeat_len > best:
                best = repeat_len
            start += repeat_extension + 1

    return float(best)
