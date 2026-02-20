"""
Identifies the longest string or average length of identical consecutive responses
for each observation.

This module provides functions to analyze patterns in response data, particularly useful for
detecting careless responding patterns such as straightlining (repeating the same response).
"""

from itertools import groupby
from typing import Literal, overload

import numpy as np

from ier._validation import MatrixLike, iter_rows, validate_matrix_input


def run_length_encode(message: str) -> list[tuple[str, int]]:
    """
    Run-length encoding. Converts a string into a list of tuples, where each tuple contains
    the character and the length of consecutive occurrences.

    Parameters:
    - message: Input string to encode

    Returns:
    - List of tuples (character, count) representing consecutive character runs

    Example:
        >>> run_length_encode("aaabbbcc")
        [('a', 3), ('b', 3), ('c', 2)]
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    return [(char, len(list(group))) for char, group in groupby(message)]


def run_length_decode(encoded_data: list[tuple[str, int]]) -> str:
    """
    Decode run-length encoded data back to original string.

    Parameters:
    - encoded_data: List of tuples (character, count) from run_length_encode

    Returns:
    - Original string

    Example:
        >>> encoded = [('a', 3), ('b', 3), ('c', 2)]
        >>> run_length_decode(encoded)
        'aaabbbcc'
    """
    if not isinstance(encoded_data, list):
        raise TypeError("encoded_data must be a list")

    return "".join([char * count for char, count in encoded_data])


def longstr_message(message: str) -> tuple[str, int] | None:
    """
    Return the longest sequence of identical characters in a string.

    Parameters:
    - message: Input string to analyze

    Returns:
    - Tuple of (character, length) for the longest run, or None if string is empty

    Example:
        >>> longstr_message("aaabbbcc")
        ('a', 3)
        >>> longstr_message("")
        None
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    if not message:
        return None

    encoded = run_length_encode(message)
    longest_run = max(encoded, key=lambda x: x[1])
    return longest_run


def avgstr_message(message: str) -> float:
    """
    Return average length of uninterrupted strings of identical characters in a string.

    Parameters:
    - message: Input string to analyze

    Returns:
    - Average length of consecutive character runs

    Example:
        >>> avgstr_message("aaabbbcc")
        2.67
        >>> avgstr_message("")
        0.0
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")

    if not message:
        return 0.0

    rle_list = run_length_encode(message)
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
    - TypeError: If input is not a string, list of strings, or numpy array
    - ValueError: If input is empty or contains invalid data

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
            return avgstr_message(messages)
        else:
            return longstr_message(messages)

    if isinstance(messages, list):
        if not messages:
            raise ValueError("messages list cannot be empty")

        if not all(isinstance(msg, str) for msg in messages):
            raise TypeError("all elements in messages list must be strings")

        if avg:
            return [avgstr_message(msg) for msg in messages]
        else:
            return [longstr_message(msg) for msg in messages]

    elif isinstance(messages, np.ndarray):
        if messages.size == 0:
            raise ValueError("messages array cannot be empty")

        is_string_dtype = messages.dtype.kind in ("U", "S")
        messages_list: list[str] = messages.tolist()

        if not is_string_dtype:
            messages_list = [str(msg) for msg in messages_list]

        if avg:
            return [avgstr_message(msg) for msg in messages_list]
        else:
            return [longstr_message(msg) for msg in messages_list]

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

    if not na_rm and np.isnan(x_array).any():
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    n_rows = x_array.shape[0]
    result = np.zeros(n_rows, dtype=float)

    for i, row in enumerate(iter_rows(x_array, na_rm)):
        if len(row) < 4:
            continue

        result[i] = _longest_repeating_pattern(row, max_pattern_length)

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

    if not na_rm and np.isnan(x_array).any():
        raise ValueError("data contains missing values. Set na_rm=True to handle them")

    scores = np.zeros(x_array.shape[0], dtype=float)

    for i, row in enumerate(x_array):
        row_data = row[~np.isnan(row)] if na_rm else row
        if row_data.size == 0:
            continue

        change_points = np.flatnonzero(np.diff(row_data) != 0) + 1
        run_lengths = np.diff(np.concatenate(([0], change_points, [row_data.size])))
        scores[i] = float(np.max(run_lengths))

    return scores


def _longest_repeating_pattern(row: np.ndarray, max_k: int) -> float:
    """Find the longest consecutive repeating sub-pattern in a numeric sequence.

    Only counts patterns where the sub-pattern contains at least 2 distinct values
    (i.e., excludes straight-line / constant sequences which are detected by longstring).
    """
    n = len(row)
    best = 0

    for k in range(2, min(max_k, n // 2) + 1):
        start = 0
        while start + k <= n:
            pattern = row[start : start + k]
            if len(np.unique(pattern)) < 2:
                start += 1
                continue
            repeat_len = k
            for j in range(start + k, n):
                if row[j] == pattern[(j - start) % k]:
                    repeat_len += 1
                else:
                    break
            if repeat_len > k and repeat_len > best:
                best = repeat_len
            start += max(1, repeat_len - k + 1)

    return float(best)
