"""
Infrequency / bogus item scoring for detecting insufficient effort responding.

Counts the number of failed attention-check (bogus/infrequency) items per respondent.
These are items with known correct answers that attentive respondents should get right
(e.g., "Please select 'Strongly Agree' for this item").

References:
- Huang, J. L., Curran, P. G., Keeney, J., Poposki, E. M., & DeShon, R. P. (2012).
  Detecting and deterring insufficient effort responding to surveys.
  Journal of Business and Psychology, 27(1), 99-114.
- Meade, A. W., & Craig, S. B. (2012). Identifying careless responses in survey data.
  Psychological Methods, 17(3), 437-455.
"""

import numpy as np

from ier._flagging import validate_threshold
from ier._validation import MatrixLike, validate_matrix_input
from ier.types import InfrequencyMissingPolicy

_MISSING_POLICIES = {"pass", "fail", "omit", "propagate"}


def infrequency(
    x: MatrixLike,
    item_indices: list[int],
    expected_responses: list[float],
    proportion: bool = False,
    missing: InfrequencyMissingPolicy = "pass",
) -> np.ndarray:
    """
    Count failed attention-check items per respondent.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - item_indices: Column indices (0-based) of the attention-check items.
    - expected_responses: Expected correct response for each attention-check item.
    - proportion: If True, return proportion of failed items instead of count.
    - missing: Missing-response policy. ``"pass"`` preserves the legacy behavior
               of treating missing checks as correct; ``"fail"`` treats them as
               failures; ``"omit"`` excludes them from proportional denominators;
               and ``"propagate"`` returns ``NaN`` when any check is missing.

    Returns:
    - A numpy array of failure counts (or proportions) per respondent. Under
      ``missing="omit"``, rows without observed checks return ``NaN``.

    Raises:
    - ValueError: If the policy, item selection, expected responses, or proportion
                  control is invalid.

    Example:
        >>> data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        >>> scores = infrequency(data, item_indices=[0, 2], expected_responses=[5, 1])
        >>> print(scores)
        [0. 1. 2.]
    """
    x_array = validate_matrix_input(x, check_type=False)
    n_cols = x_array.shape[1]

    if not isinstance(proportion, bool):
        raise ValueError("proportion must be a boolean")
    if not isinstance(missing, str) or missing not in _MISSING_POLICIES:
        raise ValueError(f"missing must be one of: {sorted(_MISSING_POLICIES)}")
    if len(item_indices) == 0:
        raise ValueError("item_indices cannot be empty")

    if len(item_indices) != len(expected_responses):
        raise ValueError(
            f"item_indices ({len(item_indices)}) and expected_responses "
            f"({len(expected_responses)}) must have the same length"
        )

    for idx in item_indices:
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise ValueError("item_indices must contain integer column indices")
        if idx < 0 or idx >= n_cols:
            raise ValueError(f"item index {idx} out of bounds for data with {n_cols} columns")
    if len(set(item_indices)) != len(item_indices):
        raise ValueError("item_indices cannot contain duplicates")

    try:
        expected = np.asarray(expected_responses, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("expected_responses must contain finite numeric values") from error
    if expected.ndim != 1 or not np.isfinite(expected).all():
        raise ValueError("expected_responses must contain finite numeric values")

    failures = np.zeros(x_array.shape[0], dtype=float)
    available_counts = np.zeros(len(x_array), dtype=np.intp) if missing == "omit" else None
    unavailable = np.zeros(len(x_array), dtype=bool) if missing == "propagate" else None

    for idx, expected_value in zip(item_indices, expected, strict=True):
        col = x_array[:, idx]
        mismatch = col != expected_value
        if missing == "fail":
            failures += mismatch
            continue

        nan_mask = np.isnan(col)
        mismatch[nan_mask] = False
        failures += mismatch
        if available_counts is not None:
            available_counts += ~nan_mask
        if unavailable is not None:
            unavailable |= nan_mask

    if proportion:
        if available_counts is None:
            failures /= len(item_indices)
        else:
            failures = np.divide(
                failures,
                available_counts,
                out=np.full(len(failures), np.nan),
                where=available_counts > 0,
            )
    elif available_counts is not None:
        failures[available_counts == 0] = np.nan

    if unavailable is not None:
        failures[unavailable] = np.nan

    return failures


def infrequency_flag(
    x: MatrixLike,
    item_indices: list[int],
    expected_responses: list[float],
    threshold: float = 1.0,
    proportion: bool = False,
    missing: InfrequencyMissingPolicy = "pass",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count failed attention-check items and flag respondents exceeding a threshold.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - item_indices: Column indices (0-based) of the attention-check items.
    - expected_responses: Expected correct response for each attention-check item.
    - threshold: Failure count or proportion at or above which to flag (default 1).
    - proportion: If True, flag failure proportions instead of counts.
    - missing: Missing-response policy passed to ``infrequency()``.

    Returns:
    - Tuple of (failure_scores, flags) where flags is True for flagged respondents.

    Example:
        >>> data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        >>> scores, flags = infrequency_flag(data, [0, 2], [5, 1], threshold=2)
        >>> print(flags)
        [False False  True]
    """
    if not isinstance(proportion, bool):
        raise ValueError("proportion must be a boolean")
    resolved_threshold = validate_threshold(threshold)
    assert resolved_threshold is not None
    if resolved_threshold < 0:
        raise ValueError("threshold must be nonnegative")
    if proportion and resolved_threshold > 1:
        raise ValueError("proportion threshold must be between 0 and 1")

    scores = infrequency(
        x,
        item_indices,
        expected_responses,
        proportion=proportion,
        missing=missing,
    )
    flags = np.zeros(len(scores), dtype=bool)
    available = ~np.isnan(scores)
    flags[available] = scores[available] >= resolved_threshold
    return scores, flags
