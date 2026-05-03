"""
Screening function that runs multiple IER detection indices at once.

Provides a single entry point for computing all available IER indices,
flagging suspected careless responders, and summarizing results.
"""

import numpy as np

from ier._registry import (
    INDEX_REGISTRY,
    build_index_options,
    default_screen_indices,
    score_registered_indices,
    validate_index_names,
)
from ier._validation import MatrixLike, validate_matrix_input
from ier.types import ScreenIndexSummary, ScreenResult


def screen(
    x: MatrixLike,
    indices: list[str] | None = None,
    na_rm: bool = True,
    percentile: float = 95.0,
    psychsyn_critval: float = 0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> ScreenResult:
    """
    Screen respondents across multiple IER detection indices.

    Computes each requested index, flags outliers using percentile-based
    thresholds, and returns structured results.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to compute. If None, uses defaults (indices not
              requiring extra config). Options: "irv", "longstring", "longstring_pattern",
              "mahad", "psychsyn", "person_total", "markov", "u3_poly", "midpoint",
              "acquiescence", "evenodd", "mad", "lz".
    - na_rm: Handle missing values in individual indices.
    - percentile: Percentile cutoff for flagging (default 95th).
    - psychsyn_critval: Critical correlation value for psychometric synonyms.
    - evenodd_factors: Factor lengths for even-odd consistency.
    - mad_positive_items: Column indices for positively-worded items (for MAD).
    - mad_negative_items: Column indices for negatively-worded items (for MAD).
    - mad_scale_max: Maximum value of response scale (for MAD).
    - scale_min: Minimum value of response scale (for u3_poly, midpoint, acquiescence).
    - scale_max: Maximum value of response scale (for u3_poly, midpoint, acquiescence).

    Returns:
    - Dictionary with:
        - "scores": dict mapping index name to score array
        - "flags": dict mapping index name to boolean flag array
        - "flag_counts": array of total flags per respondent
        - "n_indices": number of indices successfully computed
        - "indices_used": list of index names computed
        - "n_respondents": number of respondents
        - "summary": dict mapping index name to summary statistics

    Raises:
    - ValueError: If invalid index names are specified.

    Example:
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> result = screen(data)
        >>> print(result["indices_used"])
        >>> print(result["flag_counts"])
    """
    x_array = validate_matrix_input(x, check_type=False)
    n_respondents = x_array.shape[0]

    if indices is None:
        indices = default_screen_indices()
    else:
        validate_index_names(indices)

    options = build_index_options(
        na_rm,
        psychsyn_critval,
        evenodd_factors,
        mad_positive_items,
        mad_negative_items,
        mad_scale_max,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    scores, errors = score_registered_indices(x_array, indices, options)

    flags: dict[str, np.ndarray] = {}
    for name, score_arr in scores.items():
        valid_scores = score_arr[~np.isnan(score_arr)]
        if len(valid_scores) == 0:
            flags[name] = np.zeros(n_respondents, dtype=bool)
            continue

        cutoff = float(np.percentile(valid_scores, percentile))

        flag_arr = np.zeros(n_respondents, dtype=bool)
        valid_mask = ~np.isnan(score_arr)

        if INDEX_REGISTRY[name].flag_direction == "high":
            flag_arr[valid_mask] = score_arr[valid_mask] > cutoff
        else:
            low_cutoff = float(np.percentile(valid_scores, 100.0 - percentile))
            flag_arr[valid_mask] = score_arr[valid_mask] < low_cutoff

        flags[name] = flag_arr

    flag_matrix = (
        np.column_stack(list(flags.values())) if flags else np.zeros((n_respondents, 0), dtype=bool)
    )
    flag_counts: np.ndarray = np.sum(flag_matrix, axis=1)

    summary: dict[str, ScreenIndexSummary] = {}
    for name, score_arr in scores.items():
        valid = score_arr[~np.isnan(score_arr)]
        if len(valid) > 0:
            summary[name] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "n_flagged": int(np.sum(flags[name])),
            }
        else:
            summary[name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n_flagged": 0,
            }

    return {
        "scores": scores,
        "flags": flags,
        "flag_counts": flag_counts,
        "n_indices": len(scores),
        "indices_used": list(scores.keys()),
        "errors": errors,
        "n_respondents": n_respondents,
        "summary": summary,
    }
