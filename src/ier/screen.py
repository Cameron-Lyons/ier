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
    psychant_critval: float = -0.6,
    evenodd_factors: list[int] | None = None,
    mad_positive_items: list[int] | None = None,
    mad_negative_items: list[int] | None = None,
    mad_scale_max: int | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
    acquiescence_positive_items: list[int] | None = None,
    acquiescence_negative_items: list[int] | None = None,
    longstring_max_pattern_length: int = 5,
    midpoint_tolerance: float = 0.0,
    guttman_normalize: bool = True,
    onset_window_size: int = 10,
    onset_min_items: int = 20,
    reliability_n_splits: int = 100,
    reliability_random_seed: int | None = None,
    semantic_item_pairs: list[tuple[int, int]] | None = None,
    infrequency_item_indices: list[int] | None = None,
    infrequency_expected_responses: list[float] | None = None,
    infrequency_proportion: bool = False,
) -> ScreenResult:
    """
    Screen respondents across multiple IER detection indices.

    Computes each requested index, flags outliers using percentile-based
    thresholds (or presence detection for onset), and returns structured results.

    Default indices are NumPy-only and do not require SciPy. Response-time indices
    are not included here because they use a different input domain; call them
    directly.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to compute. If None, uses defaults that do not
              require extra config. Registered options include: "irv", "longstring",
              "longstring_pattern", "mahad", "psychsyn", "psychant", "person_total",
              "markov", "u3_poly", "midpoint", "acquiescence", "guttman",
              "individual_reliability", "onset", "evenodd", "mad", "lz",
              "semantic_syn", "semantic_ant", "infrequency".
    - na_rm: Handle missing values in individual indices.
    - percentile: Percentile cutoff for flagging (default 95th).
    - psychsyn_critval: Critical correlation value for psychometric synonyms.
    - psychant_critval: Critical correlation value for psychometric antonyms.
    - evenodd_factors: Factor lengths for even-odd consistency.
    - mad_positive_items: Column indices for positively-worded items (for MAD).
    - mad_negative_items: Column indices for negatively-worded items (for MAD).
    - mad_scale_max: Maximum value of response scale (for MAD).
    - scale_min: Minimum value of response scale (for u3_poly, midpoint, acquiescence).
    - scale_max: Maximum value of response scale (for u3_poly, midpoint, acquiescence).
    - acquiescence_positive_items: Positive items for balanced-pair acquiescence.
    - acquiescence_negative_items: Negative items for balanced-pair acquiescence.
    - longstring_max_pattern_length: Max repeating-pattern length for longstring_pattern.
    - midpoint_tolerance: Tolerance around scale midpoint for midpoint responding.
    - guttman_normalize: If True, return Guttman error proportions instead of counts.
    - onset_window_size: Sliding window size for onset detection.
    - onset_min_items: Minimum items required for onset detection.
    - reliability_n_splits: Split-half iterations for individual_reliability.
    - reliability_random_seed: Optional seed for individual_reliability.
    - semantic_item_pairs: Item pairs for semantic_syn / semantic_ant.
    - infrequency_item_indices: Attention-check item column indices.
    - infrequency_expected_responses: Expected responses for attention checks.
    - infrequency_proportion: If True, return failure proportions for infrequency.

    Returns:
    - Dictionary with:
        - "scores": dict mapping index name to score array
        - "flags": dict mapping index name to boolean flag array
        - "flag_counts": array of total flags per respondent
        - "n_indices": number of indices successfully computed
        - "indices_used": list of index names computed
        - "errors": dict mapping failed index names to error messages
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
        psychant_critval=psychant_critval,
        acquiescence_positive_items=acquiescence_positive_items,
        acquiescence_negative_items=acquiescence_negative_items,
        longstring_max_pattern_length=longstring_max_pattern_length,
        midpoint_tolerance=midpoint_tolerance,
        guttman_normalize=guttman_normalize,
        onset_window_size=onset_window_size,
        onset_min_items=onset_min_items,
        reliability_n_splits=reliability_n_splits,
        reliability_random_seed=reliability_random_seed,
        semantic_item_pairs=semantic_item_pairs,
        infrequency_item_indices=infrequency_item_indices,
        infrequency_expected_responses=infrequency_expected_responses,
        infrequency_proportion=infrequency_proportion,
    )
    scores, errors = score_registered_indices(x_array, indices, options)

    flags: dict[str, np.ndarray] = {}
    for name, score_arr in scores.items():
        spec = INDEX_REGISTRY[name]
        if spec.flag_mode == "present":
            flags[name] = ~np.isnan(score_arr)
            continue

        valid_scores = score_arr[~np.isnan(score_arr)]
        if len(valid_scores) == 0:
            flags[name] = np.zeros(n_respondents, dtype=bool)
            continue

        cutoff = float(np.percentile(valid_scores, percentile))

        flag_arr = np.zeros(n_respondents, dtype=bool)
        valid_mask = ~np.isnan(score_arr)

        if spec.flag_direction == "high":
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
