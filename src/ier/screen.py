"""
Screening function that runs multiple IER detection indices at once.

Provides a single entry point for computing all available IER indices,
flagging suspected careless responders, and summarizing results.
"""

from collections.abc import Mapping

import numpy as np

from ier._flagging import resolve_threshold, threshold_flags, validate_percentile
from ier._registry import (
    INDEX_REGISTRY,
    IndexOptions,
    default_screen_indices,
    resolve_index_options,
    score_registered_indices,
    validate_index_names,
    validate_min_valid_indices,
    validate_worker_count,
)
from ier._validation import MatrixLike, validate_matrix_input
from ier.types import ScreenIndexSummary, ScreenResult


def _resolve_screen_thresholds(
    thresholds: Mapping[str, float] | None,
    indices: list[str],
) -> dict[str, float]:
    if thresholds is None:
        return {}

    resolved: dict[str, float] = {}
    for name, value in thresholds.items():
        if name not in INDEX_REGISTRY:
            raise ValueError(f"unknown threshold index: {name}")
        if name not in indices:
            raise ValueError(f"threshold index is not selected: {name}")
        if INDEX_REGISTRY[name].flag_mode != "percentile":
            raise ValueError(f"{name} uses presence flagging and does not accept a threshold")
        if isinstance(value, bool):
            raise ValueError(f"threshold for {name} must be a finite number")
        try:
            cutoff = float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"threshold for {name} must be a finite number") from err
        if not np.isfinite(cutoff):
            raise ValueError(f"threshold for {name} must be a finite number")
        resolved[name] = cutoff
    return resolved


def _count_flags(flags: Mapping[str, np.ndarray], n_respondents: int) -> np.ndarray:
    """Count per-respondent flags without constructing an index matrix."""
    counts = np.zeros(n_respondents, dtype=np.int_)
    for values in flags.values():
        counts += values
    return counts


def _count_valid_scores(scores: Mapping[str, np.ndarray], n_respondents: int) -> np.ndarray:
    """Count available per-respondent scores without constructing an index matrix."""
    counts = np.zeros(n_respondents, dtype=np.int_)
    for values in scores.values():
        counts += ~np.isnan(values)
    return counts


def screen(
    x: MatrixLike,
    indices: list[str] | None = None,
    *,
    options: IndexOptions | None = None,
    percentile: float = 95.0,
    min_flags: int = 2,
    min_valid_indices: int | None = None,
    thresholds: Mapping[str, float] | None = None,
    strict: bool = False,
    workers: int = 1,
) -> ScreenResult:
    """
    Screen respondents across multiple IER detection indices.

    Computes each requested index, flags outliers using fixed or percentile-based
    thresholds (or presence detection for onset), and returns structured results.

    Configure indices with a single ``IndexOptions`` via ``options=``.

    Default indices are NumPy-only and do not require SciPy. Response-time indices
    take timing matrices (not item responses) and are intentionally outside the
    registry — call ``response_time*`` helpers directly.

    Parameters:
    - x: A matrix of data where rows are individuals and columns are item responses.
    - indices: List of indices to compute. If None, uses defaults that do not
              require extra config. Registered options include: "irv", "longstring",
              "longstring_pattern", "mahad", "psychsyn", "psychant", "person_total",
              "markov", "u3_poly", "midpoint", "acquiescence", "guttman",
              "individual_reliability", "onset", "evenodd", "mad", "lz",
              "semantic_syn", "semantic_ant", "infrequency", "missing_rate".
    - options: Shared index configuration (``IndexOptions``).
    - percentile: Percentile cutoff for flagging (default 95th).
    - min_flags: Minimum number of per-index flags required for a respondent-level
                 consensus flag (default 2).
    - min_valid_indices: Optional minimum number of available index scores required
                         before a respondent can receive a consensus flag.
    - thresholds: Optional fixed per-index cutoffs. Scores at or beyond a fixed
                  cutoff are flagged; indices without an override use percentiles.
    - strict: If True, raise when any selected index fails instead of recording
              the failure in ``errors`` (default False).
    - workers: Number of indices to score concurrently. The default of 1 preserves
               sequential execution; values above 1 trade additional peak memory
               for throughput on larger matrices.

    Returns:
    - Dictionary with:
        - "scores": dict mapping index name to score array
        - "flags": dict mapping index name to boolean flag array
        - "thresholds": actual per-index cutoffs (None for presence flagging)
        - "flag_counts": array of total flags per respondent
        - "valid_index_counts": array of available index scores per respondent
        - "consensus_eligible": respondents meeting ``min_valid_indices``
        - "consensus_flags": respondent-level flags meeting ``min_flags``
        - "min_flags": configured consensus threshold
        - "min_valid_indices": configured completeness threshold or None
        - "n_indices": number of indices successfully computed
        - "indices_used": list of index names computed
        - "errors": dict mapping failed index names to error messages
        - "n_respondents": number of respondents
        - "summary": dict mapping index name to summary statistics

    Raises:
    - ValueError: If index names, fixed thresholds, or consensus settings are invalid.

    Example:
        >>> from ier import IndexOptions, screen
        >>> data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        >>> result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
        >>> print(result["indices_used"])
        >>> print(result["flag_counts"])
        >>> print(result["consensus_flags"])
    """
    workers = validate_worker_count(workers)
    if isinstance(min_flags, bool) or not isinstance(min_flags, int) or min_flags < 1:
        raise ValueError("min_flags must be a positive integer")
    percentile = validate_percentile(percentile)

    x_array = validate_matrix_input(x, check_type=False)
    n_respondents = x_array.shape[0]

    if indices is None:
        indices = default_screen_indices()
    else:
        validate_index_names(indices)
    min_valid_indices = validate_min_valid_indices(min_valid_indices, len(indices))

    fixed_thresholds = _resolve_screen_thresholds(thresholds, indices)
    resolved = resolve_index_options(options)
    scores, errors = score_registered_indices(
        x_array,
        indices,
        resolved,
        strict=strict,
        workers=workers,
    )

    flags: dict[str, np.ndarray] = {}
    applied_thresholds: dict[str, float | None] = {}
    for name, score_arr in scores.items():
        spec = INDEX_REGISTRY[name]
        if spec.flag_mode == "present":
            flags[name] = ~np.isnan(score_arr)
            applied_thresholds[name] = None
            continue

        flag_percentile = percentile if spec.flag_direction == "high" else 100.0 - percentile
        explicit = name in fixed_thresholds
        cutoff = resolve_threshold(
            score_arr,
            fixed_thresholds[name] if explicit else None,
            flag_percentile,
        )
        flags[name] = threshold_flags(
            score_arr,
            threshold=cutoff,
            percentile=flag_percentile,
            direction=spec.flag_direction,
            inclusive=explicit,
        )
        applied_thresholds[name] = cutoff

    flag_counts = _count_flags(flags, n_respondents)
    valid_index_counts = _count_valid_scores(scores, n_respondents)
    consensus_eligible = (
        np.ones(n_respondents, dtype=bool)
        if min_valid_indices is None
        else valid_index_counts >= min_valid_indices
    )
    consensus_flags = (flag_counts >= min_flags) & consensus_eligible

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
        "thresholds": applied_thresholds,
        "flag_counts": flag_counts,
        "valid_index_counts": valid_index_counts,
        "consensus_eligible": consensus_eligible,
        "consensus_flags": consensus_flags,
        "min_flags": min_flags,
        "min_valid_indices": min_valid_indices,
        "n_indices": len(scores),
        "indices_used": list(scores.keys()),
        "errors": errors,
        "n_respondents": n_respondents,
        "summary": summary,
    }
