"""
Screening function that runs multiple IER detection indices at once.

Provides a single entry point for computing all available IER indices,
flagging suspected careless responders, and summarizing results.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

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
from ier._validation import MatrixLike, validate_matrix_input, validate_score_vectors
from ier.types import IndexThresholdSourceMap, ScreenIndexSummary, ScreenResult


def _validate_min_flags(min_flags: int) -> int:
    """Return a validated respondent-level consensus threshold."""
    if isinstance(min_flags, bool) or not isinstance(min_flags, int) or min_flags < 1:
        raise ValueError("min_flags must be a positive integer")
    return min_flags


def _validate_screen_scores(
    scores: Mapping[str, ArrayLike],
) -> tuple[dict[str, np.ndarray], int]:
    """Validate reusable scores and restrict them to registered indices."""
    validated, n_respondents = validate_score_vectors(scores)
    validate_index_names(list(validated))
    return validated, n_respondents


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


def _resolve_screen_percentiles(
    percentiles: Mapping[str, float] | None,
    indices: list[str],
    fixed_thresholds: Mapping[str, float],
) -> dict[str, float]:
    """Validate per-index tail-percentile overrides."""
    if percentiles is None:
        return {}

    resolved: dict[str, float] = {}
    for name, value in percentiles.items():
        if name not in INDEX_REGISTRY:
            raise ValueError(f"unknown percentile index: {name}")
        if name not in indices:
            raise ValueError(f"percentile index is not selected: {name}")
        if INDEX_REGISTRY[name].flag_mode != "percentile":
            raise ValueError(f"{name} uses presence flagging and does not accept a percentile")
        if name in fixed_thresholds:
            raise ValueError(f"cannot set both a threshold and percentile for index: {name}")
        try:
            resolved[name] = validate_percentile(value)
        except ValueError as error:
            raise ValueError(
                f"percentile for {name} must be a finite number between 0 and 100"
            ) from error
    return resolved


def _reduce_screen_results(
    scores: Mapping[str, np.ndarray],
    flags: Mapping[str, np.ndarray],
    n_respondents: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, ScreenIndexSummary]]:
    """Accumulate respondent counts and per-index summaries without stacking."""
    flag_counts = np.zeros(n_respondents, dtype=np.int_)
    valid_index_counts = np.zeros(n_respondents, dtype=np.int_)
    summary: dict[str, ScreenIndexSummary] = {}

    for name, score_arr in scores.items():
        flag_arr = flags[name]
        flag_counts += flag_arr
        valid_mask = ~np.isnan(score_arr)
        valid_index_counts += valid_mask
        valid = score_arr[valid_mask]
        n_valid = len(valid)
        n_flagged = int(np.count_nonzero(flag_arr))
        flag_rate = n_flagged / n_valid if n_valid > 0 else float("nan")
        if n_valid > 0:
            summary[name] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "n_valid": n_valid,
                "n_unavailable": n_respondents - n_valid,
                "n_flagged": n_flagged,
                "flag_rate": flag_rate,
            }
        else:
            summary[name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n_valid": 0,
                "n_unavailable": n_respondents,
                "n_flagged": n_flagged,
                "flag_rate": flag_rate,
            }

    return flag_counts, valid_index_counts, summary


def _build_screen_result(
    scores: dict[str, np.ndarray],
    errors: dict[str, str],
    n_respondents: int,
    *,
    percentile: float,
    min_flags: int,
    min_valid_indices: int | None,
    fixed_thresholds: Mapping[str, float],
    percentile_overrides: Mapping[str, float],
) -> ScreenResult:
    """Apply flagging and consensus rules to validated score vectors."""
    flags: dict[str, np.ndarray] = {}
    applied_thresholds: dict[str, float | None] = {}
    threshold_sources: IndexThresholdSourceMap = {}
    applied_percentiles: dict[str, float | None] = {}
    for name, score_arr in scores.items():
        spec = INDEX_REGISTRY[name]
        if spec.flag_mode == "present":
            flags[name] = ~np.isnan(score_arr)
            applied_thresholds[name] = None
            threshold_sources[name] = "presence"
            applied_percentiles[name] = None
            continue

        tail_percentile = percentile_overrides.get(name, percentile)
        flag_percentile = (
            tail_percentile if spec.flag_direction == "high" else 100.0 - tail_percentile
        )
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
        threshold_sources[name] = "fixed" if explicit else "percentile"
        applied_percentiles[name] = None if explicit else tail_percentile

    flag_counts, valid_index_counts, summary = _reduce_screen_results(
        scores,
        flags,
        n_respondents,
    )
    consensus_eligible = (
        np.ones(n_respondents, dtype=bool)
        if min_valid_indices is None
        else valid_index_counts >= min_valid_indices
    )
    consensus_flags = (flag_counts >= min_flags) & consensus_eligible

    return {
        "scores": scores,
        "flags": flags,
        "thresholds": applied_thresholds,
        "threshold_sources": threshold_sources,
        "percentiles": applied_percentiles,
        "flag_counts": flag_counts,
        "valid_index_counts": valid_index_counts,
        "consensus_eligible": consensus_eligible,
        "consensus_flags": consensus_flags,
        "min_flags": min_flags,
        "min_valid_indices": min_valid_indices,
        "n_indices": len(scores),
        "indices_used": list(scores),
        "errors": errors,
        "n_respondents": n_respondents,
        "summary": summary,
    }


def screen_scores(
    scores: Mapping[str, ArrayLike],
    *,
    percentile: float = 95.0,
    min_flags: int = 2,
    min_valid_indices: int | None = None,
    thresholds: Mapping[str, float] | None = None,
    percentiles: Mapping[str, float] | None = None,
) -> ScreenResult:
    """
    Apply screening decisions to already-computed registered-index scores.

    This is the reusable post-scoring counterpart to :func:`screen`. It supports
    fast threshold, percentile, and consensus sensitivity analysis without
    calculating any index again. Score mappings preserve insertion order; each
    value must be a non-empty one-dimensional numeric vector with the same
    respondent count. Finite values and ``NaN`` are accepted, with ``NaN``
    treated as an unavailable score.

    Compatible ``float64`` NumPy vectors are retained by reference. The function
    never mutates them, but callers should avoid changing the arrays while using
    the returned result.

    Parameters:
    - scores: Mapping from registered index names to respondent score vectors.
    - percentile: Default tail percentile for sample-relative flagging.
    - min_flags: Minimum number of index flags required for consensus.
    - min_valid_indices: Optional minimum available-score count for consensus.
    - thresholds: Optional fixed per-index cutoffs.
    - percentiles: Optional per-index tail-percentile overrides.

    Returns:
    - The same structured ``ScreenResult`` contract as :func:`screen`, with an
      empty ``errors`` mapping because no index calculation is attempted.

    Example:
        >>> from ier import screen, screen_scores
        >>> initial = screen(data, indices=["irv", "longstring"])
        >>> stricter = screen_scores(
        ...     initial["scores"],
        ...     percentiles={"irv": 99, "longstring": 99},
        ... )
    """
    validated_scores, n_respondents = _validate_screen_scores(scores)
    indices = list(validated_scores)
    percentile = validate_percentile(percentile)
    min_flags = _validate_min_flags(min_flags)
    min_valid_indices = validate_min_valid_indices(min_valid_indices, len(indices))
    fixed_thresholds = _resolve_screen_thresholds(thresholds, indices)
    percentile_overrides = _resolve_screen_percentiles(percentiles, indices, fixed_thresholds)

    return _build_screen_result(
        validated_scores,
        {},
        n_respondents,
        percentile=percentile,
        min_flags=min_flags,
        min_valid_indices=min_valid_indices,
        fixed_thresholds=fixed_thresholds,
        percentile_overrides=percentile_overrides,
    )


def screen(
    x: MatrixLike,
    indices: list[str] | None = None,
    *,
    options: IndexOptions | None = None,
    percentile: float = 95.0,
    min_flags: int = 2,
    min_valid_indices: int | None = None,
    thresholds: Mapping[str, float] | None = None,
    percentiles: Mapping[str, float] | None = None,
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
    - percentiles: Optional per-index tail-percentile overrides. High-direction
                   indices use the configured value and low-direction indices use
                   ``100 - value``. An index cannot have both override types.
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
        - "threshold_sources": fixed, percentile, or presence origin per cutoff
        - "percentiles": requested tail percentiles (None for fixed/presence rules)
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
        - "summary": per-index moments, coverage counts, and valid-score flag rates

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
    min_flags = _validate_min_flags(min_flags)
    percentile = validate_percentile(percentile)

    x_array = validate_matrix_input(x, check_type=False)
    n_respondents = x_array.shape[0]

    if indices is None:
        indices = default_screen_indices()
    else:
        validate_index_names(indices)
    min_valid_indices = validate_min_valid_indices(min_valid_indices, len(indices))

    fixed_thresholds = _resolve_screen_thresholds(thresholds, indices)
    percentile_overrides = _resolve_screen_percentiles(percentiles, indices, fixed_thresholds)
    resolved = resolve_index_options(options)
    scores, errors = score_registered_indices(
        x_array,
        indices,
        resolved,
        strict=strict,
        workers=workers,
    )

    return _build_screen_result(
        scores,
        errors,
        n_respondents,
        percentile=percentile,
        min_flags=min_flags,
        min_valid_indices=min_valid_indices,
        fixed_thresholds=fixed_thresholds,
        percentile_overrides=percentile_overrides,
    )
