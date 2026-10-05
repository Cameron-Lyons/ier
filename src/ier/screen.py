"""
Screening function that runs multiple IER detection indices at once.

Provides a single entry point for computing all available IER indices,
flagging suspected careless responders, and summarizing results.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ier._flagging import resolve_threshold, threshold_flags, validate_percentile
from ier._registry import (
    INDEX_REGISTRY,
    IndexOptions,
    default_screen_indices,
    numeric_override,
    resolve_index_options,
    resolve_index_overrides,
    score_registered_indices,
    validate_index_errors,
    validate_index_names,
    validate_min_valid_indices,
    validate_worker_count,
)
from ier._summary import observed_summary_stats
from ier._validation import (
    MatrixLike,
    validate_integer,
    validate_matrix_input,
    validate_score_vectors,
)
from ier.types import IndexThresholdSourceMap, ScreenIndexSummary, ScreenResult


def _validate_min_flags(min_flags: int) -> int:
    """Return a validated respondent-level consensus threshold."""
    return validate_integer(min_flags, message="min_flags must be a positive integer", minimum=1)


def _tail_override_rule(
    label: str, fixed_thresholds: Mapping[str, float] | None = None
) -> Callable[[str], str | None]:
    """Reject presence-flagged indices and indices that already have a fixed cutoff."""

    def accepts(name: str) -> str | None:
        if INDEX_REGISTRY[name].flag_mode != "percentile":
            return f"{name} uses presence flagging and does not accept a {label}"
        if fixed_thresholds is not None and name in fixed_thresholds:
            return f"cannot set both a threshold and percentile for index: {name}"
        return None

    return accepts


_finite_threshold = numeric_override("threshold")
_tail_percentile = numeric_override(
    "percentile", "a finite number between 0 and 100", lambda value: 0 <= value <= 100
)
_threshold_rule = _tail_override_rule("threshold")


@dataclass(frozen=True)
class _ScreenRules:
    """Validated flagging and consensus settings shared by both screening entry points."""

    percentile: float
    min_flags: int
    min_valid_indices: int | None
    fixed_thresholds: dict[str, float]
    percentile_overrides: dict[str, float]


def _resolve_screen_rules(
    indices: list[str],
    percentile: float,
    min_flags: int,
    min_valid_indices: int | None,
    thresholds: Mapping[str, float] | None,
    percentiles: Mapping[str, float] | None,
) -> _ScreenRules:
    """Validate screening decisions for selected indices in one fixed order."""
    percentile = validate_percentile(percentile)
    min_flags = _validate_min_flags(min_flags)
    min_valid_indices = validate_min_valid_indices(min_valid_indices, len(indices))
    fixed_thresholds = resolve_index_overrides(
        thresholds,
        indices,
        label="threshold",
        convert=_finite_threshold,
        accepts=_threshold_rule,
    )
    percentile_overrides = resolve_index_overrides(
        percentiles,
        indices,
        label="percentile",
        convert=_tail_percentile,
        accepts=_tail_override_rule("percentile", fixed_thresholds),
    )
    return _ScreenRules(
        percentile, min_flags, min_valid_indices, fixed_thresholds, percentile_overrides
    )


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
        n_valid = int(np.count_nonzero(valid_mask))
        valid = score_arr if n_valid == n_respondents else score_arr[valid_mask]
        n_flagged = int(np.count_nonzero(flag_arr))
        flag_rate = n_flagged / n_valid if n_valid > 0 else float("nan")
        stats = observed_summary_stats(valid)
        summary[name] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "n_valid": n_valid,
            "n_unavailable": n_respondents - n_valid,
            "n_flagged": n_flagged,
            "flag_rate": flag_rate,
        }

    return flag_counts, valid_index_counts, summary


def _build_screen_result(
    scores: dict[str, np.ndarray],
    errors: dict[str, str],
    n_respondents: int,
    rules: _ScreenRules,
) -> ScreenResult:
    """Apply flagging and consensus rules to validated score vectors."""
    fixed_thresholds = rules.fixed_thresholds
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

        tail_percentile = rules.percentile_overrides.get(name, rules.percentile)
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
        if rules.min_valid_indices is None
        else valid_index_counts >= rules.min_valid_indices
    )
    consensus_flags = (flag_counts >= rules.min_flags) & consensus_eligible

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
        "min_flags": rules.min_flags,
        "min_valid_indices": rules.min_valid_indices,
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
    errors: Mapping[str, str] | None = None,
    n_respondents: int | None = None,
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
    - errors: Optional retained per-index soft failures. Failed indices count as
              selected for validation but never as available respondent scores.
    - n_respondents: Optional positive respondent count, checked against any score
                     vectors. Required with an empty score mapping when every
                     selected index failed; provide the retained errors as well.

    Returns:
    - The same structured ``ScreenResult`` contract as :func:`screen`, with an
      ``errors`` mapping containing any supplied failure provenance.

    Example:
        >>> from ier import screen, screen_scores
        >>> data = [
        ...     [1, 2, 3, 4, 5, 4],
        ...     [3, 3, 3, 3, 3, 3],
        ...     [5, 4, 3, 2, 1, 2],
        ...     [2, 5, 1, 4, 3, 2],
        ...     [4, 4, 5, 4, 4, 5],
        ...     [1, 5, 1, 5, 1, 5],
        ... ]
        >>> initial = screen(data, indices=["irv", "longstring"])
        >>> initial["flag_counts"].tolist()
        [0, 2, 0, 0, 0, 0]
        >>> looser = screen_scores(
        ...     initial["scores"],
        ...     percentiles={"irv": 50, "longstring": 50},
        ...     min_flags=1,
        ... )
        >>> looser["consensus_flags"].tolist()
        [False, True, False, False, True, False]
    """
    validated_scores, n_respondents = validate_score_vectors(scores, n_respondents=n_respondents)
    validate_index_names(list(validated_scores))
    retained_errors = validate_index_errors(errors, list(validated_scores))
    if not validated_scores and not retained_errors:
        raise ValueError("screening requires at least one scored or failed index")
    rules = _resolve_screen_rules(
        [*validated_scores, *retained_errors],
        percentile,
        min_flags,
        min_valid_indices,
        thresholds,
        percentiles,
    )
    return _build_screen_result(validated_scores, retained_errors, n_respondents, rules)


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

    Configure indices with a single ``IndexOptions`` via ``options=``. Its
    ``reverse_keyed_items`` are reverse-scored once, only for indices that use
    keyed responses, such as ``evenodd`` and ``guttman``; the others read ``x``.

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
              "semantic_syn", "semantic_ant", "infrequency", "missing_rate", "avgstr".
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
    - ValueError: If index names, fixed thresholds, or consensus settings are invalid,
                  including a bare string passed as ``indices``.
    - TypeError: If ``thresholds`` or ``percentiles`` is neither a mapping nor a
                 mapping-like object with ``items()``, such as a pandas Series.

    Example:
        >>> from ier import IndexOptions, screen
        >>> data = [
        ...     [1, 2, 3, 4, 5, 4],
        ...     [3, 3, 3, 3, 3, 3],
        ...     [5, 4, 3, 2, 1, 2],
        ...     [2, 5, 1, 4, 3, 2],
        ...     [4, 4, 5, 4, 4, 5],
        ...     [1, 5, 1, 5, 1, 5],
        ... ]
        >>> result = screen(data, options=IndexOptions(scale_min=1, scale_max=5))
        >>> result["indices_used"]
        ['irv', 'longstring', 'longstring_pattern', 'mahad', 'psychsyn', 'person_total',
         'markov', 'u3_poly', 'midpoint', 'acquiescence', 'guttman']
        >>> result["flag_counts"].tolist()
        [0, 3, 0, 0, 2, 1]
        >>> result["consensus_flags"].tolist()
        [False, True, False, False, True, False]
    """
    workers = validate_worker_count(workers)
    if not isinstance(strict, bool):
        raise ValueError("strict must be a boolean")
    if indices is None:
        indices = default_screen_indices()
    else:
        validate_index_names(indices)
    rules = _resolve_screen_rules(
        indices, percentile, min_flags, min_valid_indices, thresholds, percentiles
    )
    x_array = validate_matrix_input(x)
    scores, errors = score_registered_indices(
        x_array,
        indices,
        resolve_index_options(options),
        strict=strict,
        workers=workers,
        validated=True,
    )
    return _build_screen_result(scores, errors, x_array.shape[0], rules)
