"""
Screening function that runs multiple IER detection indices at once.

Provides a single entry point for computing all available IER indices,
flagging suspected careless responders, and summarizing results.
"""

from typing import Any

import numpy as np

from ier._validation import MatrixLike, validate_matrix_input
from ier.acquiescence import acquiescence
from ier.evenodd import evenodd
from ier.irv import irv
from ier.longstring import longstring_pattern, longstring_scores
from ier.lz import lz
from ier.mad import add_mad_index_result
from ier.mahad import mahad
from ier.markov import markov
from ier.person_total import person_total
from ier.psychsyn import psychsyn
from ier.u3_poly import midpoint_responding, u3_poly

_DEFAULT_INDICES = [
    "irv",
    "longstring",
    "longstring_pattern",
    "mahad",
    "psychsyn",
    "person_total",
    "markov",
    "u3_poly",
    "midpoint",
    "acquiescence",
]

_OPTIONAL_INDICES = ["evenodd", "mad", "lz"]

_ALL_INDICES = _DEFAULT_INDICES + _OPTIONAL_INDICES


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
) -> dict[str, Any]:
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
        indices = list(_DEFAULT_INDICES)
    else:
        for idx in indices:
            if idx not in _ALL_INDICES:
                raise ValueError(f"invalid index '{idx}'. Valid options: {sorted(_ALL_INDICES)}")

    scores: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}

    def _mahad_scores() -> np.ndarray:
        mahad_result = mahad(x_array, na_rm=na_rm)
        if not isinstance(mahad_result, np.ndarray):
            raise ValueError("mahad returned non-array output")
        return mahad_result

    def _psychsyn_scores() -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            psyn_result = psychsyn(x_array, critval=psychsyn_critval, resample_na=na_rm)
        if not isinstance(psyn_result, np.ndarray):
            raise ValueError("psychsyn returned non-array output")
        return psyn_result

    index_handlers: dict[str, Any] = {
        "irv": lambda: irv(x_array, na_rm=na_rm),
        "longstring": lambda: longstring_scores(x_array, na_rm=na_rm),
        "longstring_pattern": lambda: longstring_pattern(x_array, na_rm=na_rm),
        "mahad": _mahad_scores,
        "psychsyn": _psychsyn_scores,
        "person_total": lambda: person_total(x_array, na_rm=na_rm),
        "markov": lambda: markov(x_array, na_rm=na_rm),
        "u3_poly": lambda: u3_poly(x_array, scale_min=scale_min, scale_max=scale_max),
        "midpoint": lambda: midpoint_responding(x_array, scale_min=scale_min, scale_max=scale_max),
        "acquiescence": lambda: acquiescence(
            x_array, scale_min=scale_min, scale_max=scale_max, na_rm=na_rm
        ),
        "lz": lambda: lz(x_array, na_rm=na_rm),
    }

    for idx in indices:
        if idx == "evenodd":
            if evenodd_factors is None:
                errors["evenodd"] = "evenodd_factors must be provided when using evenodd index"
                continue
            try:
                eo_result = evenodd(x_array, factors=evenodd_factors)
                if isinstance(eo_result, np.ndarray):
                    scores["evenodd"] = eo_result
            except ValueError as err:
                errors["evenodd"] = str(err)
            continue

        if idx == "mad":
            add_mad_index_result(
                scores,
                errors,
                x_array,
                mad_positive_items,
                mad_negative_items,
                mad_scale_max,
                na_rm,
            )
            continue

        handler = index_handlers.get(idx)
        if handler is None:
            continue
        try:
            scores[idx] = handler()
        except ValueError as err:
            errors[idx] = str(err)

    flags: dict[str, np.ndarray] = {}
    for name, score_arr in scores.items():
        valid_scores = score_arr[~np.isnan(score_arr)]
        if len(valid_scores) == 0:
            flags[name] = np.zeros(n_respondents, dtype=bool)
            continue

        cutoff = float(np.percentile(valid_scores, percentile))

        flag_arr = np.zeros(n_respondents, dtype=bool)
        valid_mask = ~np.isnan(score_arr)

        match name:
            case (
                "longstring"
                | "longstring_pattern"
                | "mahad"
                | "u3_poly"
                | "midpoint"
                | "acquiescence"
                | "mad"
            ):
                flag_arr[valid_mask] = score_arr[valid_mask] > cutoff
            case _:
                low_cutoff = float(np.percentile(valid_scores, 100.0 - percentile))
                flag_arr[valid_mask] = score_arr[valid_mask] < low_cutoff

        flags[name] = flag_arr

    flag_matrix = (
        np.column_stack(list(flags.values())) if flags else np.zeros((n_respondents, 0), dtype=bool)
    )
    flag_counts: np.ndarray = np.sum(flag_matrix, axis=1)

    summary: dict[str, dict[str, float]] = {}
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
