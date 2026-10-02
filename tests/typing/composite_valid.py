"""Real consumer workflows whose public return types must remain precise."""

from typing import assert_type

import numpy as np

from ier import (
    CompositeMethod,
    CompositeSummary,
    IndexOptions,
    MatrixLike,
    composite,
    composite_flag,
    composite_probability,
    composite_scores,
    composite_scores_summary,
    composite_summary,
)


def combine_responses(responses: MatrixLike, method: CompositeMethod) -> np.ndarray:
    """Default calls can be assigned and used as arrays without casts."""
    scores: np.ndarray = composite(responses, method=method)
    assert_type(scores, np.ndarray)
    assert_type(composite(responses, return_diagnostics=False), np.ndarray)
    assert_type(composite_probability(responses), np.ndarray)
    assert_type(composite_probability(responses, return_diagnostics=False), np.ndarray)
    scores_with_errors = composite(
        responses,
        indices=["irv", "longstring"],
        standardize=False,
        options=IndexOptions(),
        weights={"irv": 2.0},
        min_valid_indices=1,
        return_diagnostics=True,
        strict=False,
        workers=2,
    )
    assert_type(scores_with_errors, tuple[np.ndarray, dict[str, str]])
    assert_type(
        composite_probability(responses, return_diagnostics=True),
        tuple[np.ndarray, dict[str, str]],
    )
    return scores.reshape(-1)


def flag_responses(responses: MatrixLike) -> tuple[np.ndarray, np.ndarray]:
    """Flagging keeps its two- or three-element tuple contract."""
    result: tuple[np.ndarray, np.ndarray] = composite_flag(responses)
    assert_type(result, tuple[np.ndarray, np.ndarray])
    assert_type(
        composite_flag(responses, threshold=1.0, return_diagnostics=False),
        tuple[np.ndarray, np.ndarray],
    )
    assert_type(
        composite_flag(responses, percentile=99.0, return_diagnostics=True),
        tuple[np.ndarray, np.ndarray, dict[str, str]],
    )
    scores, flags = result
    return scores[flags], flags


def combine_with_runtime_setting(responses: MatrixLike, diagnostics: bool) -> None:
    """A Boolean chosen at runtime retains an honest union of return shapes."""
    assert_type(
        composite(responses, return_diagnostics=diagnostics),
        np.ndarray | tuple[np.ndarray, dict[str, str]],
    )
    assert_type(
        composite_flag(responses, return_diagnostics=diagnostics),
        tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, str]],
    )
    result = composite_probability(responses, return_diagnostics=diagnostics)
    assert_type(result, np.ndarray | tuple[np.ndarray, dict[str, str]])
    if isinstance(result, tuple):
        scores, errors = result
        assert_type(scores, np.ndarray)
        assert_type(errors, dict[str, str])
    else:
        assert_type(result, np.ndarray)


def reuse_components(responses: MatrixLike) -> np.ndarray:
    """The fresh-to-retained-score workflow remains usable with public types."""
    details = composite_summary(responses, indices=["irv", "longstring"])
    assert_type(details, CompositeSummary)
    result = composite_scores(details["indices"], errors=details["errors"])
    assert_type(result, np.ndarray)
    assert_type(
        composite_scores_summary(details["indices"], errors=details["errors"]), CompositeSummary
    )
    return result
