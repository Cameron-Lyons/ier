"""Validated composite and response-time reports shared by command-line serializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ier.types import ResponseTimeFlagDirection, ResponseTimeMetric


def validate_respondent_ids(n_respondents: int, respondent_ids: list[str] | None) -> None:
    """Validate optional respondent identifiers against the result length."""
    if respondent_ids is not None and len(respondent_ids) != n_respondents:
        raise ValueError("respondent ID count must match result length")


def validate_composite_components(
    n_respondents: int,
    component_scores: Mapping[str, np.ndarray] | None,
    valid_index_counts: np.ndarray | None,
) -> None:
    """Validate optional respondent-aligned composite detail arrays."""
    if (component_scores is None) != (valid_index_counts is None):
        raise ValueError("component scores and valid index counts must be provided together")
    if component_scores is None:
        return
    assert valid_index_counts is not None
    if len(valid_index_counts) != n_respondents:
        raise ValueError("valid index count length must match composite score length")
    for name, values in component_scores.items():
        if len(values) != n_respondents:
            raise ValueError(f"component score length for {name} must match composite score length")


def validate_composite_flags(
    n_respondents: int,
    flags: np.ndarray | None,
    flag_threshold: float | None,
    flag_percentile: float | None,
) -> None:
    """Validate optional respondent-aligned composite flag output."""
    if (flags is None) != (flag_threshold is None):
        raise ValueError("composite flags and threshold must be provided together")
    if flags is None:
        if flag_percentile is not None:
            raise ValueError("composite percentile requires flags and threshold")
        return
    assert flag_threshold is not None
    if len(flags) != n_respondents:
        raise ValueError("composite flag length must match composite score length")
    if not np.isfinite(flag_threshold):
        raise ValueError("composite flag threshold must be finite")
    if flag_percentile is not None and not (
        np.isfinite(flag_percentile) and 0.0 <= flag_percentile <= 100.0
    ):
        raise ValueError("composite flag percentile must be between 0 and 100")


def validate_composite_probabilities(
    n_respondents: int,
    probabilities: np.ndarray | None,
) -> None:
    """Validate optional respondent-aligned logistic composite values."""
    if probabilities is not None and len(probabilities) != n_respondents:
        raise ValueError("composite probability length must match composite score length")


@dataclass(frozen=True)
class CompositeReport:
    """One composite result, validated once for every output format.

    Optional details are absent unless requested: components and their
    availability counts travel together, as do flags and their threshold.
    """

    scores: np.ndarray
    method: str
    respondent_ids: list[str] | None = None
    weights: Mapping[str, float] | None = None
    min_valid_indices: int | None = None
    errors: Mapping[str, str] = field(default_factory=dict)
    component_scores: Mapping[str, np.ndarray] | None = None
    valid_index_counts: np.ndarray | None = None
    standardized: bool = True
    flags: np.ndarray | None = None
    flag_threshold: float | None = None
    flag_percentile: float | None = None
    probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_respondents = len(self.scores)
        validate_respondent_ids(n_respondents, self.respondent_ids)
        validate_composite_components(n_respondents, self.component_scores, self.valid_index_counts)
        validate_composite_flags(
            n_respondents, self.flags, self.flag_threshold, self.flag_percentile
        )
        validate_composite_probabilities(n_respondents, self.probabilities)


@dataclass(frozen=True)
class ResponseTimeReport:
    """One flagged response-time result, validated once for every output format."""

    scores: np.ndarray
    flags: np.ndarray
    metric: ResponseTimeMetric
    direction: ResponseTimeFlagDirection
    cutoff: float
    respondent_ids: list[str] | None = None

    def __post_init__(self) -> None:
        validate_respondent_ids(len(self.scores), self.respondent_ids)
        if len(self.flags) != len(self.scores):
            raise ValueError("response-time flag length must match score length")
