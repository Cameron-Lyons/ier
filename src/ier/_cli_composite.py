"""Shared validation for composite command-line serializers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


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
