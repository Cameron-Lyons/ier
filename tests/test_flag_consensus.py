"""Tests for cross-domain respondent-level flag consensus."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ier import flag_consensus, response_time_score_flags, screen_scores


def test_flag_consensus_combines_screen_and_response_time_with_coverage() -> None:
    registered_scores = {
        "irv": np.asarray([0.1, 0.8, np.nan, 0.2]),
        "longstring": np.asarray([2.0, 8.0, 7.0, np.nan]),
    }
    screened = screen_scores(
        registered_scores,
        thresholds={"irv": 0.5, "longstring": 5.0},
        min_flags=1,
    )
    timing_scores = np.asarray([0.5, 0.8, np.nan, 0.4])
    timing_flags = response_time_score_flags(timing_scores, threshold=0.5)

    combined = flag_consensus(
        {**screened["flags"], "response_time": timing_flags},
        scores={**screened["scores"], "response_time": timing_scores},
        min_flags=2,
        min_valid_signals=3,
    )
    registered_only = flag_consensus(
        screened["flags"],
        scores=screened["scores"],
        min_flags=1,
    )

    assert combined["min_flags"] == 2
    assert combined["min_valid_signals"] == 3
    assert combined["n_signals"] == 3
    assert combined["n_respondents"] == 4
    np.testing.assert_array_equal(combined["flag_counts"], [2, 1, 1, 2])
    np.testing.assert_array_equal(combined["valid_signal_counts"], [3, 3, 1, 2])
    np.testing.assert_array_equal(combined["consensus_eligible"], [True, True, False, False])
    np.testing.assert_array_equal(combined["consensus_flags"], [True, False, False, False])
    np.testing.assert_array_equal(registered_only["flag_counts"], screened["flag_counts"])
    np.testing.assert_array_equal(
        registered_only["valid_signal_counts"],
        screened["valid_index_counts"],
    )
    np.testing.assert_array_equal(
        registered_only["consensus_flags"],
        screened["consensus_flags"],
    )


def test_flag_consensus_treats_scores_as_optional_availability_without_mutation() -> None:
    flags = {
        "pattern": np.asarray([True, False, False]),
        "timing": np.asarray([False, True, True]),
    }
    scores = {"pattern": np.asarray([1.0, np.nan, 3.0])}
    original_flags = {name: values.copy() for name, values in flags.items()}
    original_scores = {name: values.copy() for name, values in scores.items()}

    result = flag_consensus(
        flags,
        scores=scores,
        min_flags=1,
        min_valid_signals=2,
    )

    np.testing.assert_array_equal(result["flag_counts"], [1, 1, 1])
    np.testing.assert_array_equal(result["valid_signal_counts"], [2, 1, 2])
    np.testing.assert_array_equal(result["consensus_eligible"], [True, False, True])
    np.testing.assert_array_equal(result["consensus_flags"], [True, False, True])
    for name, values in flags.items():
        np.testing.assert_array_equal(values, original_flags[name])
    for name, values in scores.items():
        np.testing.assert_array_equal(values, original_scores[name])


@pytest.mark.parametrize(
    ("flags", "scores", "kwargs", "message"),
    [
        ({}, None, {}, "at least one signal"),
        ({" ": [True]}, None, {}, "nonblank strings"),
        ({"signal": [0, 1]}, None, {}, "Boolean array"),
        ({"a": [True], "b": [False, True]}, None, {}, "same respondent count"),
        ({"a": [True]}, {"other": [1.0]}, {}, "not a selected flag signal"),
        ({"a": [True]}, {"a": [1.0, 2.0]}, {}, "same respondent count"),
        ({"a": [True, False]}, {"a": [np.nan, 1.0]}, {}, "false where scores"),
        ({"a": [True]}, None, {"min_flags": 0}, "min_flags"),
        ({"a": [True]}, None, {"min_valid_signals": 0}, "min_valid_signals"),
        ({"a": [True]}, None, {"min_valid_signals": 2}, "cannot exceed"),
    ],
)
def test_flag_consensus_rejects_invalid_contracts(
    flags: Any,
    scores: Any,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        flag_consensus(flags, scores=scores, **kwargs)


def test_flag_consensus_requires_mapping_inputs() -> None:
    with pytest.raises(TypeError, match="flags must be a mapping"):
        flag_consensus([np.asarray([True])])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scores must be a mapping"):
        flag_consensus({"signal": [True]}, scores=[1.0])  # type: ignore[arg-type]
