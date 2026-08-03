"""Tests for applying screen decisions to reusable score vectors."""

from typing import Any, cast

import numpy as np
import pytest

from ier import screen, screen_scores


def test_precomputed_scores_match_direct_screen_decisions() -> None:
    rng = np.random.default_rng(20260803)
    data = rng.integers(1, 6, size=(80, 16)).astype(float)
    data[rng.random(data.shape) < 0.05] = np.nan
    settings = {
        "percentile": 90.0,
        "min_flags": 2,
        "min_valid_indices": 2,
        "thresholds": {"longstring": 4.0},
        "percentiles": {"irv": 85.0},
    }

    direct = screen(data, indices=["irv", "longstring", "markov"], **settings)
    reused = screen_scores(direct["scores"], **settings)

    assert reused["indices_used"] == direct["indices_used"]
    assert reused["thresholds"] == direct["thresholds"]
    assert reused["threshold_sources"] == direct["threshold_sources"]
    assert reused["percentiles"] == direct["percentiles"]
    assert reused["summary"] == direct["summary"]
    assert reused["errors"] == {}
    for name in direct["indices_used"]:
        np.testing.assert_array_equal(reused["scores"][name], direct["scores"][name])
        np.testing.assert_array_equal(reused["flags"][name], direct["flags"][name])
    np.testing.assert_array_equal(reused["flag_counts"], direct["flag_counts"])
    np.testing.assert_array_equal(reused["valid_index_counts"], direct["valid_index_counts"])
    np.testing.assert_array_equal(reused["consensus_eligible"], direct["consensus_eligible"])
    np.testing.assert_array_equal(reused["consensus_flags"], direct["consensus_flags"])


def test_precomputed_float_arrays_are_reused_without_mutation() -> None:
    irv_scores = np.array([0.0, 0.5, 1.0, np.nan])
    longstring_scores = np.array([5.0, 4.0, 3.0, 2.0])
    irv_before = irv_scores.copy()
    longstring_before = longstring_scores.copy()

    result = screen_scores(
        {"irv": irv_scores, "longstring": longstring_scores},
        thresholds={"irv": 0.5, "longstring": 4.0},
        min_flags=1,
    )

    assert result["scores"]["irv"] is irv_scores
    assert result["scores"]["longstring"] is longstring_scores
    np.testing.assert_array_equal(irv_scores, irv_before)
    np.testing.assert_array_equal(longstring_scores, longstring_before)
    np.testing.assert_array_equal(result["flags"]["irv"], [True, True, False, False])
    np.testing.assert_array_equal(result["flags"]["longstring"], [True, True, False, False])


def test_precomputed_sequences_are_converted_to_float_vectors_in_order() -> None:
    result = screen_scores(
        {"longstring": [1, 2, 3], "irv": (3, 2, 1)},
        thresholds={"longstring": 2, "irv": 2},
        min_flags=1,
    )

    assert result["indices_used"] == ["longstring", "irv"]
    assert result["scores"]["longstring"].dtype == np.dtype(float)
    assert result["scores"]["irv"].dtype == np.dtype(float)
    np.testing.assert_array_equal(result["consensus_flags"], [False, True, True])


def test_precomputed_presence_scores_retain_presence_semantics() -> None:
    result = screen_scores({"onset": [np.nan, 12.0, np.nan, 4.0]}, min_flags=1)

    np.testing.assert_array_equal(result["flags"]["onset"], [False, True, False, True])
    np.testing.assert_array_equal(result["consensus_flags"], [False, True, False, True])
    assert result["thresholds"] == {"onset": None}
    assert result["threshold_sources"] == {"onset": "presence"}
    assert result["percentiles"] == {"onset": None}
    assert result["summary"]["onset"]["n_valid"] == 2
    assert result["summary"]["onset"]["flag_rate"] == 1.0


def test_precomputed_missing_scores_drive_consensus_eligibility() -> None:
    result = screen_scores(
        {
            "irv": [0.1, np.nan, 0.4],
            "longstring": [8.0, 9.0, np.nan],
        },
        thresholds={"irv": 0.2, "longstring": 8.0},
        min_flags=1,
        min_valid_indices=2,
    )

    np.testing.assert_array_equal(result["valid_index_counts"], [2, 1, 1])
    np.testing.assert_array_equal(result["consensus_eligible"], [True, False, False])
    np.testing.assert_array_equal(result["flag_counts"], [2, 1, 0])
    np.testing.assert_array_equal(result["consensus_flags"], [True, False, False])


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({}, "at least one"),
        ({"unknown": [1.0]}, "invalid index"),
        ({"irv": []}, "cannot be empty"),
        ({"irv": [[1.0, 2.0]]}, "one-dimensional"),
        ({"irv": [1.0, float("inf")]}, "finite values or NaN"),
        ({"irv": [1.0, 2.0], "longstring": [1.0]}, "same respondent count"),
        ({"irv": ["not-a-number"]}, "numeric array"),
    ],
)
def test_invalid_precomputed_score_mappings_raise(
    scores: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        screen_scores(scores)


def test_non_mapping_precomputed_scores_raise() -> None:
    with pytest.raises(TypeError, match="scores must be a mapping"):
        screen_scores(cast("Any", [np.array([1.0])]))


def test_precomputed_settings_use_regular_screen_validation() -> None:
    scores = {"irv": np.array([0.1, 0.2]), "longstring": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="min_flags"):
        screen_scores(scores, min_flags=0)
    with pytest.raises(ValueError, match="min_valid_indices"):
        screen_scores(scores, min_valid_indices=3)
    with pytest.raises(ValueError, match="not selected"):
        screen_scores(scores, thresholds={"markov": 1.0})
    with pytest.raises(ValueError, match="both a threshold and percentile"):
        screen_scores(
            scores,
            thresholds={"irv": 0.2},
            percentiles={"irv": 90.0},
        )
