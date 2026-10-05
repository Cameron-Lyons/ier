"""Tests for combining reusable registered-index score vectors."""

from typing import Any, cast

import numpy as np
import pytest

from ier import (
    ScreenResult,
    composite,
    composite_scores,
    composite_scores_summary,
    composite_summary,
    index_catalog,
    screen,
)


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
@pytest.mark.parametrize("standardize", [False, True])
def test_precomputed_composite_matches_direct_scoring(
    method: str,
    standardize: bool,
) -> None:
    rng = np.random.default_rng(20260803)
    data = rng.integers(1, 6, size=(120, 20)).astype(float)
    data[rng.random(data.shape) < 0.04] = np.nan
    indices = ["irv", "longstring", "person_total"]
    weights = {"irv": 2.0, "longstring": 0.75}
    initial = composite_summary(data, indices=indices)

    direct = composite(
        data,
        indices=indices,
        method=cast("Any", method),
        standardize=standardize,
        weights=weights,
        min_valid_indices=2,
    )
    reused = composite_scores(
        initial["indices"],
        method=cast("Any", method),
        standardize=standardize,
        weights=weights,
        min_valid_indices=2,
    )

    np.testing.assert_allclose(reused, direct, rtol=1e-14, atol=1e-14, equal_nan=True)


def test_precomputed_composite_applies_registered_directions_and_weights() -> None:
    scores = {
        "irv": np.array([0.1, 0.2, np.nan, 0.4]),
        "longstring": np.array([1.0, 3.0, 5.0, np.nan]),
    }

    unweighted = composite_scores(scores, standardize=False)
    weighted = composite_scores(
        scores,
        standardize=False,
        weights={"irv": 2.0, "longstring": 1.0},
    )

    np.testing.assert_allclose(unweighted, [0.45, 1.4, 5.0, -0.4], equal_nan=True)
    np.testing.assert_allclose(
        weighted,
        [(2.0 * -0.1 + 1.0) / 3.0, (2.0 * -0.2 + 3.0) / 3.0, 5.0, -0.4],
        equal_nan=True,
    )


def test_precomputed_composite_does_not_mutate_input_arrays() -> None:
    irv_scores = np.array([0.1, 0.2, 0.4, np.nan])
    longstring_scores = np.array([1.0, 3.0, 5.0, 7.0])
    before = {"irv": irv_scores.copy(), "longstring": longstring_scores.copy()}

    composite_scores(
        {"irv": irv_scores, "longstring": longstring_scores},
        standardize=True,
        weights={"irv": 2.0},
    )

    np.testing.assert_array_equal(irv_scores, before["irv"])
    np.testing.assert_array_equal(longstring_scores, before["longstring"])


def test_precomputed_composite_completeness_masks_under_supported_rows() -> None:
    result = composite_scores(
        {
            "irv": [0.1, np.nan, 0.4],
            "longstring": [8.0, 9.0, np.nan],
        },
        standardize=False,
        min_valid_indices=2,
    )

    np.testing.assert_allclose(result, [3.95, np.nan, np.nan], equal_nan=True)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({}, "at least one"),
        ({"unknown": [1.0]}, "invalid index"),
        ({"midpoint": [1.0]}, "invalid index"),
        ({"irv": []}, "cannot be empty"),
        ({"irv": [[1.0, 2.0]]}, "one-dimensional"),
        ({"irv": [1.0, float("inf")]}, "finite values or NaN"),
        ({"irv": [1.0, 2.0], "longstring": [1.0]}, "same respondent count"),
        ({"irv": ["not-a-number"]}, "numeric array"),
    ],
)
def test_invalid_precomputed_composite_mappings_raise(
    scores: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        composite_scores(scores)


def test_non_mapping_precomputed_composite_raises() -> None:
    with pytest.raises(TypeError, match="scores must be a mapping"):
        composite_scores(cast("Any", [np.array([1.0])]))


def test_precomputed_composite_settings_are_validated() -> None:
    scores = {"irv": np.array([0.1, 0.2]), "longstring": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="method must be"):
        composite_scores(scores, method=cast("Any", "best_subset"))
    with pytest.raises(ValueError, match="standardize must be a boolean"):
        composite_scores(scores, standardize=cast("Any", 1))
    with pytest.raises(ValueError, match="min_valid_indices"):
        composite_scores(scores, min_valid_indices=3)
    with pytest.raises(ValueError, match="positive finite"):
        composite_scores(scores, weights={"irv": 0.0})
    with pytest.raises(ValueError, match="not selected"):
        composite_scores(scores, weights={"mahad": 2.0})


def test_raw_composite_rejects_non_boolean_standardization() -> None:
    with pytest.raises(ValueError, match="standardize must be a boolean"):
        composite([[1.0, 2.0], [2.0, 1.0]], standardize=cast("Any", 1))


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
def test_precomputed_summary_reports_hand_calculated_coverage_and_scores(method: str) -> None:
    components = {
        "irv": np.array([0.0, 2.0, np.nan, 1.0]),
        "longstring": np.array([4.0, 1.0, 3.0, np.nan]),
    }
    for values in components.values():
        values.flags.writeable = False
    expected = {
        "mean": [4.0 / 3.0, -1.0, np.nan, np.nan],
        "sum": [4.0, -3.0, np.nan, np.nan],
        "max": [4.0, 1.0, np.nan, np.nan],
    }[method]
    result = composite_scores_summary(
        components,
        method=cast("Any", method),
        standardize=False,
        weights={"irv": 2.0},
        min_valid_indices=2,
    )
    np.testing.assert_allclose(result["composite"], expected, equal_nan=True)
    np.testing.assert_array_equal(result["valid_index_counts"], [2, 2, 1, 1])
    assert result["indices"]["irv"] is components["irv"]
    assert result["indices"]["longstring"] is components["longstring"]
    assert result["weights"] == {"irv": 2.0, "longstring": 1.0}
    assert result["indices_used"] == ["irv", "longstring"]
    assert result["errors"] == {}
    assert result["n_total"] == 4
    assert result["n_valid"] == 2
    assert result["mean"] == pytest.approx((expected[0] + expected[1]) / 2)
    assert result["std"] == pytest.approx(abs(expected[0] - expected[1]) / 2)
    assert result["min"] == expected[1]
    assert result["max"] == expected[0]


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
def test_precomputed_summary_preserves_sparse_calibration_policy(method: str) -> None:
    result = composite_scores_summary(
        {"irv": [0.1, np.nan, np.nan], "longstring": [1.0, 3.0, 5.0]},
        method=cast("Any", method),
    )
    # The existing single-observation policy retains its raw score and availability.
    np.testing.assert_array_equal(result["valid_index_counts"], [2, 1, 1])
    first = {
        "mean": (-0.1 - np.sqrt(1.5)) / 2,
        "sum": -0.1 - np.sqrt(1.5),
        "max": -0.1,
    }[method]
    np.testing.assert_allclose(result["composite"], [first, 0.0, np.sqrt(1.5)])
    assert result["n_valid"] == 3


def test_precomputed_summary_all_missing_results_have_unavailable_statistics() -> None:
    result = composite_scores_summary({"irv": [np.nan, np.nan]}, min_valid_indices=1)
    assert result["n_valid"] == 0
    np.testing.assert_array_equal(result["valid_index_counts"], [0, 0])
    assert np.isnan([result[key] for key in ["mean", "std", "min", "max"]]).all()


@pytest.fixture(scope="module")
def default_screen() -> ScreenResult:
    rng = np.random.default_rng(20261005)
    data = rng.integers(1, 6, size=(80, 12)).astype(float)
    data[:3] = 3.0
    data[rng.random(data.shape) < 0.03] = np.nan
    return screen(data)


def _composite_subset(mapping: dict[str, Any]) -> dict[str, Any]:
    catalog = index_catalog()
    return {name: value for name, value in mapping.items() if catalog[name]["composite_enabled"]}


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
def test_dropping_unsupported_indices_reuses_default_screen_scores(
    default_screen: ScreenResult, method: str
) -> None:
    scores = default_screen["scores"]
    subset = _composite_subset(scores)
    skipped = [name for name in scores if name not in subset]
    assert skipped == ["u3_poly", "midpoint", "acquiescence"]

    with pytest.raises(ValueError, match="invalid index 'u3_poly'"):
        composite_scores(scores, method=cast("Any", method))
    dropped = composite_scores(scores, method=cast("Any", method), unsupported="drop")
    np.testing.assert_array_equal(dropped, composite_scores(subset, method=cast("Any", method)))

    summary = composite_scores_summary(
        scores, method=cast("Any", method), weights={"mahad": 2.0}, unsupported="drop"
    )
    expected = composite_scores_summary(subset, method=cast("Any", method), weights={"mahad": 2.0})
    assert summary["indices_used"] == expected["indices_used"] == list(subset)
    assert summary["weights"] == expected["weights"]
    np.testing.assert_array_equal(summary["composite"], expected["composite"])
    np.testing.assert_array_equal(summary["valid_index_counts"], expected["valid_index_counts"])


def test_dropping_unsupported_indices_filters_failure_provenance(
    default_screen: ScreenResult,
) -> None:
    errors = {"mahad": "singular covariance", "acquiescence": "invalid polarity pairs"}
    scores = {
        name: values for name, values in default_screen["scores"].items() if name not in errors
    }
    subset = _composite_subset(scores)
    minimum = len(subset) + 1

    with pytest.raises(ValueError, match="invalid index 'u3_poly'"):
        composite_scores(scores, errors=errors)
    # The retained composite failure still counts as selected for weights and completeness.
    summary = composite_scores_summary(
        scores, errors=errors, weights={"mahad": 2.0}, min_valid_indices=minimum, unsupported="drop"
    )
    expected = composite_scores_summary(
        subset, errors={"mahad": "singular covariance"}, weights={"mahad": 2.0}
    )
    assert summary["errors"] == {"mahad": "singular covariance"}
    assert summary["weights"] == expected["weights"]
    assert summary["n_valid"] == 0
    np.testing.assert_array_equal(summary["valid_index_counts"], expected["valid_index_counts"])
    np.testing.assert_array_equal(
        composite_scores(scores, errors=errors, unsupported="drop"),
        composite_scores(subset, errors={"mahad": "singular covariance"}),
    )
    with pytest.raises(ValueError, match="weight index is not selected: acquiescence"):
        composite_scores(scores, errors=errors, weights={"acquiescence": 1.0}, unsupported="drop")


@pytest.mark.parametrize(
    ("scores", "errors", "message"),
    [
        ({"irv": [0.1, 0.2], "unknown": [1.0, 2.0]}, None, "invalid index 'unknown'"),
        ({"irv": [0.1, 0.2]}, {"unknown": "failed"}, "invalid index 'unknown'"),
        ({"irv": [0.1, 0.2]}, {"irv": "failed"}, "both scores and errors"),
        (
            {"u3_poly": [0.1, 0.2]},
            {"mad": "failed"},
            "^scores must contain at least one composite-enabled index after dropping "
            "unsupported indices: u3_poly$",
        ),
        ({}, None, "at least one registered index"),
    ],
)
def test_dropping_unsupported_indices_still_rejects_invalid_requests(
    scores: dict[str, Any], errors: dict[str, str] | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        composite_scores(scores, errors=errors, unsupported="drop")
    with pytest.raises(ValueError, match=message):
        composite_scores_summary(scores, errors=errors, unsupported="drop")


def test_unsupported_policy_is_validated() -> None:
    with pytest.raises(ValueError, match="unsupported must be 'error' or 'drop'"):
        composite_scores({"irv": [0.1, 0.2]}, unsupported=cast("Any", "skip"))


@pytest.mark.parametrize("weights", [[("irv", 2.0)], (("irv", 2.0),), "irv"])
def test_precomputed_non_mapping_weights_raise_type_error(weights: object) -> None:
    with pytest.raises(TypeError, match="^weights must be a mapping of registered index names"):
        composite_scores({"irv": [0.1, 0.2]}, weights=cast("Any", weights))
