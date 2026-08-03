"""Tests for combining reusable registered-index score vectors."""

from typing import Any, cast

import numpy as np
import pytest

from ier import composite, composite_scores, composite_summary


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
