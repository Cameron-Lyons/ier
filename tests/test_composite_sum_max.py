"""Weighted sums and maxima preserve representable results and missing-score policy."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ier import composite_scores
from ier._registry import INDEX_REGISTRY
from ier.composite import _combine_scores

_MAX = float(np.finfo(float).max)
_MIN = float(np.nextafter(0.0, 1.0))
_CASES = [
    (
        "sum",
        {"longstring": [2.0, 4.0], "irv": [2.0, 4.0]},
        {"longstring": _MAX, "irv": _MAX},
    ),
    ("sum", {"longstring": [_MAX], "mahad": [_MAX], "irv": [_MAX]}, None),
    ("sum", {"longstring": [_MAX], "irv": [-1e-300], "mahad": [-_MAX]}, None),
    (
        "sum",
        dict.fromkeys(("mahad", "missing_rate", "infrequency", "mad"), [_MIN]),
        dict.fromkeys(("mahad", "missing_rate", "infrequency", "mad"), 0.25),
    ),
    (
        "max",
        {"irv": [_MAX, _MAX, np.nan], "longstring": [0.0, 1.0, np.nan]},
        {"irv": _MAX, "longstring": 1.0},
    ),
    (
        "max",
        {"irv": [_MAX, _MAX], "longstring": [-1.0, -2.0]},
        {"irv": _MAX, "longstring": 1.0},
    ),
    (
        "max",
        {"irv": [_MAX, np.nan], "longstring": [_MIN, _MIN]},
        {"irv": _MAX, "longstring": 1.0},
    ),
    (
        "max",
        {"longstring": [4.0, -4.0, 0.0], "mahad": [2.0, -2.0, np.nan]},
        {"longstring": 0.5, "mahad": 1.0},
    ),
]


def _reference(
    scores: dict[str, np.ndarray],
    method: str,
    weights: dict[str, float] | None,
    minimum: int | None = None,
) -> np.ndarray:
    expected = np.full(len(next(iter(scores.values()))), np.nan)
    for row in range(len(expected)):
        terms = [
            Fraction(float(values[row]))
            * Fraction(weights.get(name, 1.0) if weights is not None else 1.0)
            * Fraction(INDEX_REGISTRY[name].composite_multiplier)
            for name, values in scores.items()
            if not np.isnan(values[row])
        ]
        if minimum is not None and len(terms) < minimum:
            continue
        if method == "sum":
            expected[row] = float(sum(terms))
        elif terms:
            expected[row] = float(max(terms))
    return expected


@pytest.mark.parametrize(("method", "data", "weights"), _CASES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("minimum", [None, 1, 2])
def test_finite_reductions_match_fraction(
    method: str,
    data: dict[str, list[float]],
    weights: dict[str, float] | None,
    strided: bool,
    minimum: int | None,
) -> None:
    scores = {name: np.array(values) for name, values in data.items()}
    if strided:
        scores = {name: np.repeat(values, 2)[::-2] for name, values in scores.items()}
    originals = {name: values.copy() for name, values in scores.items()}
    for values in scores.values():
        values.flags.writeable = False
    expected = _reference(scores, method, weights, minimum)
    actual = composite_scores(
        scores,
        method=method,
        weights=weights,
        standardize=False,
        min_valid_indices=minimum,
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0)
    for name in scores:
        np.testing.assert_array_equal(scores[name], originals[name])


@pytest.mark.parametrize("method", ["sum", "max"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("minimum", [None, 1])
def test_single_component_and_missing_policy(
    method: str, standardize: bool, minimum: int | None
) -> None:
    scores = np.array([1.0, 2.0, 3.0, np.nan])
    original = scores.copy()
    scores.flags.writeable = False
    actual = composite_scores(
        {"irv": scores},
        method=method,
        standardize=standardize,
        weights={"irv": 2.0},
        min_valid_indices=minimum,
    )
    prepared = (scores - 2) / np.std([1.0, 2.0, 3.0]) if standardize else scores
    expected = prepared * -2
    if method == "sum" and minimum is None:
        expected[np.isnan(expected)] = 0.0
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(scores, original)
    assert not np.shares_memory(actual, scores)


@pytest.mark.parametrize("method", ["sum", "max"])
@pytest.mark.parametrize("minimum", [None, 1])
def test_all_missing_results_keep_established_policy(method: str, minimum: int | None) -> None:
    expected = 0.0 if method == "sum" and minimum is None else np.nan
    for names in [("longstring",), ("longstring", "irv")]:
        actual = composite_scores(
            dict.fromkeys(names, [np.nan] * 7),
            method=method,
            min_valid_indices=minimum,
            weights=dict.fromkeys(names, _MAX),
        )
        np.testing.assert_array_equal(actual, np.full(7, expected))


@pytest.mark.parametrize("method", ["sum", "max"])
@pytest.mark.parametrize("name", ["longstring", "irv"])
def test_final_out_of_range_score_raises_clear_error(method: str, name: str) -> None:
    with pytest.raises(ValueError, match=rf"composite {method}.*finite float range.*index 0"):
        composite_scores({name: [2.0]}, method=method, standardize=False, weights={name: _MAX})


@pytest.mark.parametrize("method", ["sum", "max"])
def test_final_out_of_range_multi_component_score_raises(method: str) -> None:
    with pytest.raises(ValueError, match=rf"composite {method}.*finite float range"):
        composite_scores(
            {"longstring": [_MAX], "mahad": [_MAX]},
            method=method,
            standardize=False,
            weights={"longstring": 2.0, "mahad": 2.0},
        )


@pytest.mark.parametrize("method", ["sum", "max"])
def test_ineligible_respondents_do_not_raise_range_errors(method: str) -> None:
    scores = {"longstring": np.array([2.0, np.nan]), "irv": np.array([np.nan, 2.0])}
    counts = np.empty(2, dtype=int)
    actual = _combine_scores(
        scores,
        {},
        method,
        False,
        dict.fromkeys(scores, _MAX),
        min_valid_indices=2,
        valid_counts_out=counts,
        multipliers={"irv": -1.0},
    )
    np.testing.assert_array_equal(actual, [np.nan, np.nan])
    np.testing.assert_array_equal(counts, [1, 1])


def test_sum_cancellation_is_independent_of_component_order() -> None:
    scores = {"longstring": [_MAX], "irv": [-1e-300], "mahad": [-_MAX]}
    for order in itertools.permutations(scores):
        actual = composite_scores(
            {name: scores[name] for name in order},
            method="sum",
            standardize=False,
        )
        np.testing.assert_array_equal(actual, [1e-300])


def test_batched_max_preserves_winners_counts_and_eligibility() -> None:
    rng = np.random.default_rng(20260930)
    winners = rng.normal(size=10001)
    winners[rng.random(len(winners)) < 0.3] = np.nan
    scores = {"irv": np.full(len(winners), _MAX), "longstring": winners}
    counts = np.empty(len(winners), dtype=int)
    actual = _combine_scores(
        scores,
        {},
        "max",
        False,
        {"irv": _MAX},
        min_valid_indices=2,
        valid_counts_out=counts,
        multipliers={"irv": -1.0},
    )
    np.testing.assert_array_equal(actual, winners)
    np.testing.assert_array_equal(counts, 1 + (~np.isnan(winners)).astype(int))


def test_batched_sum_preserves_tiny_residuals_counts_and_eligibility() -> None:
    residuals = np.full(10001, 1e-300)
    residuals[::3] = np.nan
    scores = {
        "longstring": np.full(len(residuals), _MAX),
        "irv": -residuals,
        "mahad": np.full(len(residuals), -_MAX),
    }
    counts = np.empty(len(residuals), dtype=int)
    actual = _combine_scores(
        scores,
        {},
        "sum",
        False,
        min_valid_indices=3,
        valid_counts_out=counts,
        multipliers={"irv": -1.0},
    )
    np.testing.assert_array_equal(actual, residuals)
    np.testing.assert_array_equal(counts, 2 + (~np.isnan(residuals)).astype(int))


def test_standardized_sum_cancels_overflowing_weighted_components() -> None:
    scores = np.arange(7, dtype=float)
    actual = composite_scores(
        {"longstring": scores, "irv": scores},
        method="sum",
        weights={"longstring": _MAX, "irv": _MAX},
    )
    np.testing.assert_array_equal(actual, np.zeros(len(scores)))


def test_standardized_max_discards_oversized_sparse_component() -> None:
    actual = composite_scores(
        {"irv": [_MAX, np.nan], "longstring": [1.1, 1.1]},
        method="max",
        weights={"irv": _MAX},
    )
    np.testing.assert_array_equal(actual, [0.0, 0.0])


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    rows=st.lists(
        st.tuples(
            *[st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.just(np.nan))] * 3
        ),
        min_size=1,
        max_size=7,
    ),
    masses=st.tuples(*[st.floats(min_value=_MIN, allow_nan=False, allow_infinity=False)] * 3),
    method=st.sampled_from(["sum", "max"]),
)
def test_finite_inputs_match_fraction_or_raise_only_for_final_overflow(
    rows: list[tuple[float, float, float]], masses: tuple[float, float, float], method: str
) -> None:
    names = ("longstring", "mahad", "irv")
    scores = {name: np.array([row[index] for row in rows]) for index, name in enumerate(names)}
    weights = dict(zip(names, masses, strict=True))
    try:
        expected = _reference(scores, method, weights)
    except OverflowError:
        with pytest.raises(ValueError, match="finite float range"):
            composite_scores(scores, method=method, weights=weights, standardize=False)
    else:
        actual = composite_scores(scores, method=method, weights=weights, standardize=False)
        for value, reference in zip(actual, expected, strict=True):
            assert (math.isnan(value) and math.isnan(reference)) or math.isclose(
                value,
                reference,
                rel_tol=5e-14,
                abs_tol=0,
            )
