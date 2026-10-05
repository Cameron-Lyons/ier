"""Composite means retain finite weighted values, coverage, and caller ownership."""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ier import composite_scores
from ier._composite_reductions import standardize_index_scores
from ier._registry import INDEX_REGISTRY
from ier.composite import _combine_scores

_MAX = float(np.finfo(float).max)
_MIN = float(np.nextafter(0.0, 1.0))
_CASES = [
    (
        {"longstring": [1.0, 2.0, 3.0], "mahad": [1.0, 2.0, 3.0]},
        {"longstring": _MAX, "mahad": _MAX},
    ),
    ({"longstring": [_MAX, _MAX / 2], "mahad": [_MAX, _MAX / 2]}, None),
    (
        {"longstring": [1e-300, 2e-300], "mahad": [1e-300, 2e-300]},
        {"longstring": 1e-300, "mahad": 1e-300},
    ),
    (
        {"longstring": [np.nan, 1e-300], "mahad": [3.0, np.nan]},
        {"longstring": _MIN, "mahad": _MAX},
    ),
    (
        {"longstring": [_MAX, 1.0], "mahad": [_MIN, 2.0]},
        {"longstring": _MIN, "mahad": _MAX},
    ),
    ({"longstring": [_MAX], "irv": [-1e-300], "mahad": [-_MAX]}, None),
    ({"longstring": [1e100], "irv": [-1e-100], "mahad": [-1e100]}, None),
    ({"longstring": [_MAX, -_MAX], "mahad": [-_MAX, _MAX]}, {"longstring": 0.1, "mahad": 0.1}),
    (
        {"longstring": [_MAX, _MIN, 0.0], "mahad": [_MAX / 2, 2 * _MIN, _MIN]},
        {"longstring": _MIN, "mahad": _MIN},
    ),
    (
        {"longstring": [1.0, -1.0], "mahad": [-1.0, 1.0]},
        {"longstring": 1.0, "mahad": math.nextafter(1.0, 0.0)},
    ),
]


def _summation_tolerance(
    scores: dict[str, np.ndarray], weights: dict[str, float], row: int
) -> float:
    """Return the float64 summation error bound for one weighted mean.

    Only catastrophic cancellation is repaired exactly. Ordinary rows retain the
    usual bound of a few rounding units of the largest absolute weighted terms.
    """
    terms, mass = Fraction(0), Fraction(0)
    for name, values in scores.items():
        if not np.isnan(values[row]):
            weight = Fraction(weights[name])
            terms += abs(Fraction(float(values[row])) * weight)
            mass += weight
    return 8 * float(np.finfo(float).eps) * float(terms / mass) if mass else 0.0


def _reference(
    scores: dict[str, np.ndarray], weights: dict[str, float] | None, minimum: int = 1
) -> np.ndarray:
    expected = np.full(len(next(iter(scores.values()))), np.nan)
    for row in range(len(expected)):
        numerator, denominator = Fraction(0), Fraction(0)
        count = 0
        for name, values in scores.items():
            if np.isnan(values[row]):
                continue
            weight = Fraction(weights.get(name, 1.0) if weights is not None else 1.0)
            direction = Fraction(INDEX_REGISTRY[name].composite_multiplier)
            numerator += Fraction(float(values[row])) * weight * direction
            denominator += weight
            count += 1
        if count >= minimum:
            expected[row] = float(numerator / denominator)
    return expected


@pytest.mark.parametrize(("data", "weights"), _CASES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("minimum", [1, 2])
def test_weighted_means_match_exact_reference(
    data: dict[str, list[float]], weights: dict[str, float] | None, strided: bool, minimum: int
) -> None:
    scores = {name: np.array(values) for name, values in data.items()}
    if strided:
        scores = {name: np.repeat(values, 2)[::-2] for name, values in scores.items()}
    originals = {name: values.copy() for name, values in scores.items()}
    for values in scores.values():
        values.flags.writeable = False
    expected = _reference(scores, weights, minimum)
    actual = composite_scores(scores, standardize=False, weights=weights, min_valid_indices=minimum)
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0)
    for name in scores:
        np.testing.assert_array_equal(scores[name], originals[name])


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("scale", [_MIN, 1e-300, 1.0, 1e300, _MAX / 4])
def test_common_weight_scale_preserves_means_with_missing_scores(
    scale: float, standardize: bool
) -> None:
    scores = {
        "longstring": np.array([1.0, 2.0, np.nan, 4.0, np.nan]),
        "mahad": np.array([2.0, np.nan, 3.0, 4.0, np.nan]),
        "irv": np.array([np.nan, 0.5, 1.0, 2.0, np.nan]),
    }
    weights = {name: scale * (index + 1) for index, name in enumerate(scores)}
    prepared = {
        name: standardize_index_scores(values) if standardize else values
        for name, values in scores.items()
    }
    expected = _reference(prepared, weights)
    actual = composite_scores(scores, weights=weights, standardize=standardize)
    np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0)


@pytest.mark.parametrize("name", ["longstring", "irv"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("weight", [_MIN, 1.0, _MAX])
def test_single_component_mean_omits_its_weight(
    name: str, standardize: bool, weight: float
) -> None:
    scores = np.array([np.nan, _MAX / 2, _MAX, np.nan])
    scores.flags.writeable = False
    prepared = standardize_index_scores(scores) if standardize else scores
    expected = prepared * INDEX_REGISTRY[name].composite_multiplier
    actual = composite_scores({name: scores}, standardize=standardize, weights={name: weight})
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, scores)


def test_cancellation_is_independent_of_component_order() -> None:
    scores = {
        "longstring": np.array([_MAX]),
        "irv": np.array([-1e-300]),
        "mahad": np.array([-_MAX]),
    }
    expected = _reference(scores, None)
    for order in itertools.permutations(scores):
        actual = composite_scores({name: scores[name] for name in order}, standardize=False)
        np.testing.assert_array_equal(actual, expected)


def test_scaled_repair_preserves_counts_across_many_respondents() -> None:
    rng = np.random.default_rng(20260930)
    scores = {name: rng.normal(size=10001) for name in ("longstring", "mahad", "irv")}
    for values in scores.values():
        values[rng.random(len(values)) < 0.3] = np.nan
    weights = dict.fromkeys(scores, _MAX / 2)
    counts = np.empty(10001, dtype=np.int_)
    multipliers = {name: INDEX_REGISTRY[name].composite_multiplier for name in scores}
    actual = _combine_scores(
        scores,
        {},
        "mean",
        False,
        weights,
        min_valid_indices=2,
        valid_counts_out=counts,
        multipliers=multipliers,
    )
    expected_counts = sum((~np.isnan(values)).astype(int) for values in scores.values())
    np.testing.assert_array_equal(counts, expected_counts)
    sampled = np.linspace(0, len(counts) - 1, 32, dtype=int)
    expected = _reference({name: values[sampled] for name, values in scores.items()}, weights, 2)
    np.testing.assert_allclose(actual[sampled], expected, rtol=5e-14, atol=0)


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
)
def test_finite_weighted_means_match_fraction_oracle(
    rows: list[tuple[float, float, float]], masses: tuple[float, float, float]
) -> None:
    names = ("longstring", "mahad", "irv")
    scores = {name: np.array([row[index] for row in rows]) for index, name in enumerate(names)}
    weights = dict(zip(names, masses, strict=True))
    actual = composite_scores(scores, weights=weights, standardize=False)
    expected = _reference(scores, weights)
    for row, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        assert (math.isnan(value) and math.isnan(reference)) or math.isclose(
            value,
            reference,
            rel_tol=5e-14,
            abs_tol=_summation_tolerance(scores, weights, row),
        )
