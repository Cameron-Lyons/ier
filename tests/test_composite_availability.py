"""Complete, missing, and unavailable calibrated scores retain coverage semantics."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from ier import IndexOptions, composite_scores, composite_summary
from ier._registry import INDEX_REGISTRY
from ier.composite import _combine_scores


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("minimum", [None, 2])
@pytest.mark.parametrize(
    "order",
    [
        ("longstring", "irv", "missing_rate", "mahad"),
        ("irv", "missing_rate", "mahad", "longstring"),
        ("mahad", "irv", "longstring", "missing_rate"),
    ],
)
def test_complete_and_missing_components_preserve_scores_and_counts(
    method: str,
    standardize: bool,
    weighted: bool,
    minimum: int | None,
    order: tuple[str, ...],
) -> None:
    data = {
        "longstring": [1.1, 2.0, 3.0, 4.0, 5.0, 6.0],
        "irv": [1.0, np.nan, 3.0, np.nan, 5.0, np.nan],
        "missing_rate": [0.0, 0.0, np.nan, 3.0, 4.0, np.nan],
        "mahad": [np.nan] * 6,
    }
    scores = {name: np.repeat(data[name], 2)[::2] for name in order}
    for values in scores.values():
        values.flags.writeable = False
    weights = {"longstring": 1.0, "irv": 2.5, "missing_rate": 0.75} if weighted else None
    expected_counts = np.array([3, 2, 2, 2, 3, 1])
    prepared = {}
    for name, values in scores.items():
        available = values[~np.isnan(values)]
        prepared[name] = (
            (values - np.mean(available)) / np.std(available)
            if standardize and len(available) > 1
            else values
        )
    expected = np.full(6, np.nan)
    for row, count in enumerate(expected_counts):
        if minimum is not None and count < minimum:
            continue
        terms = []
        masses = []
        for name, values in prepared.items():
            if np.isnan(values[row]):
                continue
            weight = Fraction(weights.get(name, 1.0) if weights is not None else 1.0)
            masses.append(weight)
            terms.append(
                Fraction(float(values[row]))
                * weight
                * Fraction(INDEX_REGISTRY[name].composite_multiplier)
            )
        expected[row] = float(
            sum(terms) / sum(masses)
            if method == "mean"
            else max(terms)
            if method == "max"
            else sum(terms)
        )

    counts = np.empty(6, dtype=np.int_)
    multipliers = {name: INDEX_REGISTRY[name].composite_multiplier for name in scores}
    actual = _combine_scores(
        scores,
        {},
        method,
        standardize,
        weights,
        min_valid_indices=minimum,
        valid_counts_out=counts,
        multipliers=multipliers,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(counts, expected_counts)
    public = composite_scores(
        scores,
        method=method,
        standardize=standardize,
        weights=weights,
        min_valid_indices=minimum,
    )
    np.testing.assert_allclose(public, expected, rtol=1e-13, atol=1e-14)
    for name, values in scores.items():
        np.testing.assert_array_equal(values, data[name])


@pytest.mark.parametrize("method", ["mean", "sum", "max"])
@pytest.mark.parametrize("minimum", [None, 2])
def test_unavailable_calibration_retains_existing_scores_and_coverage(
    method: str, minimum: int | None
) -> None:
    maximum = float(np.finfo(float).max)
    data = [[maximum, maximum], [1.0, 2.0], [-maximum, -maximum]]
    with np.errstate(invalid="ignore"):
        result = composite_summary(
            data,
            indices=["mad", "longstring"],
            method=method,
            options=IndexOptions(mad_positive_items=[0], mad_negative_items=[1]),
            min_valid_indices=minimum,
        )
    expected = [1 / np.sqrt(2), -np.sqrt(2), 1 / np.sqrt(2)] if minimum is None else [np.nan] * 3
    np.testing.assert_allclose(result["composite"], expected, rtol=1e-13)
    np.testing.assert_array_equal(result["valid_index_counts"], [1, 1, 1])
