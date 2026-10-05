"""Item-step person-fit statistics against brute-force definitions, the registry, and the CLI."""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import numpy as np
import pytest

import ier
from ier import (
    IndexOptions,
    composite_summary,
    gpoly,
    gpoly_flag,
    guttman,
    ht,
    ht_flag,
    index_catalog,
    screen,
    u3poly,
    u3poly_flag,
)
from ier import person_fit as person_fit_module
from ier._flagging import threshold_flags
from ier._registry import INDEX_REGISTRY
from ier.cli import main

if TYPE_CHECKING:
    from collections.abc import Callable

Step = tuple[int, int]
_DOCS = Path(__file__).resolve().parents[1] / "docs"


# Independent oracles written straight from the definitions (Molenaar, 1991;
# Emons, 2008; Sijtsma & Meijer, 1992). They use exact rationals, explicit
# enumeration, and no code from ier.person_fit.


def _popularity(x: np.ndarray, item: int, level: int) -> Fraction:
    observed = x[~np.isnan(x[:, item]), item]
    return (
        Fraction(int(np.sum(observed >= level)), len(observed)) if len(observed) else Fraction(-1)
    )


def _step_order(x: np.ndarray, n_steps: int) -> list[Step]:
    """Steps by descending popularity; a stable sort keeps PerFit's item-major ties."""
    steps = [(item, level) for item in range(x.shape[1]) for level in range(1, n_steps + 1)]
    return sorted(steps, key=lambda step: -_popularity(x, *step))


def _step_pair_errors(row: np.ndarray, order: list[Step]) -> int:
    """Count observed step pairs with a failed popular step before a passed one."""
    errors = 0
    for first, (item, level) in enumerate(order):
        if math.isnan(row[item]) or row[item] >= level:
            continue
        for later_item, later_level in order[first + 1 :]:
            if not math.isnan(row[later_item]) and row[later_item] >= later_level:
                errors += 1
    return errors


def _nested_patterns(n_items: int, n_steps: int) -> list[np.ndarray]:
    return [
        np.array(pattern, dtype=float)
        for pattern in itertools.product(range(n_steps + 1), repeat=n_items)
    ]


def _gpoly_oracle(x: np.ndarray, n_steps: int, *, normalize: bool) -> np.ndarray:
    order = _step_order(x, n_steps)
    maxima: dict[int, int] = {}
    for pattern in _nested_patterns(x.shape[1], n_steps):
        total = int(pattern.sum())
        maxima[total] = max(maxima.get(total, 0), _step_pair_errors(pattern, order))
    scores = []
    for row in x:
        missing = np.isnan(row)
        if missing.all() or (normalize and missing.any()):
            scores.append(math.nan)
            continue
        errors = _step_pair_errors(row, order)
        if not normalize:
            scores.append(float(errors))
            continue
        maximum = maxima[int(row.sum())]
        scores.append(errors / maximum if maximum else math.nan)
    return np.array(scores)


def _logit_weights(x: np.ndarray, n_steps: int) -> np.ndarray:
    weights = np.zeros((x.shape[1], n_steps))
    for item in range(x.shape[1]):
        observed = x[~np.isnan(x[:, item]), item]
        for level in range(1, n_steps + 1):
            if len(observed):
                passed = min(max(float(np.sum(observed >= level)), 0.5), len(observed) - 0.5)
                weights[item, level - 1] = math.log(passed / (len(observed) - passed))
    return weights


def _u3poly_oracle(x: np.ndarray, n_steps: int) -> np.ndarray:
    weights = _logit_weights(x, n_steps)

    def weight(pattern: np.ndarray) -> float:
        return math.fsum(
            weights[item, level]
            for item in range(len(pattern))
            for level in range(int(pattern[item]))
        )

    largest: dict[int, float] = {}
    smallest: dict[int, float] = {}
    for pattern in _nested_patterns(x.shape[1], n_steps):
        total, value = int(pattern.sum()), weight(pattern)
        largest[total] = max(largest.get(total, -math.inf), value)
        smallest[total] = min(smallest.get(total, math.inf), value)
    scores = []
    for row in x:
        if np.isnan(row).any():
            scores.append(math.nan)
            continue
        total = int(row.sum())
        spread = largest[total] - smallest[total]
        scores.append((largest[total] - weight(row)) / spread if spread > 1e-9 else math.nan)
    return np.array(scores)


def _ht_oracle(x: np.ndarray) -> np.ndarray:
    """Naive O(N^2 J) Ht in exact integers over the complete respondents."""
    complete = ~np.isnan(x).any(axis=1)
    rows = [[int(value) for value in row] for row in x[complete]]
    n_items = x.shape[1]
    totals = [sum(row) for row in rows]
    scores = np.full(len(x), np.nan)
    values = []
    for person, row in enumerate(rows):
        covariance = maximum = 0
        for other, other_row in enumerate(rows):
            if other == person:
                continue
            product = totals[person] * totals[other]
            covariance += (
                n_items * sum(a * b for a, b in zip(row, other_row, strict=True)) - product
            )
            maximum += n_items * min(totals[person], totals[other]) - product
        values.append(float(Fraction(covariance, maximum)) if maximum > 0 else math.nan)
    scores[complete] = values
    return scores


def _random_steps(seed: int, *, missing: bool = False) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    n_items, n_steps = int(rng.integers(2, 5)), int(rng.integers(1, 4))
    x = rng.integers(0, n_steps + 1, size=(int(rng.integers(3, 12)), n_items)).astype(float)
    if missing:
        x[rng.random(x.shape) < 0.15] = np.nan
    # Pin both scale endpoints so the inferred scale has n_steps steps.
    x[0, :2] = [0, n_steps]
    return x, n_steps


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "missing"])
@pytest.mark.parametrize("seed", range(40))
def test_gpoly_matches_step_pair_enumeration_and_brute_force_maxima(
    seed: int, missing: bool
) -> None:
    x, n_steps = _random_steps(seed, missing=missing)

    np.testing.assert_array_equal(
        gpoly(x, normalize=False), _gpoly_oracle(x, n_steps, normalize=False)
    )
    np.testing.assert_array_equal(gpoly(x), _gpoly_oracle(x, n_steps, normalize=True))


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "missing"])
@pytest.mark.parametrize("seed", range(40))
def test_u3poly_matches_brute_force_weight_extremes(seed: int, missing: bool) -> None:
    x, n_steps = _random_steps(seed, missing=missing)

    np.testing.assert_allclose(u3poly(x), _u3poly_oracle(x, n_steps), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "missing"])
@pytest.mark.parametrize("seed", range(40))
def test_ht_matches_naive_integer_covariances(seed: int, missing: bool) -> None:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(int(rng.integers(2, 15)), int(rng.integers(2, 7)))).astype(float)
    if missing:
        x[rng.random(x.shape) < 0.1] = np.nan

    np.testing.assert_array_equal(ht(x), _ht_oracle(x))


def test_ht_is_the_ratio_of_person_covariances_to_their_maxima() -> None:
    rng = np.random.default_rng(3)
    x = (rng.random((40, 9)) < np.linspace(0.9, 0.2, 9)).astype(float)
    proportions = x.mean(axis=1)
    covariances = np.cov(x, bias=True)
    maxima = np.minimum.outer(proportions, proportions) - np.outer(proportions, proportions)
    np.fill_diagonal(covariances, 0.0)
    np.fill_diagonal(maxima, 0.0)

    np.testing.assert_allclose(ht(x), covariances.sum(axis=1) / maxima.sum(axis=1), rtol=1e-12)


@pytest.mark.parametrize("maximize", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("seed", range(25))
def test_nested_step_extremes_match_enumeration(seed: int, maximize: bool) -> None:
    rng = np.random.default_rng(seed)
    n_items, n_steps = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    # Sorted increments make each row convex; reversing them makes it concave.
    increments = np.sort(rng.integers(-20, 21, size=(n_items, n_steps)), axis=1)
    if not maximize:
        increments = increments[:, ::-1]
    table = np.zeros((n_items, n_steps + 1), dtype=np.int64)
    table[:, 1:] = np.cumsum(increments, axis=1)
    expected: dict[int, int] = {}
    choose = max if maximize else min
    for pattern in itertools.product(range(n_steps + 1), repeat=n_items):
        value = int(sum(table[item, level] for item, level in enumerate(pattern)))
        total = sum(pattern)
        expected[total] = choose(expected.get(total, value), value)

    extremes = person_fit_module._nested_step_extremes(table, maximize=maximize)

    assert extremes.tolist() == [expected[total] for total in range(n_items * n_steps + 1)]


def test_nested_step_extremes_without_steps_are_zero() -> None:
    table = np.zeros((3, 1), dtype=np.int64)
    for maximize in (True, False):
        assert person_fit_module._nested_step_extremes(table, maximize=maximize).tolist() == [0]


def test_dichotomous_gpoly_is_guttman_normalized_by_its_maximum() -> None:
    rng = np.random.default_rng(8)
    x = (rng.random((60, 7)) < np.linspace(0.85, 0.15, 7)).astype(float)
    x[0], x[1] = 0, 1
    raw = guttman(x, normalize=False)
    correct = x.sum(axis=1)
    expected = np.divide(
        raw,
        correct * (7 - correct),
        out=np.full(len(x), np.nan),
        where=(correct > 0) & (correct < 7),
    )

    np.testing.assert_array_equal(gpoly(x, normalize=False), raw)
    np.testing.assert_array_equal(gpoly(x, ncat=2), expected)
    assert np.isnan(gpoly(x)[:2]).all()


def test_perfect_patterns_score_exactly_zero_and_reversals_exactly_one() -> None:
    patterns = np.array(
        [
            [3, 3, 2, 1, 0],
            [3, 2, 1, 1, 0],
            [2, 1, 1, 0, 0],
            [3, 3, 3, 2, 1],
            [1, 1, 0, 0, 0],
            [3, 3, 2, 2, 1],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 0, 0],
            [3, 3, 3, 3, 3],
        ],
        dtype=float,
    )

    normalized = gpoly(patterns)
    fit = u3poly(patterns)

    assert normalized[:6].tolist() == [0.0] * 6
    assert fit[:6].tolist() == [0.0] * 6
    assert normalized[6] == 1.0
    assert fit[6] == 1.0
    assert np.isnan(normalized[7:]).all()
    assert np.isnan(fit[7:]).all()
    assert gpoly(patterns, normalize=False)[7:].tolist() == [0.0, 0.0]


def test_incomplete_respondents_follow_the_documented_missing_policy() -> None:
    x = np.array(
        [[2, 2, 1, 0], [2, 1, np.nan, 0], [np.nan] * 4, [1, 1, 0, 0], [0, 0, 1, 2]], dtype=float
    )

    raw = gpoly(x, normalize=False)
    assert raw.tolist()[:2] == [0.0, 0.0]
    assert np.isnan(raw[2])
    np.testing.assert_array_equal(raw, _gpoly_oracle(x, 2, normalize=False))
    assert np.isnan(gpoly(x)[1:3]).all()
    assert np.isnan(u3poly(x)[1:3]).all()
    binary = np.where(np.isnan(x), np.nan, x > 0)
    assert np.isnan(ht(binary)[1:3]).all()
    # Incomplete respondents are left out of everyone else's Ht comparisons.
    np.testing.assert_array_equal(ht(binary)[[0, 3, 4]], ht(binary[[0, 3, 4]]))

    for scorer in (gpoly, u3poly, ht):
        with pytest.raises(ValueError, match="data contains missing values"):
            scorer(binary, na_rm=False)
    with pytest.raises(ValueError, match="data contains missing values"):
        gpoly(np.full((2, 3), np.nan), na_rm=False)


def test_unobserved_items_and_respondents_are_unavailable() -> None:
    x = np.array([[1, 2, np.nan], [0, 1, np.nan], [2, 0, np.nan]])

    assert np.isnan(gpoly(x)).all()
    assert np.isnan(u3poly(x)).all()
    np.testing.assert_array_equal(gpoly(x, normalize=False), _gpoly_oracle(x, 2, normalize=False))
    assert np.isnan(gpoly(np.full((2, 3), np.nan))).all()
    assert np.isnan(u3poly(np.full((2, 3), np.nan), ncat=4)).all()
    assert np.isnan(gpoly(np.full((2, 3), np.nan), scale_min=0, scale_max=2, normalize=False)).all()


def test_single_category_and_balanced_steps_have_no_spread() -> None:
    constant = np.full((4, 3), 2.0)
    assert gpoly(constant, normalize=False).tolist() == [0.0] * 4
    assert np.isnan(gpoly(constant)).all()
    assert np.isnan(u3poly(constant)).all()
    # Every step has popularity 0.5, so every pattern has zero weight.
    balanced = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    assert np.isnan(u3poly(balanced)).all()


@pytest.mark.parametrize(
    "convert",
    [
        lambda x: x.astype(np.int8),
        lambda x: x.astype(np.uint8),
        lambda x: x.astype(np.float32),
        lambda x: (x - 7).astype(np.int64),
        lambda x: x.astype(np.uint64) + np.uint64(2**64 - 8),
        lambda x: x.astype(np.int64) + np.iinfo(np.int64).min,
        lambda x: x.astype(object),
    ],
    ids=["int8", "uint8", "float32", "negative", "uint64-max", "int64-min", "object"],
)
def test_integer_dtypes_and_offsets_give_identical_scores(
    convert: Callable[[np.ndarray], np.ndarray],
) -> None:
    rng = np.random.default_rng(12)
    x = rng.integers(0, 5, size=(30, 6)).astype(float)
    x[0, :2] = [0, 4]
    converted = convert(x)

    for normalize in (True, False):
        np.testing.assert_array_equal(
            gpoly(converted, normalize=normalize), gpoly(x, normalize=normalize)
        )
    np.testing.assert_array_equal(u3poly(converted), u3poly(x))


def test_boolean_and_integer_responses_score_ht_like_floats() -> None:
    x = np.random.default_rng(4).random((25, 6)) < 0.6

    expected = ht(x.astype(float))
    np.testing.assert_array_equal(ht(x), expected)
    np.testing.assert_array_equal(ht(x.astype(np.uint64)), expected)
    np.testing.assert_array_equal(gpoly(x), gpoly(x.astype(float)))


def test_masked_dichotomous_responses_score_ht_like_nan_coded_responses() -> None:
    rng = np.random.default_rng(6)
    raw = rng.integers(0, 2, size=(30, 8))
    raw[rng.random(raw.shape) < 0.05] = -9
    masked = np.ma.masked_equal(raw, -9)

    np.testing.assert_array_equal(ht(masked), ht(np.where(raw == -9, np.nan, raw)))


def test_scale_bounds_and_category_counts_set_the_item_steps() -> None:
    x = np.array([[2, 3, 4], [3, 2, 2], [4, 4, 3], [2, 2, 3], [3, 4, 2]], dtype=float)
    # Category 1 exists but was never chosen: it adds an item step everyone passes.
    shifted = gpoly(x - 1, scale_min=0, scale_max=3)

    np.testing.assert_array_equal(gpoly(x, scale_min=1), shifted)
    np.testing.assert_array_equal(gpoly(x, ncat=4, scale_max=4), shifted)
    np.testing.assert_array_equal(gpoly(x, ncat=4, scale_min=1, scale_max=4), shifted)
    np.testing.assert_array_equal(gpoly(x, ncat=np.int64(4), scale_min=1.0), shifted)
    np.testing.assert_array_equal(u3poly(x, scale_min=1), u3poly(x - 1, ncat=4, scale_min=0))
    assert not np.array_equal(gpoly(x), shifted, equal_nan=True)
    # Without bounds, ncat extends the observed minimum upward.
    np.testing.assert_array_equal(gpoly(x, ncat=4), gpoly(x, scale_min=2, scale_max=5))


@pytest.mark.parametrize(
    ("data", "keywords", "message"),
    [
        ([[1, 1.5], [3, 1]], {}, "responses must be integer categories from 1 to 3"),
        ([[1, 2.5], [2, 1]], {}, "responses must be finite integer categories"),
        ([[1, 2], [2, 7]], {"scale_max": 5}, "responses must be integer categories from 1 to 5"),
        ([[0, 2], [2, 1]], {"scale_min": 1}, "responses must be integer categories from 1 to 2"),
        ([[1, 9], [2, 1]], {"ncat": 3}, "responses must be integer categories from 1 to 3"),
        ([[1, 2], [2, 1]], {"scale_min": 0.5}, "scale_min must be a finite integer"),
        ([[1, 2], [2, 1]], {"scale_max": 2.5}, "scale_max must be a finite integer"),
        ([[1, np.inf], [2, 1]], {}, "responses must be finite integer categories"),
        ([[1, 2], [2, 1]], {"ncat": 1}, "ncat must be an integer of at least 2"),
        ([[1, 2], [2, 1]], {"ncat": 2.5}, "ncat must be an integer of at least 2"),
        ([[1, 2], [2, 1]], {"ncat": True}, "ncat must be an integer of at least 2"),
        (
            [[1, 2], [2, 1]],
            {"ncat": 3, "scale_min": 1, "scale_max": 2},
            r"ncat must equal scale_max - scale_min \+ 1",
        ),
        ([[0, 2000], [2, 1]], {}, "at most 1024 response categories \\(got 2001\\)"),
        ([[1, 2], [2, 1]], {"ncat": 1025}, "at most 1024 response categories \\(got 1025\\)"),
        ([[1], [2]], {}, "data must have at least 2 columns"),
        # ncat and one endpoint fix the scale, so no observed extreme is blamed.
        ([[0, 0], [0, 0]], {"ncat": 2, "scale_min": 1}, "integer categories from 1 to 2$"),
        ([[3, 4], [4, 3]], {"ncat": 2, "scale_max": 2}, "integer categories from 1 to 2$"),
        ([[0, 9], [0, 0]], {"ncat": 3, "scale_min": 1}, "integer categories from 1 to 3$"),
        ([[1, 2], [2, 1]], {"ncat": 3, "scale_min": 0.5}, "scale_min must be a finite integer"),
        ([[1, 2], [2, 1]], {"ncat": 3, "scale_max": np.inf}, "scale_max must be a finite integer"),
        ([[1, 2], [2, 1]], {"scale_min": np.nan}, "scale_min must be a finite integer"),
    ],
)
def test_invalid_categories_and_scales_raise(
    data: list[list[float]], keywords: dict[str, Any], message: str
) -> None:
    for scorer in (gpoly, u3poly):
        with pytest.raises(ValueError, match=message):
            scorer(data, **keywords)


@pytest.mark.parametrize("ncat", [3, 4])
def test_ncat_with_one_endpoint_does_not_read_the_observed_extremes(ncat: int) -> None:
    x = np.array([[2, 3, 2], [3, 2, 2], [2, 2, 3], [3, 3, 2]], dtype=float)
    expected = gpoly(x, scale_min=1, scale_max=ncat), u3poly(x, scale_min=1, scale_max=ncat)

    with patch.object(
        person_fit_module, "resolve_scale_bounds", wraps=person_fit_module.resolve_scale_bounds
    ) as resolve:
        np.testing.assert_array_equal(gpoly(x, ncat=ncat, scale_min=1), expected[0])
        np.testing.assert_array_equal(u3poly(x, ncat=ncat, scale_max=ncat), expected[1])

    assert resolve.call_count == 2
    for call in resolve.call_args_list:
        assert call.kwargs == {"scale_min": 1, "scale_max": ncat}


_EXTENDED = np.finfo(np.longdouble).nmant >= 60


@pytest.mark.skipif(not _EXTENDED, reason="np.longdouble has no extended precision here")
def test_extended_precision_fractions_are_not_rounded_to_categories() -> None:
    tiny = np.longdouble(2) ** -60
    whole = np.array([[1, 2, 3], [3, 2, 1], [2, 3, 1], [1, 1, 2]], dtype=np.longdouble)
    interior, low_end, high_end = whole.copy(), whole.copy(), whole.copy()
    interior[0, 1] += tiny
    low_end[0, 0] -= tiny
    high_end[0, 2] += tiny
    assert float(interior[0, 1]) == 2.0  # float64 would round the fraction away

    for scorer in (gpoly, u3poly):
        for data in (interior, low_end, high_end):
            with pytest.raises(ValueError, match="integer categories"):
                scorer(data)
            with pytest.raises(ValueError, match="integer categories from 0 to 4"):
                scorer(data, scale_min=0, scale_max=4)
        with pytest.raises(ValueError, match="scale_min must be a finite integer"):
            scorer(whole, scale_min=np.longdouble(1) - tiny)
        with pytest.raises(ValueError, match="scale_max must be a finite integer"):
            scorer(whole, ncat=3, scale_max=np.longdouble(3) + tiny)
    binary = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.longdouble)
    binary[0, 2] -= tiny
    with pytest.raises(ValueError, match="ht requires dichotomous responses"):
        ht(binary)


@pytest.mark.skipif(not _EXTENDED, reason="np.longdouble has no extended precision here")
def test_extended_precision_categories_are_exact_beyond_float64() -> None:
    rng = np.random.default_rng(16)
    x = rng.integers(0, 4, size=(40, 5)).astype(float)
    x[0, :2] = [0, 3]
    x[1, 2] = np.nan
    # 2**62 + k is exact in np.longdouble but rounds to a multiple of 1024 in float64.
    offset = np.longdouble(2) ** 62
    shifted = x.astype(np.longdouble) + offset

    for normalize in (True, False):
        np.testing.assert_array_equal(
            gpoly(shifted, normalize=normalize), gpoly(x, normalize=normalize)
        )
    np.testing.assert_array_equal(u3poly(shifted), u3poly(x))
    lower = 2**62
    np.testing.assert_array_equal(
        u3poly(shifted, ncat=5, scale_min=lower), u3poly(x, ncat=5, scale_min=0)
    )
    with pytest.raises(ValueError, match=f"integer categories from {lower + 1} to {lower + 3}"):
        gpoly(shifted, scale_min=lower + 1, scale_max=lower + 3)


def test_floating_responses_of_every_width_score_alike() -> None:
    rng = np.random.default_rng(17)
    x = rng.integers(0, 4, size=(30, 6)).astype(float)
    x[0, :2] = [0, 3]
    x[2, [1, 4]] = np.nan
    x[3] = np.nan

    raw = gpoly(x, normalize=False)
    for dtype in (np.float16, np.float32, np.longdouble):
        converted = x.astype(dtype)
        np.testing.assert_array_equal(gpoly(converted), gpoly(x), err_msg=str(dtype))
        np.testing.assert_array_equal(gpoly(converted, normalize=False), raw, err_msg=str(dtype))
        np.testing.assert_array_equal(u3poly(converted), u3poly(x), err_msg=str(dtype))
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 6):
            # One respondent per batch, so one batch has no response at all.
            np.testing.assert_array_equal(
                gpoly(converted, normalize=False), raw, err_msg=str(dtype)
            )


@pytest.mark.parametrize(
    "data",
    [[[0, 2], [1, 0]], [[0, 0.5], [1, 0]], [[-1, 0], [1, 1]], [[0, 1], [1, np.inf]]],
)
def test_ht_requires_dichotomous_responses(data: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="ht requires dichotomous responses coded 0 and 1"):
        ht(data)


def test_ht_switches_to_python_integers_before_int64_products_overflow() -> None:
    x = np.random.default_rng(9).random((30, 7)) < 0.5
    expected = ht(x)

    with patch.object(person_fit_module, "_HT_INT64_LIMIT", 1):
        np.testing.assert_array_equal(ht(x), expected)


def test_large_samples_match_small_batches_and_the_definitions() -> None:
    rng = np.random.default_rng(10)
    x = rng.integers(0, 4, size=(30_000, 12)).astype(float)
    x[rng.random(x.shape) < 0.01] = np.nan
    binary = np.where(np.isnan(x), np.nan, x >= 2)
    expected = {
        "raw": gpoly(x, normalize=False),
        "normalized": gpoly(x),
        "u3": u3poly(x),
        "ht": ht(binary),
    }

    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 97):
        np.testing.assert_array_equal(gpoly(x, normalize=False), expected["raw"])
        np.testing.assert_array_equal(gpoly(x), expected["normalized"])
        np.testing.assert_array_equal(u3poly(x), expected["u3"])
        np.testing.assert_array_equal(ht(binary), expected["ht"])

    order = _step_order(x, 3)
    for row in rng.choice(len(x), size=40, replace=False):
        if not np.isnan(x[row]).all():
            assert expected["raw"][row] == _step_pair_errors(x[row], order)
    complete = ~np.isnan(binary).any(axis=1)
    sample = binary[complete].astype(np.int64)
    totals = sample.sum(axis=1)
    for row in rng.choice(len(sample), size=20, replace=False):
        product = totals[row] * totals
        covariance = 12 * (sample @ sample[row]) - product
        maximum = 12 * np.minimum(totals[row], totals) - product
        covariance[row] = maximum[row] = 0
        if maximum.sum() > 0:
            ratio = float(Fraction(int(covariance.sum()), int(maximum.sum())))
            assert expected["ht"][np.flatnonzero(complete)[row]] == ratio


def _auc(scores: np.ndarray, careless: np.ndarray) -> float:
    """Probability that a careless respondent scores above an attentive one."""
    valid = ~np.isnan(scores)
    positive = scores[valid & careless][:, np.newaxis]
    negative = scores[valid & ~careless][np.newaxis, :]
    return float(np.mean((positive > negative) + 0.5 * (positive == negative)))


def test_person_fit_detects_random_responders_in_simulated_surveys() -> None:
    rng = np.random.default_rng(0)
    n_attentive, n_careless = 1000, 100
    careless = np.r_[np.zeros(n_attentive, dtype=bool), np.ones(n_careless, dtype=bool)]
    # Graded response model: P(X >= k) = logistic(a * (theta - b_k)), 5 categories.
    theta = rng.normal(size=(n_attentive, 1, 1))
    discrimination = rng.uniform(1.0, 2.5, size=(30, 1))
    thresholds = np.sort(rng.normal(size=(30, 4)), axis=1)
    passing = 1 / (1 + np.exp(-discrimination * (theta - thresholds)))
    graded = np.sum(rng.random((n_attentive, 30, 1)) < passing, axis=2)
    polytomous = np.vstack([graded, rng.integers(0, 5, size=(n_careless, 30))])
    # Two-parameter logistic model for 20 dichotomous items.
    ability = rng.normal(size=(n_attentive, 1))
    correct = 1 / (1 + np.exp(-rng.uniform(1.0, 2.5, size=20) * (ability - rng.normal(size=20))))
    attentive_binary = rng.random(correct.shape) < correct
    dichotomous = np.vstack([attentive_binary, rng.integers(0, 2, size=(n_careless, 20))])

    assert _auc(gpoly(polytomous), careless) >= 0.9
    assert _auc(gpoly(polytomous, normalize=False), careless) >= 0.9
    assert _auc(u3poly(polytomous), careless) >= 0.9
    assert _auc(-ht(dichotomous), careless) >= 0.9


def test_registry_entries_and_catalog_metadata() -> None:
    catalog = index_catalog()
    expected = {"gpoly": "high", "u3poly_fit": "high", "ht": "low"}

    for name, direction in expected.items():
        metadata = {key: catalog[name][key] for key in catalog[name] if key != "flag_direction"}
        assert catalog[name]["flag_direction"] == direction
        assert {
            "flag_mode": "percentile",
            "default_screen": False,
            "default_composite": False,
            "composite_enabled": True,
            "required_options": (),
        }.items() <= metadata.items()
    assert INDEX_REGISTRY["ht"].composite_multiplier == -1.0
    assert INDEX_REGISTRY["gpoly"].composite_multiplier == 1.0
    assert IndexOptions().person_fit_ncat is None


def test_screen_and_composites_pass_person_fit_options() -> None:
    rng = np.random.default_rng(14)
    x = rng.integers(2, 6, size=(80, 10)).astype(float)
    binary = (x >= 4).astype(float)
    options = IndexOptions(person_fit_ncat=5, scale_min=1)

    result = screen(x, indices=["gpoly", "u3poly_fit"], options=options)
    np.testing.assert_array_equal(result["scores"]["gpoly"], gpoly(x, ncat=5, scale_min=1))
    np.testing.assert_array_equal(result["scores"]["u3poly_fit"], u3poly(x, ncat=5, scale_min=1))
    assert not np.array_equal(result["scores"]["gpoly"], gpoly(x))
    np.testing.assert_array_equal(result["flags"]["gpoly"], gpoly_flag(x, ncat=5, scale_min=1)[1])
    fit = screen(binary, indices=["ht"])
    np.testing.assert_array_equal(fit["scores"]["ht"], ht(binary))
    np.testing.assert_array_equal(fit["flags"]["ht"], ht_flag(binary)[1])

    details = composite_summary(binary, indices=["ht", "gpoly"], standardize=False)
    np.testing.assert_array_equal(details["indices"]["ht"], ht(binary))
    np.testing.assert_array_equal(details["indices"]["gpoly"], gpoly(binary))
    # Ht is low-direction, so composites reverse it before combining.
    reversed_ht = composite_summary(binary, indices=["ht"], standardize=False)["composite"]
    np.testing.assert_array_equal(reversed_ht, -ht(binary))


def test_registry_reports_person_fit_failures_softly() -> None:
    x = np.array([[1, 2, 3], [3, 2, np.nan], [2, 2, 1], [1, 3, 2]])

    result = screen(x, indices=["ht", "gpoly", "irv"], options=IndexOptions(na_rm=False))

    assert result["errors"] == {
        "ht": "ht requires dichotomous responses coded 0 and 1",
        "gpoly": "data contains missing values. Set na_rm=True to handle them",
    }
    assert result["indices_used"] == ["irv"]
    with pytest.raises(ValueError, match="index 'gpoly' failed"):
        screen(x, indices=["gpoly"], options=IndexOptions(person_fit_ncat=2), strict=True)


_HELPERS = [
    ("gpoly", gpoly_flag, gpoly),
    ("u3poly_fit", u3poly_flag, u3poly),
    ("ht", ht_flag, ht),
]


def _flag_data(name: str) -> np.ndarray:
    rng = np.random.default_rng(15)
    x = rng.integers(0, 4, size=(200, 8)).astype(float)
    x[:150] = np.sort(x[:150], axis=1)[:, ::-1]
    return (x >= 2).astype(float) if name == "ht" else x


@pytest.mark.parametrize(
    ("name", "helper", "scorer"), _HELPERS, ids=[name for name, *_ in _HELPERS]
)
def test_flag_helpers_follow_registry_directions_and_screen(
    name: str,
    helper: Callable[..., tuple[np.ndarray, np.ndarray]],
    scorer: Callable[..., np.ndarray],
) -> None:
    data = _flag_data(name)
    with patch("ier.person_fit.threshold_flags", wraps=threshold_flags) as flagged:
        scores, flags = helper(data)
    assert flagged.call_args.kwargs["direction"] == INDEX_REGISTRY[name].flag_direction
    result = screen(data, indices=[name])
    np.testing.assert_array_equal(scores, result["scores"][name])
    np.testing.assert_array_equal(flags, result["flags"][name])
    assert 0 < np.count_nonzero(flags) < len(data)

    fixed_scores, fixed = helper(data, threshold=0.25)
    expected = fixed_scores <= 0.25 if name == "ht" else fixed_scores >= 0.25
    np.testing.assert_array_equal(fixed, expected & ~np.isnan(fixed_scores))
    np.testing.assert_array_equal(fixed_scores, scorer(data))


def test_flag_helpers_pass_scorer_options_through() -> None:
    x = _flag_data("gpoly")
    x[0, 0] = np.nan

    raw, flags = gpoly_flag(x, percentile=90, ncat=5, scale_min=0, normalize=False, na_rm=True)
    np.testing.assert_array_equal(raw, gpoly(x, ncat=5, scale_min=0, normalize=False))
    assert flags.dtype == bool
    fit, _ = u3poly_flag(x, scale_min=0, scale_max=4)
    np.testing.assert_array_equal(fit, u3poly(x, scale_min=0, scale_max=4))
    with pytest.raises(ValueError, match="data contains missing values"):
        ht_flag(np.where(np.isnan(x), np.nan, x > 1), na_rm=False)


def _cli_scores(arguments: list[str]) -> dict[str, list[float | None]]:
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main([*arguments, "--format", "json"])
    assert code == 0
    scores: dict[str, list[float | None]] = json.loads(stdout.getvalue())["scores"]
    return scores


def _nan_as_none(values: np.ndarray) -> list[float | None]:
    return [None if math.isnan(value) else value for value in values.tolist()]


def test_cli_scores_person_fit_indices_with_category_counts(tmp_path: Path) -> None:
    x = np.array([[2, 3, 4, 3], [3, 2, 2, 4], [4, 4, 3, 3], [2, 2, 3, 2], [3, 4, 2, 2]])
    source = tmp_path / "responses.csv"
    source.write_text(
        "a,b,c,d\n" + "".join(",".join(map(str, row)) + "\n" for row in x), encoding="utf-8"
    )
    binary = tmp_path / "binary.csv"
    binary.write_text(
        "a,b,c,d\n" + "".join(",".join(str(int(v >= 3)) for v in row) + "\n" for row in x),
        encoding="utf-8",
    )

    scores = _cli_scores(
        ["screen", str(source), "--indices", "gpoly", "u3poly_fit"]
        + ["--person-fit-ncat", "5", "--scale-min", "1"]
    )
    assert scores["gpoly"] == _nan_as_none(gpoly(x, ncat=5, scale_min=1))
    assert scores["u3poly_fit"] == _nan_as_none(u3poly(x, ncat=5, scale_min=1))
    config = tmp_path / "ier.toml"
    config.write_text("[composite]\nperson_fit_ncat = 5\nscale_min = 1\n", encoding="utf-8")
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        code = main(
            ["composite", str(source), "--indices", "gpoly", "--no-standardize"]
            + ["--config", str(config), "--format", "json"]
        )
    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["scores"] == _nan_as_none(gpoly(x, ncat=5, scale_min=1))
    assert _cli_scores(["screen", str(binary), "--indices", "ht"])["ht"] == _nan_as_none(ht(x >= 3))


@pytest.mark.parametrize("value", ["1", "2.5", "none"])
def test_cli_rejects_invalid_category_counts(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["screen", "responses.csv", "--person-fit-ncat", value])
    assert error.value.code == 2
    assert "must be an integer of at least 2" in capsys.readouterr().err


def _person_fit_docs() -> dict[str, str]:
    """The person-fit passages of the user docs, with whitespace collapsed."""
    indices = (_DOCS / "indices.md").read_text(encoding="utf-8")
    comparison = (_DOCS / "r-comparison.md").read_text(encoding="utf-8")
    passages = {
        "indices.md": indices[indices.index("`gpoly`, `u3poly_fit`, and `ht` (all opt-in)") :],
        "r-comparison.md": comparison[comparison.index("- item-step person fit") :],
    }
    passages["indices.md"] = passages["indices.md"][: passages["indices.md"].index("\n## ")]
    passages["r-comparison.md"] = passages["r-comparison.md"][
        : passages["r-comparison.md"].index("\n- ", 1)
    ]
    return {name: " ".join(text.split()) for name, text in passages.items()}


def test_docs_state_that_perfit_scores_perfect_vectors_zero() -> None:
    polytomous = np.array([[0, 0, 0], [2, 2, 2], [2, 1, 0], [1, 2, 0], [0, 1, 2]])
    dichotomous = (polytomous >= 1).astype(int)

    for scores in (gpoly(polytomous), u3poly(polytomous), gpoly(dichotomous)):
        assert np.isnan(scores[:2]).all()
    for name, text in _person_fit_docs().items():
        assert "`Gnormed.poly` and `U3poly`" in text, name
    bullet = _person_fit_docs()["r-comparison.md"]
    assert "Normalized `gpoly` and `u3poly` return `NaN` for perfect response vectors" in bullet
    assert "and `Gnormed` for dichotomous items, return 0" in bullet


def test_docs_describe_perfits_default_missing_data_method() -> None:
    x = np.array([[2, 2, 1, 0], [2, 1, 1, 0], [1, 1, 0, 0], [2, 0, 2, 0], [0, np.nan, 1, 2]])

    assert np.isnan(gpoly(x)[-1])
    assert np.isnan(u3poly(x)[-1])
    for name, text in _person_fit_docs().items():
        assert 'PerFit\'s default `NA.method = "Pairwise"`' in text, name
        assert "scores incomplete respondents from their observed items" in text, name
        assert "only its `Hotdeck`, `NPModel`, and `PModel` methods impute" in text, name
        assert "PerFit instead imputes" not in text, name
        assert "while PerFit imputes" not in text, name


def test_person_fit_functions_are_exported() -> None:
    names = ["gpoly", "gpoly_flag", "u3poly", "u3poly_flag", "ht", "ht_flag"]

    assert set(names) <= set(ier.__all__)
    assert all(getattr(ier, name) is getattr(person_fit_module, name) for name in names)
