"""Lagged autocorrelation matches per-lag Pearson correlations and joins the workflows."""

import json
import warnings
from fractions import Fraction
from io import StringIO
from math import isqrt
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

import ier
from ier import (
    IndexOptions,
    autocorrelation,
    autocorrelation_flag,
    composite_summary,
    index_catalog,
    screen,
)
from ier._flagging import threshold_flags
from ier._registry import INDEX_REGISTRY
from ier.autocorrelation import AutocorrelationStatistic
from ier.cli import _build_parser, _options_from_args, main


def _pearson_scores(
    x: Any,
    max_lag: int | None = 10,
    min_lag: int = 1,
    statistic: AutocorrelationStatistic = "max_abs",
    na_rm: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate every usable lag one respondent at a time with np.corrcoef.

    A lag whose window does not vary scores 1, as in rp.acors, and lags within
    rounding of the largest magnitude tie, with the smallest winning.
    """
    scores: list[float] = []
    lags: list[float] = []
    for raw in np.asarray(x, dtype=float):
        missing = np.isnan(raw)
        row = raw[~missing]
        n = len(row)
        top = n - 3 if max_lag is None else min(max_lag, n - 3)
        magnitudes: list[float] = []
        positions: list[int] = []
        if not ((missing.any() and not na_rm) or top < min_lag):
            for lag in range(min_lag, top + 1):
                first, second = row[: n - lag], row[lag:]
                if np.ptp(first) == 0 or np.ptp(second) == 0:
                    magnitudes.append(1.0)
                else:
                    magnitudes.append(abs(float(np.corrcoef(first, second)[0, 1])))
                positions.append(lag)
        if not magnitudes:
            scores.append(np.nan)
            lags.append(np.nan)
            continue
        best = max(magnitudes)
        scores.append(best if statistic == "max_abs" else sum(magnitudes))
        tied = [
            lag for lag, value in zip(positions, magnitudes, strict=True) if value >= best - 1e-12
        ]
        lags.append(float(tied[0]))
    return np.array(scores), np.array(lags)


def _rp_acors(row: list[float], max_lag: int | None = None) -> tuple[float, float, float]:
    """Port rp.acors (responsePatterns R/rp_acors.R) for one complete row.

    Returns the maximum and summed absolute lag correlations and R's
    ``which(acors == max(acors))[1]`` lag. A lag scores 1 when either window
    has zero variance.
    """
    n = len(row)
    top = n - 3 if max_lag is None else min(max_lag, n - 3)
    magnitudes = []
    for lag in range(1, top + 1):
        first = np.asarray(row[: n - lag], dtype=float)
        second = np.asarray(row[lag:], dtype=float)
        if np.var(first) == 0 or np.var(second) == 0:
            magnitudes.append(1.0)
        else:
            magnitudes.append(abs(float(np.corrcoef(first, second)[0, 1])))
    best = max(magnitudes)
    return best, sum(magnitudes), float(magnitudes.index(best) + 1)


def _exact_correlation(first: list[float], second: list[float]) -> float:
    """Pearson correlation in rational arithmetic, rounded once."""
    a = [Fraction(value) for value in first]
    b = [Fraction(value) for value in second]
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    product = sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
    scale = 10**60
    return float(covariance / Fraction(isqrt(int(product * scale * scale)), scale))


def _responses(seed: int = 20261005, shape: tuple[int, int] = (400, 30)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 6, size=shape).astype(float)
    data[rng.random(shape) < 0.15] = np.nan
    data[0] = 3.0
    data[1, :] = np.nan
    data[2, :-4] = np.nan
    data[3] = 2.0
    data[3, -1] = 5.0
    data[4, ::2] = np.nan
    data[4, 1::2] = 0.1
    return data


@pytest.mark.parametrize("statistic", ["max_abs", "sum_abs"])
@pytest.mark.parametrize(("min_lag", "max_lag"), [(1, 10), (1, None), (2, 5), (3, 3), (1, 50)])
@pytest.mark.parametrize("na_rm", [True, False])
def test_scores_and_lags_match_per_lag_pearson_correlations(
    statistic: AutocorrelationStatistic, min_lag: int, max_lag: int | None, na_rm: bool
) -> None:
    data = _responses()
    scores, lags = autocorrelation(
        data,
        max_lag=max_lag,
        min_lag=min_lag,
        statistic=statistic,
        na_rm=na_rm,
        return_lags=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected, expected_lags = _pearson_scores(data, max_lag, min_lag, statistic, na_rm)

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(lags, expected_lags)
    assert np.isfinite(scores).any()


def test_default_call_returns_only_scores() -> None:
    data = _responses()
    scores = autocorrelation(data)
    assert isinstance(scores, np.ndarray)
    np.testing.assert_array_equal(scores, autocorrelation(data, return_lags=True)[0])


def test_each_respondent_uses_only_lags_with_three_pairs() -> None:
    data = np.full((4, 12), np.nan)
    data[0, :6] = [1, 2, 3, 1, 2, 4]
    data[1, :4] = [1, 2, 4, 3]
    data[2, :3] = [1, 2, 3]
    data[3] = [1, 3, 2, 5, 4, 1, 3, 2, 5, 4, 1, 3]

    scores, lags = autocorrelation(data, statistic="sum_abs", return_lags=True)
    expected, expected_lags = _pearson_scores(data, statistic="sum_abs")

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(lags, expected_lags)
    # Six responses support lags 1-3; four support only lag 1; three support none.
    first = data[0, :6]
    manual = sum(abs(np.corrcoef(first[: 6 - k], first[k:])[0, 1]) for k in (1, 2, 3))
    assert scores[0] == pytest.approx(manual, abs=1e-12)
    assert scores[1] == pytest.approx(abs(np.corrcoef([1, 2, 4], [2, 4, 3])[0, 1]), abs=1e-15)
    assert lags[1] == 1.0
    assert np.isnan(scores[2]) and np.isnan(lags[2])
    assert lags[3] == 5.0

    shifted = autocorrelation(data, min_lag=2)
    assert np.isnan(shifted[1])
    from_lag_two = max(abs(np.corrcoef(first[: 6 - k], first[k:])[0, 1]) for k in (2, 3))
    assert shifted[0] == pytest.approx(from_lag_two, abs=1e-12)


def test_identical_responses_correlate_perfectly_at_every_lag() -> None:
    data = np.array(
        [
            [4.0] * 12,
            [0.1] * 12,
            [np.nan, 2.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, np.nan, 2.0, 2.0, 2.0],
            [7.0] * 3 + [np.nan] * 9,
        ]
    )
    scores, lags = autocorrelation(data, return_lags=True)
    np.testing.assert_array_equal(scores[:3], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(lags[:3], [1.0, 1.0, 1.0])
    assert np.isnan(scores[3])

    # Twelve responses support lags 1-9; nine observed responses support 1-6.
    sums = autocorrelation(data, statistic="sum_abs", max_lag=None)
    np.testing.assert_array_equal(sums[:3], [9.0, 9.0, 6.0])
    np.testing.assert_array_equal(autocorrelation(data, min_lag=4, return_lags=True)[1][:3], 4.0)
    assert autocorrelation(np.full((1, 6), 5, dtype=np.int64)).tolist() == [1.0]


def test_lags_with_a_constant_window_score_one() -> None:
    nearly_constant = [2.0] * 11 + [5.0]
    data = np.array([nearly_constant, [2.0] * 6 + [5.0] + [2.0] * 5])
    scores, lags = autocorrelation(data, return_lags=True)
    sums = autocorrelation(data, max_lag=None, statistic="sum_abs")
    # Every first window of the first row repeats one value, so every lag scores 1.
    assert scores[0] == 1.0 and lags[0] == 1.0 and sums[0] == 9.0
    expected, expected_lags = _pearson_scores(data)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(lags, expected_lags)
    # In the second row, windows are constant from lag 6 (r[0:6]) on, so lags 6-9 score 1.
    per_lag = [autocorrelation(data[1:], min_lag=lag, max_lag=lag)[0] for lag in range(1, 10)]
    assert per_lag[5:] == [1.0] * 4 and max(per_lag[:5]) < 1.0
    assert sums[1] == pytest.approx(sum(per_lag), abs=1e-12)


@pytest.mark.parametrize(
    "row",
    [
        [3.0] * 39 + [5.0],
        [1.0] + [3.0] * 39,
        [4.0, 2.0, 5.0, 1.0, 3.0] + [3.0] * 35,
        [2.0, 4.0] + [3.0] * 30 + [5.0, 1.0, 2.0, 4.0, 1.0, 5.0, 2.0, 3.0],
        [3.0] * 40,
        [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0] * 5,
        np.random.default_rng(8).integers(1, 6, size=40).astype(float).tolist(),
    ],
    ids=[
        "last-differs",
        "first-differs",
        "varied-then-flat",
        "flat-middle",
        "flat",
        "zigzag",
        "random",
    ],
)
def test_constant_windows_follow_rp_acors(row: list[float]) -> None:
    expected_max, expected_sum, expected_lag = _rp_acors(row)
    scores, lags = autocorrelation([row], max_lag=None, return_lags=True)
    sums = autocorrelation([row], max_lag=None, statistic="sum_abs")

    assert scores[0] == pytest.approx(expected_max, abs=1e-12)
    assert sums[0] == pytest.approx(expected_sum, abs=1e-11)
    assert lags[0] == expected_lag


def test_near_straight_liners_are_flagged_with_straight_liners() -> None:
    attentive = np.random.default_rng(6).integers(1, 6, size=(20, 40)).astype(float)
    near = np.full((3, 40), 3.0)
    near[0, -1] = 5.0
    near[1, 0] = 1.0
    near[2, :5] = [4.0, 2.0, 5.0, 1.0, 3.0]
    data = np.vstack([attentive, near, np.full((1, 40), 3.0)])
    data[-4, ::7] = np.nan

    scores, flags = autocorrelation_flag(data, threshold=0.9)
    sums = autocorrelation(data, max_lag=None, statistic="sum_abs")

    np.testing.assert_array_equal(scores[-4:], [1.0, 1.0, 1.0, 1.0])
    assert flags[-4:].all() and not flags[:-4].any()
    # Thirty-four observed responses support lags 1-31; the others support 1-37.
    np.testing.assert_array_equal(sums[[-4, -3, -1]], [31.0, 37.0, 37.0])
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 40 * 3):
        np.testing.assert_array_equal(autocorrelation(data), scores)
        batched = autocorrelation(data, max_lag=None, statistic="sum_abs")
    np.testing.assert_array_equal(batched, sums)


@pytest.mark.parametrize("na_rm", [True, False])
def test_constant_windows_in_compacted_batches_match_the_oracle(na_rm: bool) -> None:
    rng = np.random.default_rng(14)
    data = np.full((80, 18), 3.0)
    for row in data:
        opening, closing = rng.integers(0, 4, size=2)
        row[:opening] = rng.integers(1, 6, size=opening)
        row[len(row) - closing :] = rng.integers(1, 6, size=closing)
    data[rng.random(data.shape) < 0.1] = np.nan
    data[::4, 12:] = np.nan
    expected, expected_lags = _pearson_scores(data, None, 1, "sum_abs", na_rm)

    # Four-row batches mostly lack a complete row, so most are compacted below 18 columns.
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 18 * 4):
        scores, lags = autocorrelation(
            data, max_lag=None, statistic="sum_abs", na_rm=na_rm, return_lags=True
        )

    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(lags, expected_lags)
    assert np.isfinite(scores).sum() >= 10


def _exact_best_lags(row: list[float], max_lag: int) -> tuple[Fraction, int]:
    """Return the largest exact squared lag correlation and the smallest lag within
    a relative 1e-12 of it (decimal inputs such as 29.2 tie only up to rounding).
    """
    n = len(row)
    squares: list[Fraction] = []
    for lag in range(1, min(max_lag, n - 3) + 1):
        a = [Fraction(value) for value in row[: n - lag]]
        b = [Fraction(value) for value in row[lag:]]
        mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
        covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
        variances = sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)
        squares.append(covariance**2 / variances if variances else Fraction(1))
    best = max(squares)
    tied = best * (1 - Fraction(1, 10**12))
    return best, next(lag for lag, value in enumerate(squares, 1) if value >= tied)


@pytest.mark.parametrize(
    ("unit", "length", "expected_lag"),
    [
        ([1, 1, 2], 10, 3),
        ([1, 2, 3, 4, 5, 4, 3, 2], 12, 4),
        ([1, 2, 3, 4, 5, 4, 3, 2], 30, 4),
        ([1, 3, 5, 3], 30, 2),
        ([1, 5], 20, 1),
        ([2, 3, 4], 30, 3),
    ],
)
def test_perfect_cycles_score_one_at_the_smallest_tied_lag(
    unit: list[int], length: int, expected_lag: int
) -> None:
    row = np.resize(np.asarray(unit, dtype=float), length)
    best, smallest = _exact_best_lags(row.tolist(), 10)
    assert best == 1 and smallest == expected_lag

    scores, lags = autocorrelation(row[None, :], return_lags=True)
    assert scores[0] == 1.0 and lags[0] == expected_lag
    _, flags = autocorrelation_flag(row[None, :], threshold=1.0, max_lag=5)
    assert flags[0]


def test_lag_ties_go_to_the_smallest_lag_despite_rounding() -> None:
    rng = np.random.default_rng(21)
    rows = []
    for index in range(300):
        period = int(rng.integers(2, 7))
        if index % 3 == 0:
            unit = rng.integers(1, 6, size=period).astype(float)
        elif index % 3 == 1:
            unit = rng.integers(1, 8, size=period) * 3.7 + 1e6
        else:
            unit = rng.normal(size=period)
        row = np.roll(np.resize(unit, int(rng.integers(10, 31))), int(rng.integers(0, period)))
        if np.ptp(unit) > 0:
            rows.append(row)

    for row in rows:
        best, smallest = _exact_best_lags(row.tolist(), 10)
        scores, lags = autocorrelation(row[None, :], return_lags=True)
        assert scores[0] == pytest.approx(float(best) ** 0.5, abs=1e-13)
        if best == 1:
            assert scores[0] == 1.0
        assert lags[0] == smallest, row.tolist()


def test_missing_responses_are_removed_or_propagated() -> None:
    data = _responses()
    complete = data[np.isfinite(data).all(axis=1)]
    removed = autocorrelation(data)
    propagated = autocorrelation(data, na_rm=False)

    incomplete = np.isnan(data).any(axis=1)
    assert np.isnan(propagated[incomplete]).all()
    np.testing.assert_array_equal(propagated[~incomplete], removed[~incomplete])
    np.testing.assert_array_equal(propagated[~incomplete], autocorrelation(complete))
    packed = data[2][~np.isnan(data[2])][None, :]
    np.testing.assert_array_equal(removed[2:3], autocorrelation(packed))


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: x + 1e9,
        lambda x: x + 1e15,
        lambda x: x * 1e-300,
        lambda x: x * 5e-324,
        lambda x: x * 1e300,
        lambda x: (x - 3.0) * 5e307,
        lambda x: 7.5 - x,
    ],
    ids=[
        "offset-1e9",
        "offset-1e15",
        "scale-1e-300",
        "subnormal",
        "scale-1e300",
        "near-max",
        "reversed",
    ],
)
@pytest.mark.parametrize("statistic", ["max_abs", "sum_abs"])
def test_location_and_scale_do_not_change_scores(
    transform: Any, statistic: AutocorrelationStatistic
) -> None:
    data = _responses(shape=(200, 24))
    expected = autocorrelation(data, statistic=statistic, max_lag=None)
    transformed = transform(data)

    actual = autocorrelation(transformed, statistic=statistic, max_lag=None)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


def test_large_integers_keep_adjacent_categories() -> None:
    data = np.random.default_rng(3).integers(1, 6, size=(50, 15))
    expected = autocorrelation(data)

    for offset, dtype in [(2**62, np.int64), (2**63 + 2**62, np.uint64)]:
        shifted = data.astype(dtype) + dtype(offset)
        np.testing.assert_allclose(autocorrelation(shifted), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [1.0, 2.0**1000], ids=["unscaled", "huge"])
def test_nearly_constant_windows_are_recomputed_from_their_own_responses(scale: float) -> None:
    rng = np.random.default_rng(11)
    data = rng.normal(scale=1e-6, size=(12, 30))
    data[:, -1] = 1e3
    data[::2, 0] = -1e3
    data[1] = 1e6 + rng.integers(0, 3, size=30) * 1e-6
    data[1, 5] += 1.0

    # Power-of-two scaling is exact, so huge rows keep the same correlations.
    scores = autocorrelation(data * scale, max_lag=6)

    for row, score in zip(data, scores, strict=True):
        values = row.tolist()
        expected = max(
            abs(_exact_correlation(values[: len(values) - lag], values[lag:]))
            for lag in range(1, 7)
        )
        assert score == pytest.approx(expected, abs=1e-14)


def test_dtype_and_memory_layout_do_not_change_scores() -> None:
    data = np.random.default_rng(5).integers(1, 8, size=(120, 20))
    data_float = data.astype(float)
    expected = autocorrelation(data_float, statistic="sum_abs")

    for integers in (data, data.astype(np.int32), data.astype(np.uint8)):
        np.testing.assert_array_equal(autocorrelation(integers, statistic="sum_abs"), expected)
    np.testing.assert_array_equal(
        autocorrelation(np.asfortranarray(data_float), statistic="sum_abs"), expected
    )
    np.testing.assert_array_equal(
        autocorrelation(data_float.astype(np.float32), statistic="sum_abs"), expected
    )
    np.testing.assert_array_equal(
        autocorrelation(data_float.tolist(), statistic="sum_abs"), expected
    )
    missing = _responses(shape=(80, 20))
    np.testing.assert_array_equal(
        autocorrelation(np.asfortranarray(missing)), autocorrelation(missing)
    )


def test_batches_with_different_observed_widths_agree_with_one_batch() -> None:
    data = _responses(shape=(53, 16))
    data[10:20] = np.random.default_rng(9).integers(1, 6, size=(10, 16))
    data[30:40, 5:] = np.nan
    expected = autocorrelation(data, return_lags=True)
    expected_sums = autocorrelation(data, statistic="sum_abs", max_lag=None)

    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 16 * 7):
        batched = autocorrelation(data, return_lags=True)
        batched_sums = autocorrelation(data, statistic="sum_abs", max_lag=None)

    np.testing.assert_array_equal(batched[0], expected[0])
    np.testing.assert_array_equal(batched[1], expected[1])
    np.testing.assert_array_equal(batched_sums, expected_sums)

    # A batch whose respondents all answered too few items has no usable lag.
    sparse = data.copy()
    sparse[7:14, 3:] = np.nan
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 16 * 7):
        batched = autocorrelation(sparse, return_lags=True)
    assert np.isnan(batched[0][7:14]).all() and np.isnan(batched[1][7:14]).all()
    np.testing.assert_array_equal(batched[0], autocorrelation(sparse))


def test_infinite_responses_leave_only_their_rows_unavailable() -> None:
    data = _responses(shape=(10, 12))
    data[5, 3] = np.inf
    data[6, 0] = -np.inf
    data[7, :] = [np.inf] * 12

    scores, lags = autocorrelation(data, return_lags=True)

    assert np.isnan(scores[5:8]).all() and np.isnan(lags[5:8]).all()
    finite = np.ones(10, dtype=bool)
    finite[5:8] = False
    np.testing.assert_array_equal(scores[finite], autocorrelation(data[finite]))


def _simulated_survey(seed: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Factor-structured attentive responses and noisy zigzag, seesaw, and cycle patterns."""
    rng = np.random.default_rng(seed)
    traits = rng.normal(size=(1000, 5))
    keys = np.where(rng.random(40) < 0.3, -1.0, 1.0)
    latent = np.repeat(traits, 8, axis=1) * keys * 0.8
    attentive = np.clip(np.rint(3 + latent + rng.normal(size=latent.shape)), 1, 5)
    patterns = [[1, 2, 3, 4, 5, 4, 3, 2], [1, 5], [2, 3, 4]]
    careless = np.array(
        [np.roll(np.resize(patterns[i % 3], 40), rng.integers(0, 8)) for i in range(150)],
        dtype=float,
    )
    perturbed = rng.random(careless.shape) < noise
    careless[perturbed] = rng.integers(1, 6, size=int(perturbed.sum()))
    return attentive, careless


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    greater = np.mean(positive[:, None] > negative[None, :])
    ties = np.mean(positive[:, None] == negative[None, :])
    return float(greater + 0.5 * ties)


@pytest.mark.parametrize("statistic", ["max_abs", "sum_abs"])
def test_noisy_repetitive_patterns_are_detected(statistic: AutocorrelationStatistic) -> None:
    attentive, careless = _simulated_survey(20261005, noise=0.2)

    def score(data: np.ndarray, max_lag: int | None) -> np.ndarray:
        return autocorrelation(data, max_lag=max_lag, statistic=statistic)

    focused = _auc(score(careless, 10), score(attentive, 10))
    assert focused >= 0.9
    # Long lags pair few responses and dilute short cycles, hence the default of 10.
    assert focused > _auc(score(careless, None), score(attentive, None))


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"min_lag": 0}, "min_lag must be an integer of at least 1"),
        ({"min_lag": 1.5}, "min_lag must be an integer of at least 1"),
        ({"min_lag": True}, "min_lag must be an integer of at least 1"),
        ({"max_lag": 0}, "max_lag must be None or an integer of at least min_lag"),
        ({"max_lag": 2, "min_lag": 3}, "max_lag must be None or an integer of at least min_lag"),
        ({"max_lag": "10"}, "max_lag must be None or an integer of at least min_lag"),
        ({"statistic": "max"}, "statistic must be 'max_abs' or 'sum_abs'"),
        ({"statistic": None}, "statistic must be 'max_abs' or 'sum_abs'"),
        ({"return_lags": 1}, "return_lags must be a boolean"),
        ({"min_lag": 7}, "data must have at least 10 columns"),
    ],
)
def test_invalid_arguments_are_rejected(keywords: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        autocorrelation(np.ones((3, 9)), **keywords)


def test_integer_like_lags_are_accepted() -> None:
    data = _responses(shape=(20, 12))
    expected = autocorrelation(data, max_lag=4, min_lag=2)
    np.testing.assert_array_equal(
        autocorrelation(data, max_lag=np.int64(4), min_lag=np.uint8(2)), expected
    )


def test_flag_helper_uses_high_tail_and_skips_unavailable_scores() -> None:
    data = _responses()
    scores, flags = autocorrelation_flag(data, max_lag=6, statistic="sum_abs")
    np.testing.assert_array_equal(scores, autocorrelation(data, max_lag=6, statistic="sum_abs"))
    options = IndexOptions(autocorrelation_max_lag=6, autocorrelation_statistic="sum_abs")
    result = screen(data, indices=["autocorrelation"], options=options)
    np.testing.assert_array_equal(flags, result["flags"]["autocorrelation"])
    assert 0 < np.count_nonzero(flags) < len(data)

    scores, flags = autocorrelation_flag(data, threshold=1.0, na_rm=False, min_lag=2)
    np.testing.assert_array_equal(flags, scores >= 1.0)
    assert flags.any() and not flags[np.isnan(scores)].any()


def test_flag_helper_uses_the_registry_direction_and_passes_options() -> None:
    data = _responses()
    with patch("ier.autocorrelation.threshold_flags", wraps=threshold_flags) as flagged:
        scores, flags = autocorrelation_flag(
            data, percentile=10.0, max_lag=4, min_lag=2, statistic="sum_abs", na_rm=False
        )
    assert flagged.call_count == 1
    assert flagged.call_args.kwargs["direction"] == INDEX_REGISTRY["autocorrelation"].flag_direction
    expected = autocorrelation(data, max_lag=4, min_lag=2, statistic="sum_abs", na_rm=False)
    np.testing.assert_array_equal(scores, expected)
    assert flags.dtype == bool and flags.shape == scores.shape


def test_registry_screen_and_composites_use_index_options() -> None:
    data = _responses()
    entry = index_catalog()["autocorrelation"]
    assert entry == {
        "flag_direction": "high",
        "flag_mode": "percentile",
        "default_screen": False,
        "default_composite": False,
        "composite_enabled": True,
        "required_options": (),
        "alternative_options": (),
        "uses_keyed_responses": False,
    }
    assert INDEX_REGISTRY["autocorrelation"].composite_multiplier == 1.0

    default = screen(data, indices=["autocorrelation"])
    np.testing.assert_array_equal(default["scores"]["autocorrelation"], autocorrelation(data))

    # Complete straight-lining rows 0 and 3 would tie for the top score, and
    # percentile flags exclude ties, so row 3 is left incomplete here.
    data[3, 0] = np.nan
    options = IndexOptions(
        autocorrelation_max_lag=None, autocorrelation_statistic="sum_abs", na_rm=False
    )
    result = screen(data, indices=["autocorrelation"], options=options)
    expected = autocorrelation(data, max_lag=None, statistic="sum_abs", na_rm=False)
    np.testing.assert_array_equal(result["scores"]["autocorrelation"], expected)
    flagged = result["flags"]["autocorrelation"]
    assert flagged.any() and not flagged[np.isnan(expected)].any()

    details = composite_summary(data, indices=["autocorrelation", "irv"], standardize=False)
    np.testing.assert_array_equal(details["indices"]["autocorrelation"], autocorrelation(data))
    assert "autocorrelation" not in ier.screen(data)["scores"]


def test_cli_options_reach_the_index(tmp_path: Path) -> None:
    data = np.random.default_rng(2).integers(1, 6, size=(30, 14)).astype(float)
    source = tmp_path / "responses.csv"
    lines = [",".join(f"q{column}" for column in range(14))]
    lines.extend(",".join(str(int(value)) for value in row) for row in data)
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stdout = StringIO()
    arguments = ["screen", str(source), "--indices", "autocorrelation", "--format", "json"]
    arguments += ["--autocorrelation-max-lag", "none", "--autocorrelation-statistic", "sum_abs"]
    with patch("sys.stdout", stdout):
        assert main(arguments) == 0
    scores = json.loads(stdout.getvalue())["scores"]["autocorrelation"]
    expected = autocorrelation(data, max_lag=None, statistic="sum_abs")
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)

    args = _build_parser().parse_args(
        ["composite", "responses.csv", "--autocorrelation-max-lag", "4"]
    )
    assert _options_from_args(args) == IndexOptions(autocorrelation_max_lag=4)
    args = _build_parser().parse_args(["screen", "responses.csv"])
    assert args.autocorrelation_max_lag == 10
    assert args.autocorrelation_statistic == "max_abs"


@pytest.mark.parametrize("value", ["0", "2.5", "ten"])
def test_cli_rejects_invalid_maximum_lags(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["screen", "responses.csv", "--autocorrelation-max-lag", value])
    assert error.value.code == 2
    assert (
        "argument --autocorrelation-max-lag: must be 'none' or an integer of at least 1"
        in capsys.readouterr().err
    )
