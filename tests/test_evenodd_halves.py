"""Half-scale even-odd consistency matches careless::evenodd and keeps legacy pairs intact."""

import json
import math
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, evenodd, screen
from ier._scale_halves import spearman_brown
from ier.cli import main
from ier.evenodd import calculate_correlations


def _exact_correlation(left: list[float], right: list[float]) -> float:
    """Pearson correlation of exact values; zero variance is undefined, as in stats::cor."""
    xs, ys = [Fraction(value) for value in left], [Fraction(value) for value in right]
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_squares = sum((x - x_mean) ** 2 for x in xs)
    y_squares = sum((y - y_mean) ** 2 for y in ys)
    if x_squares == 0 or y_squares == 0:
        return math.nan
    squared = covariance * covariance / (x_squares * y_squares)
    return min(1.0, math.copysign(math.sqrt(float(squared)), covariance))


def _careless_evenodd(data: np.ndarray, factors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Port careless R/evenodd.R one respondent at a time, returning its negated scores.

    R's long-double ``mean()`` rounds each half mean once, so the port rounds exact
    half means. Correlations of those half means are exact; R's long-double ``cor()``
    can differ from them by about 1e-7 when half means vary only in their last bit.
    """
    scores = np.empty(len(data))
    available = np.empty(len(data), dtype=np.intp)
    for row_number, row in enumerate(np.asarray(data, dtype=float)):
        halves = np.full((len(factors), 2), np.nan)
        start = 0
        for factor, size in enumerate(factors):
            items = row[start : start + size]
            # R positions are 1-based: column 1 holds even positions, column 2 odd ones.
            for column, positions in enumerate((items[1::2], items[0::2])):
                observed = positions[~np.isnan(positions)]
                if len(observed):
                    exact = sum(map(Fraction, observed.tolist()), Fraction(0)) / len(observed)
                    halves[factor, column] = float(exact)
            start += size
        complete = ~np.isnan(halves).any(axis=1)
        available[row_number] = np.count_nonzero(complete)
        even, odd = halves[complete, 0], halves[complete, 1]
        r = _exact_correlation(even.tolist(), odd.tolist()) if len(even) >= 2 else math.nan
        if math.isnan(r):
            scores[row_number] = np.nan
            continue
        corrected = -1.0 if r <= -1 else max(-1.0, 2 * r / (1 + r))
        scores[row_number] = -corrected
    return scores, available


def _legacy_item_pairs(data: np.ndarray, factors: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the established within-factor item-pair reduction."""
    correlation_sum = np.zeros(len(data))
    counts = np.zeros(len(data), dtype=np.intp)
    start = 0
    for size in factors:
        stop = start + size
        if size >= 4:
            correlations = calculate_correlations(
                data[:, start:stop:2], data[:, start + 1 : stop : 2]
            )
            valid = ~np.isnan(correlations)
            np.add(correlation_sum, correlations, out=correlation_sum, where=valid)
            np.add(counts, valid, out=counts, casting="unsafe")
        start = stop
    scores = np.divide(correlation_sum, counts, out=np.full(len(data), np.nan), where=counts > 0)
    return scores, counts


def _structured_responses(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Six correlated 8-item Likert scales: 1000 attentive and 100 uniform random rows."""
    rng = np.random.default_rng(seed)
    traits = rng.normal(size=(1000, 6))
    latent = np.repeat(traits, 8, axis=1) + rng.normal(scale=0.8, size=(1000, 48))
    attentive = np.clip(np.rint(3 + 1.2 * latent), 1, 5)
    random = rng.integers(1, 6, size=(100, 48)).astype(float)
    careless = np.r_[np.zeros(1000, dtype=bool), np.ones(100, dtype=bool)]
    return np.vstack((attentive, random)), careless


def _auc(consistency: np.ndarray, careless: np.ndarray) -> float:
    """Probability that a careless respondent is less consistent; NaN counts as least."""
    scores = np.where(np.isnan(consistency), -np.inf, consistency)
    lower = scores[careless][:, None] < scores[~careless][None, :]
    ties = scores[careless][:, None] == scores[~careless][None, :]
    return float(np.mean(lower) + 0.5 * np.mean(ties))


@pytest.mark.parametrize(
    "factors",
    [[4, 4, 4], [3, 5, 7], [1, 2, 3, 4, 5], [6, 6], [1, 1, 4, 4], [2, 2, 2, 2, 2, 2], [5, 5]],
)
@pytest.mark.parametrize("missing_rate", [0.0, 0.2])
@pytest.mark.parametrize("coding", ["integers", "tenths", "sixths"])
def test_halves_match_careless_reference(
    factors: list[int], missing_rate: float, coding: str
) -> None:
    rng = np.random.default_rng(sum(factors))
    responses = rng.integers(1, 8, size=(300, sum(factors)))
    # Straight-lined odd positions repeat one decimal response within every factor, so
    # their exact half means do not vary even where rounded sums would.
    responses[:60, ::2] = rng.integers(1, 8, size=(60, 1))
    data = {
        "integers": responses.astype(float),
        "tenths": responses / 10,
        "sixths": (responses - 1) / 6,
    }[coding]
    data[rng.random(data.shape) < missing_rate] = np.nan
    expected, expected_available = _careless_evenodd(data, factors)

    scores, available = evenodd(data, factors, diag=True, method="halves")

    np.testing.assert_allclose(scores, -expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(available, expected_available)
    assert np.all(np.isnan(scores) | ((scores >= -1) & (scores <= 1)))


def test_half_with_every_response_missing_drops_only_that_factor() -> None:
    data = np.array(
        [
            [np.nan, 2, np.nan, 4, 1, 1, 2, 2, 5, 5, 4, 4],
            [np.nan, np.nan, np.nan, np.nan, 1, 1, 2, 2, 5, 5, 4, 4],
            [1, 2, 1, 2, 3, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )
    expected, expected_available = _careless_evenodd(data, [4, 4, 4])

    scores, available = evenodd(data, [4, 4, 4], diag=True, method="halves")

    np.testing.assert_array_equal(available, [2, 2, 1])
    np.testing.assert_array_equal(available, expected_available)
    np.testing.assert_allclose(scores, -expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.isnan(scores), [False, False, True])


def test_two_available_factors_correlate_by_direction() -> None:
    data = np.array([[1, 1, 5, 5], [1, 5, 5, 1], [2, 4, 2, 4]])

    scores, available = evenodd(data, [2, 2], diag=True, method="halves")

    np.testing.assert_array_equal(scores[:2], [1.0, -1.0])
    assert np.isnan(scores[2])
    np.testing.assert_array_equal(available, [2, 2, 2])


def test_constant_decimal_half_means_are_unavailable() -> None:
    # Every factor's odd positions hold one decimal response, so careless's odd half
    # means do not vary; rounded sums once made 0.7 * 3 / 3 differ from 0.7 by an ulp.
    rows = np.array(
        [
            [0.7, 0.2, 0.7, 0.4, 0.7, 0.7, 0.9, 0.7, 0.5],
            [0.7, 0.9, 0.7, 0.4, 0.7, 0.7, 0.2, 0.7, 0.5],
            [0.1, 0.2, 0.1, 0.4, 0.1, 0.1, 0.9, 0.1, 0.5],
        ]
    )

    scores, available = evenodd(rows, [5, 2, 2], diag=True, method="halves")

    assert np.isnan(scores).all()
    np.testing.assert_array_equal(available, [3, 3, 3])
    expected, _ = _careless_evenodd(rows, [5, 2, 2])
    assert np.isnan(expected).all()
    # Reordered or rescaled, the profiles keep identical half means.
    assert np.isnan(evenodd(rows[:, [4, 1, 2, 3, 0, 5, 6, 7, 8]], [5, 2, 2], method="halves")).all()
    assert np.isnan(evenodd(rows * 1e300, [5, 2, 2], method="halves")).all()


@pytest.mark.parametrize("value", [3, 0.1, 1.1, 1e-300, 1e300])
def test_straightliners_are_unavailable(value: float) -> None:
    # Correctly rounded half means of a constant profile are identical.
    data = np.full((4, 15), float(value))
    data[1, ::4] = np.nan
    data[3] = np.nan

    scores, available = evenodd(data, [3, 5, 7], diag=True, method="halves")

    assert np.isnan(scores).all()
    np.testing.assert_array_equal(available, [3, 3, 3, 0])
    assert np.isnan(evenodd(np.full((2, 15), 3), [3, 5, 7], method="halves")).all()


def test_halves_count_factors_with_two_finite_half_means() -> None:
    data = np.array([[1, 2, 3, 4, 5, 6], [1, np.inf, 3, 4, 5, 6]])

    scores, available = evenodd(data, [2, 2, 2], diag=True, method="halves")

    np.testing.assert_array_equal(available, [3, 2])
    np.testing.assert_allclose(scores[:1], [1.0])
    assert np.isnan(scores[1])


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_halves_are_invariant_to_layout_batching_and_dtype(layout: str) -> None:
    rng = np.random.default_rng(5)
    integers = rng.integers(1, 6, size=(41, 22))
    data = integers.astype(float)
    data[rng.random(data.shape) < 0.1] = np.nan
    factors = [5, 6, 1, 7, 3]
    expected = evenodd(data, factors, method="halves")
    if layout == "strided":
        data = np.ascontiguousarray(data[::-1, ::-1])[::-1, ::-1]
    else:
        data = np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False

    for budget in (7, 64, 10_000):
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
            actual = evenodd(data, factors, method="halves")
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, original)
    np.testing.assert_allclose(
        evenodd(integers, factors, method="halves"),
        evenodd(integers.astype(float), factors, method="halves"),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("scale", [1e-300, 1e300, 1e308])
def test_halves_are_invariant_to_response_scale(scale: float) -> None:
    rng = np.random.default_rng(11)
    data = rng.choice([-1.0, -0.5, 0.5, 1.0], size=(25, 16))
    data[rng.random(data.shape) < 0.1] = np.nan
    expected = evenodd(data, [4, 4, 4, 4], method="halves")

    actual = evenodd(data * scale, [4, 4, 4, 4], method="halves")

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_item_pairs_remain_bit_identical_and_default() -> None:
    rng = np.random.default_rng(42)
    data = rng.integers(1, 6, size=(200, 23)).astype(float)
    data[rng.random(data.shape) < 0.1] = np.nan
    factors = [4, 5, 3, 6, 1, 4]
    expected_scores, expected_counts = _legacy_item_pairs(data, factors)

    for scores, counts in (
        evenodd(data, factors, diag=True),
        evenodd(data, factors, diag=True, method="item_pairs"),
    ):
        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(counts, expected_counts)


def test_halves_validation() -> None:
    with pytest.raises(ValueError, match="at least two factors"):
        evenodd([[1, 2, 3, 4]], [4], method="halves")
    with pytest.raises(ValueError, match="method must be"):
        evenodd([[1, 2, 3, 4]], [2, 2], method="pairs")  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="must equal number of columns"):
        evenodd([[1, 2, 3, 4]], [2, 3], method="halves")
    with pytest.raises(ValueError, match="positive integers"):
        evenodd([[1, 2, 3, 4]], [2, 0, 2], method="halves")


def test_halves_detect_random_responders() -> None:
    data, careless = _structured_responses(20261005)

    halves = evenodd(data, [8] * 6, method="halves")
    pairs = evenodd(data, [8] * 6)

    assert _auc(halves, careless) >= 0.9
    assert _auc(pairs, careless) < 0.7
    assert np.nanmean(halves[~careless]) > 0.8


def test_spearman_brown_follows_careless_clamp() -> None:
    corrected = spearman_brown(np.array([1.0, 0.5, 0.0, -1 / 3, -0.5, -1.0, np.nan]))

    np.testing.assert_allclose(corrected[:6], [1.0, 2 / 3, 0.0, -1.0, -1.0, -1.0])
    assert np.isnan(corrected[6])


def test_registry_and_cli_select_halves(tmp_path: Path) -> None:
    data = np.random.default_rng(3).integers(1, 6, size=(12, 12))
    expected = evenodd(data, [4, 4, 4], method="halves")
    options = IndexOptions(evenodd_factors=[4, 4, 4], evenodd_method="halves")

    result = screen(data, indices=["evenodd"], options=options, strict=True)

    np.testing.assert_array_equal(result["scores"]["evenodd"], expected)
    path = tmp_path / "responses.csv"
    path.write_text(
        "\n".join(
            [",".join(f"q{column}" for column in range(12))]
            + [",".join(map(str, row)) for row in data.tolist()]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "scores.json"
    arguments = [
        "screen",
        str(path),
        "--indices",
        "evenodd",
        "--evenodd-factors",
        "4,4,4",
        "--format",
        "json",
        "--output",
        str(output),
    ]
    assert main([*arguments, "--evenodd-method", "halves"]) == 0
    halves = json.loads(output.read_text(encoding="utf-8"))["scores"]["evenodd"]
    np.testing.assert_allclose(np.array(halves, dtype=float), expected, equal_nan=True)
    assert main(arguments) == 0
    pairs = json.loads(output.read_text(encoding="utf-8"))["scores"]["evenodd"]
    np.testing.assert_allclose(
        np.array(pairs, dtype=float), evenodd(data, [4, 4, 4]), equal_nan=True
    )
