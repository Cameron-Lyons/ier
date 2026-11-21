"""Split-half reliability correctness, numerical stability, and missing scores."""

import json
import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ier import (
    IndexOptions,
    composite,
    evenodd,
    individual_reliability,
    individual_reliability_flag,
    person_total,
    screen,
)
from ier import reliability as reliability_module
from ier._correlation import selected_row_correlations
from ier._scale_halves import factor_bounds
from ier.cli import main


def _reference(data: np.ndarray, splits: int, seed: int) -> np.ndarray:
    """Calculate the existing mean-correlation correction using scalar paired reductions."""
    rng = np.random.RandomState(seed)
    permutations = [rng.permutation(data.shape[1]) for _ in range(splits)]
    half = data.shape[1] // 2
    result = []
    for row in data:
        correlations = []
        for order in permutations:
            pairs = [
                (float(row[i]), float(row[j]))
                for i, j in zip(order[:half], order[half : 2 * half], strict=True)
                if not np.isnan(row[i]) and not np.isnan(row[j])
            ]
            if len(pairs) < 2 or not all(math.isfinite(a) and math.isfinite(b) for a, b in pairs):
                continue
            first, second = zip(*pairs, strict=True)
            mean1 = math.fsum(first) / len(first)
            mean2 = math.fsum(second) / len(second)
            centered1 = [value - mean1 for value in first]
            centered2 = [value - mean2 for value in second]
            norm1 = math.sqrt(math.fsum(value**2 for value in centered1))
            norm2 = math.sqrt(math.fsum(value**2 for value in centered2))
            if not norm1 or not norm2:
                continue
            if len(pairs) == 2:
                correlation = 1.0 if (first[0] < first[1]) == (second[0] < second[1]) else -1.0
            else:
                covariance = math.fsum(a * b for a, b in zip(centered1, centered2, strict=True))
                correlation = min(1.0, max(-1.0, covariance / norm1 / norm2))
            correlations.append(correlation)
        mean_correlation = math.fsum(correlations) / len(correlations) if correlations else np.nan
        result.append(
            2 * mean_correlation / (1 + mean_correlation) if mean_correlation > -1 else np.nan
        )
    return np.asarray(result)


@pytest.mark.parametrize("items", [4, 5, 12, 13])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
def test_seeded_reliability_matches_scalar_reference(
    items: int, layout: str, missing: bool
) -> None:
    data = np.random.default_rng(28).integers(1, 6, size=(19, items)).astype(float)
    data[0] = 3
    if missing:
        data[::3, 0] = np.nan
        data[1] = np.nan
        data[2, 1:] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    data.flags.writeable = False
    original = data.copy()
    expected = _reference(data, 11, 17)
    for budget in (7, 43):
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
            actual = individual_reliability(data, n_splits=11, random_seed=17)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("scale", [1e-300, 1e-100, 1e100, 1e300])
@pytest.mark.parametrize("missing", [False, True])
def test_reliability_is_invariant_to_response_scale(scale: float, missing: bool) -> None:
    data = np.random.default_rng(4).integers(1, 6, size=(5, 12)).astype(float)
    if missing:
        data[1:3, 0] = np.nan
    expected = individual_reliability(data, n_splits=7, random_seed=17)
    actual = individual_reliability(data * scale, n_splits=7, random_seed=17)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


def test_undefined_correction_remains_unavailable_through_workflows() -> None:
    data = [[1, 2, 3, 4], [1, 3, 2, 4], [3, 3, 3, 3]]
    scores = individual_reliability(data, n_splits=1, random_seed=0)
    assert np.isnan(scores).all()
    assert individual_reliability_flag(data, n_splits=1, random_seed=0).all()
    options = IndexOptions(reliability_n_splits=1, reliability_random_seed=0)
    screened = screen(data, indices=["individual_reliability"], options=options, strict=True)
    assert not screened["errors"]
    np.testing.assert_array_equal(screened["valid_index_counts"], [0, 0, 0])
    np.testing.assert_array_equal(screened["flag_counts"], [0, 0, 0])
    assert np.isnan(composite(data, indices=["individual_reliability"], options=options)).all()


@pytest.mark.parametrize("dtype", [np.int64, np.float32])
def test_numeric_dtypes_match_double_precision(dtype: type) -> None:
    data = np.random.default_rng(28).integers(1, 6, size=(31, 12)).astype(dtype)
    expected = individual_reliability(data.astype(float), n_splits=13, random_seed=17)
    np.testing.assert_allclose(
        individual_reliability(data, n_splits=13, random_seed=17), expected, atol=1e-13, rtol=1e-13
    )


def test_numpy_integer_split_count_is_supported() -> None:
    data = [[1, 2, 3, 4, 5, 6], [5, 3, 1, 6, 4, 2]]
    np.testing.assert_array_equal(
        individual_reliability(data, n_splits=np.int64(7), random_seed=17),
        individual_reliability(data, n_splits=7, random_seed=17),
    )


def test_valid_negative_reliability_is_not_clipped_to_minus_one() -> None:
    # Finite negative split correlations can give corrected values below -1.
    data = [[3, 3, 1, 2, 2, 1]]
    expected = _reference(np.asarray(data, dtype=float), 1, 0)
    actual = individual_reliability(data, n_splits=1, random_seed=0)
    assert expected[0] < -1
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_extreme_rows_are_restored_without_modifying_input(layout: str) -> None:
    rng = np.random.default_rng(3)
    ordinary = rng.choice([-1.0, -0.5, 0.5, 1.0], size=(17, 12))
    ordinary[1:5, 0] = np.nan
    ordinary[0] = np.nan
    expected = individual_reliability(ordinary, n_splits=13, random_seed=17)
    data = np.array(ordinary * 1e308, order=layout)
    original = data.copy()
    data.flags.writeable = False
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = individual_reliability(data, n_splits=13, random_seed=17)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_array_equal(data, original)


def test_infinite_observations_do_not_contaminate_other_splits() -> None:
    data = np.random.default_rng(43).integers(1, 6, size=(11, 13)).astype(float)
    data[::2, 0] = np.inf
    data[1::3, 4] = -np.inf
    data[::3, 1] = np.nan
    expected = _reference(data, 17, 21)
    actual = individual_reliability(data, n_splits=17, random_seed=21)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("value", [True, np.bool_(True), 0, -1, 1.5, np.float64(3)])
def test_invalid_split_counts_raise_value_error(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        individual_reliability([[1, 2, 3, 4]], n_splits=value)  # type: ignore[arg-type]


def test_constant_decimal_profiles_keep_each_indices_zero_variance_policy() -> None:
    data = np.full((7, 14), 1.1)
    assert np.isnan(individual_reliability(data, n_splits=7, random_seed=17)).all()
    assert np.isnan(person_total(data)).all()
    np.testing.assert_array_equal(evenodd(data, [14]), np.zeros(7))
    options = IndexOptions(reliability_n_splits=7, reliability_random_seed=17)
    result = screen(
        data, indices=["individual_reliability", "person_total"], options=options, strict=True
    )
    assert not result["errors"]
    np.testing.assert_array_equal(result["valid_index_counts"], np.zeros(7))
    np.testing.assert_array_equal(result["flag_counts"], np.zeros(7))
    assert np.isnan(composite(data, indices=["individual_reliability"], options=options)).all()


@pytest.mark.parametrize("offset", [1.1, 1e100, 1e-100, 1e308])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_nearly_constant_reliability_preserves_seeded_scores(
    offset: float, missing: bool, layout: str
) -> None:
    steps = np.random.default_rng(22).integers(0, 9, size=(11, 15)).astype(float)
    steps[0] = 0
    if missing:
        steps[1, :3] = np.nan
        steps[2, 1:] = np.nan
        steps[3] = np.nan
    if layout == "strided":
        steps = steps[::-1, ::-1]
    expected = _reference(steps, 13, 17)
    data = offset + steps * np.spacing(offset)
    if layout == "strided":
        data = np.ascontiguousarray(data[::-1, ::-1])[::-1, ::-1]
    else:
        data = np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 100):
        actual = individual_reliability(data, n_splits=13, random_seed=17)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(data, original)


def _scalar_correlation(pairs: list[tuple[float, float]]) -> float:
    """Correlate finite pairs exactly with the kernel's two-pair and zero-variance rules."""
    if len(pairs) < 2 or not all(math.isfinite(a) and math.isfinite(b) for a, b in pairs):
        return math.nan
    first, second = zip(*pairs, strict=True)
    # Exact moments keep half means that differ only in their last bit meaningful.
    exact1, exact2 = [Fraction(value) for value in first], [Fraction(value) for value in second]
    mean1, mean2 = sum(exact1) / len(exact1), sum(exact2) / len(exact2)
    squares1 = sum((value - mean1) ** 2 for value in exact1)
    squares2 = sum((value - mean2) ** 2 for value in exact2)
    if not squares1 or not squares2:
        return math.nan
    if len(pairs) == 2:
        return 1.0 if (first[0] < first[1]) == (second[0] < second[1]) else -1.0
    covariance = sum((a - mean1) * (b - mean2) for a, b in zip(exact1, exact2, strict=True))
    correlation = math.sqrt(float(covariance * covariance / (squares1 * squares2)))
    return min(1.0, max(-1.0, math.copysign(correlation, covariance)))


def _exact_mean(values: list[float]) -> Fraction:
    """Return the exact rational mean of observed responses."""
    return sum(map(Fraction, values), Fraction(0)) / len(values)


def _rounded_mean(values: list[float]) -> float:
    """Round the exact mean once, as R's long-double mean() does."""
    return float(_exact_mean(values))


def _factor_reference(
    data: np.ndarray,
    factors: list[int],
    splits: int,
    permutation: Callable[[int], np.ndarray],
    *,
    half_mean: Callable[[list[float]], float | Fraction] = _rounded_mean,
) -> np.ndarray:
    """Average scalar across-scale correlations of random within-scale half means."""
    plans = []
    for _ in range(splits):
        plan = []
        start = 0
        for size in factors:
            order = start + permutation(size)
            plan.append((order[: size // 2], order[size // 2 : 2 * (size // 2)]))
            start += size
        plans.append(plan)
    result = []
    for row in np.asarray(data, dtype=float):
        observed = row[~np.isnan(row)]
        if not len(observed) or np.all(observed == observed[0]):
            result.append(np.nan)
            continue
        correlations = []
        for plan in plans:
            pairs = []
            for first, second in plan:
                first_values = [float(row[i]) for i in first if not np.isnan(row[i])]
                second_values = [float(row[i]) for i in second if not np.isnan(row[i])]
                if first_values and second_values:
                    pairs.append((half_mean(first_values), half_mean(second_values)))
            correlation = _scalar_correlation(pairs)
            if not math.isnan(correlation):
                correlations.append(correlation)
        if not correlations:
            result.append(np.nan)
            continue
        mean = math.fsum(correlations) / len(correlations)
        result.append(-1.0 if mean <= -1 else max(-1.0, 2 * mean / (1 + mean)))
    return np.asarray(result)


def _structured_responses(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Six correlated 8-item Likert scales: 1000 attentive and 100 uniform random rows."""
    rng = np.random.default_rng(seed)
    traits = rng.normal(size=(1000, 6))
    latent = np.repeat(traits, 8, axis=1) + rng.normal(scale=0.8, size=(1000, 48))
    attentive = np.clip(np.rint(3 + 1.2 * latent), 1, 5)
    random = rng.integers(1, 6, size=(100, 48)).astype(float)
    careless = np.r_[np.zeros(1000, dtype=bool), np.ones(100, dtype=bool)]
    return np.vstack((attentive, random)), careless


@pytest.mark.parametrize("factors", [[4, 4, 4], [3, 5, 2, 6], [1, 4, 4, 5], [2, 2]])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.1])
def test_scale_aware_reliability_matches_scalar_reference(
    factors: list[int], layout: str, missing: bool, scale: float
) -> None:
    data = np.random.default_rng(29).integers(1, 6, size=(23, sum(factors))) * scale
    data[0] = 3
    # A constant 0.1 profile has identical half means and stays unavailable.
    data[1] = 0.1
    if missing:
        data[::3, 1] = np.nan
        data[2] = np.nan
        data[3, 1:] = np.nan
        data[4, : factors[0]] = np.nan
    if layout == "strided":
        data = np.ascontiguousarray(data[::-1, ::-1])[::-1, ::-1]
    else:
        data = np.array(data, order=layout)
    data.flags.writeable = False
    original = data.copy()
    expected = _factor_reference(data, factors, 11, np.random.RandomState(17).permutation)
    for budget in (7, 43, 10_000):
        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget):
            actual = individual_reliability(data, n_splits=11, random_seed=17, factors=factors)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(data, original)
    assert np.isnan(actual[:2]).all()
    assert np.all(np.isnan(actual) | ((actual >= -1) & (actual <= 1)))


@pytest.mark.parametrize("scale", [1e-300, 1e300, 1e308])
def test_scale_aware_reliability_is_invariant_to_response_scale(scale: float) -> None:
    data = np.random.default_rng(8).choice([-1.0, -0.5, 0.5, 1.0], size=(17, 15))
    data[1:4, 0] = np.nan
    expected = individual_reliability(data, n_splits=9, random_seed=3, factors=[5, 5, 5])
    actual = individual_reliability(data * scale, n_splits=9, random_seed=3, factors=[5, 5, 5])
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_scale_aware_reliability_ignores_decimal_rounding_noise() -> None:
    # Every split averages repeated 0.7 responses into one half-mean vector whose exact
    # values do not vary, so no split is usable; rounded sums once scored the row -1.
    data = np.array(
        [
            [0.7] * 8 + [0.7, 0.2],
            [0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.3, 0.3, 0.9, 0.2],
        ]
    )

    scores = individual_reliability(data, n_splits=20, random_seed=0, factors=[6, 2, 2])

    expected = _factor_reference(data, [6, 2, 2], 20, np.random.RandomState(0).permutation)
    assert np.isnan(scores[0])
    np.testing.assert_allclose(scores, expected, rtol=1e-13, atol=1e-13)
    assert individual_reliability_flag(data, n_splits=20, random_seed=0, factors=[6, 2, 2])[0]


def test_scale_aware_reliability_accepts_integer_responses() -> None:
    data = np.random.default_rng(9).integers(1, 6, size=(31, 12))
    expected = individual_reliability(
        data.astype(float), n_splits=7, random_seed=1, factors=[4, 4, 4]
    )
    actual = individual_reliability(data, n_splits=7, random_seed=1, factors=[4, 4, 4])
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("factors", [None, [3, 3, 3]])
def test_unseeded_reliability_does_not_touch_global_rng(factors: list[int] | None) -> None:
    data = np.random.default_rng(2).integers(1, 6, size=(9, 9)).astype(float)

    np.random.seed(123)
    expected_next = np.random.random()

    np.random.seed(123)
    state = np.random.get_state()
    individual_reliability(data, n_splits=5, factors=factors)
    after = np.random.get_state()
    actual_next = np.random.random()

    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert actual_next == expected_next


def test_integer_seeds_keep_established_split_sequence() -> None:
    data = np.random.default_rng(6).integers(1, 6, size=(12, 11)).astype(float)
    recorded: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(x: np.ndarray, left: np.ndarray, right: np.ndarray, **options: object) -> np.ndarray:
        recorded.append((left.copy(), right.copy()))
        return selected_row_correlations(x, left, right, **options)  # type: ignore[arg-type]

    with patch("ier.reliability.selected_row_correlations", side_effect=spy):
        individual_reliability(data, n_splits=6, random_seed=41)

    established = np.random.RandomState(41)
    assert len(recorded) == 6
    for left, right in recorded:
        order = established.permutation(11)
        np.testing.assert_array_equal(left, order[:5])
        np.testing.assert_array_equal(right, order[5:10])


@pytest.mark.parametrize("factors", [None, [4, 3, 5]])
def test_generator_seeds_are_reproducible_and_advanced(factors: list[int] | None) -> None:
    data = np.random.default_rng(12).integers(1, 6, size=(30, 12)).astype(float)
    data[::4, 2] = np.nan

    first = individual_reliability(
        data, n_splits=7, random_seed=np.random.default_rng(5), factors=factors
    )
    second = individual_reliability(
        data, n_splits=7, random_seed=np.random.default_rng(5), factors=factors
    )
    generator = np.random.default_rng(5)
    individual_reliability(data, n_splits=7, random_seed=generator, factors=factors)
    advanced = individual_reliability(data, n_splits=7, random_seed=generator, factors=factors)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, advanced, equal_nan=True)
    np.testing.assert_array_equal(
        individual_reliability_flag(
            data, n_splits=7, random_seed=np.random.default_rng(5), factors=factors
        ),
        (first < 0.3) | np.isnan(first),
    )
    if factors is not None:
        expected = _factor_reference(data, factors, 7, np.random.default_rng(5).permutation)
        np.testing.assert_allclose(first, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        ([], "cannot be empty"),
        ([12], "at least two factors"),
        ([6, 0, 6], "positive integers"),
        ([6, 1.5], "positive integers"),
        ([6, True, 5], "positive integers"),
        ([6, 5], "must equal number of columns"),
    ],
)
def test_invalid_reliability_factors_raise_value_error(factors: list[int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        individual_reliability(np.ones((2, 12)), factors=factors)


def test_single_item_scales_contribute_no_halves() -> None:
    data = np.random.default_rng(14).integers(1, 6, size=(19, 10)).astype(float)
    expected = _factor_reference(data, [1, 3, 1, 2, 3], 9, np.random.RandomState(0).permutation)

    actual = individual_reliability(data, n_splits=9, random_seed=0, factors=[1, 3, 1, 2, 3])

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert np.isfinite(actual).any()
    # Only singleton scales and one larger scale leave fewer than two half-mean pairs.
    assert np.isnan(individual_reliability(data, n_splits=3, factors=[1, 1, 8])).all()


def test_scale_aware_reliability_detects_random_responders() -> None:
    data, careless = _structured_responses(20261005)

    scores = individual_reliability(data, random_seed=0, factors=[8] * 6)

    ranks = np.where(np.isnan(scores), -np.inf, scores)
    lower = ranks[careless][:, None] < ranks[~careless][None, :]
    ties = ranks[careless][:, None] == ranks[~careless][None, :]
    assert float(np.mean(lower) + 0.5 * np.mean(ties)) >= 0.95
    assert np.nanmean(scores[~careless]) > 0.8


def test_registry_and_cli_pass_reliability_factors(tmp_path: Path) -> None:
    data = np.random.default_rng(15).integers(1, 6, size=(10, 9))
    expected = individual_reliability(data, n_splits=4, random_seed=2, factors=[3, 3, 3])
    options = IndexOptions(
        reliability_n_splits=4, reliability_random_seed=2, reliability_factors=[3, 3, 3]
    )

    result = screen(data, indices=["individual_reliability"], options=options, strict=True)

    np.testing.assert_array_equal(result["scores"]["individual_reliability"], expected)
    path = tmp_path / "responses.csv"
    rows = [",".join(f"q{column}" for column in range(9))]
    rows.extend(",".join(map(str, row)) for row in data.tolist())
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    output = tmp_path / "scores.json"
    code = main(
        [
            "screen",
            str(path),
            "--indices",
            "individual_reliability",
            "--reliability-n-splits",
            "4",
            "--reliability-random-seed",
            "2",
            "--reliability-factors",
            "3,3,3",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    scores = json.loads(output.read_text(encoding="utf-8"))["scores"]["individual_reliability"]
    np.testing.assert_allclose(np.array(scores, dtype=float), expected, equal_nan=True)


_Exact = reliability_module._ExactSplitCorrelations


def _rounded_reliability(data: np.ndarray, factors: list[int], seed: int) -> np.ndarray:
    """Score with the correctly rounded half means, bypassing the exact integer scorer."""
    with patch.object(_Exact, "build", return_value=None):
        return individual_reliability(data, n_splits=40, random_seed=seed, factors=factors)


def _exact_reliability(data: np.ndarray, factors: list[int], seed: int) -> tuple[np.ndarray, int]:
    """Score normally and count the respondents the exact integer scorer received."""
    received: list[int] = []
    score_block = _Exact.__call__

    def counted(self: object, block: np.ndarray, *totals: np.ndarray) -> None:
        received.append(len(block))
        score_block(self, block, *totals)  # type: ignore[arg-type]

    with patch.object(_Exact, "__call__", counted):
        scores = individual_reliability(data, n_splits=40, random_seed=seed, factors=factors)
    return scores, sum(received)


def _survey(seed: int, shape: tuple[int, int], low: int, high: int, missing: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(shape[0], 1))
    middle = (low + high) / 2
    spread = (high - low) / 4
    data = np.clip(np.rint(middle + spread * (latent + rng.normal(size=shape))), low, high)
    data[: shape[0] // 5] = rng.integers(low, high + 1, size=(shape[0] // 5, shape[1]))
    data[rng.random(shape) < missing] = np.nan
    data[1] = low  # constant
    data[2] = np.nan  # empty
    data[3, ::2] = np.nan  # every second item
    return data


def _exact_scorer(factors: list[int]) -> reliability_module._ExactSplitCorrelations | None:
    """Build the exact scorer for three random splits of ``factors``."""
    bounds = factor_bounds(factors)
    rng = np.random.default_rng(0)
    splits = []
    for _ in range(3):
        first, second = [], []
        for start, stop in bounds:
            half = (stop - start) // 2
            order = start + rng.permutation(stop - start)
            first.append(order[:half])
            second.append(order[half : 2 * half])
        splits.append((first, second))
    return _Exact.build(splits, bounds)


@pytest.mark.parametrize("missing", [0.0, 0.02, 0.3])
@pytest.mark.parametrize(
    ("factors", "low", "high"),
    [
        ([10, 10, 10, 10], 1, 5),
        ([2, 2, 2, 2, 2, 2], 1, 7),
        ([7, 4, 9], -3, 3),
        ([1, 6, 5], 0, 100),
        ([3, 3], 1, 5),
    ],
    ids=["four-scales", "pairs", "signed-odd", "one-item-scale", "two-scales"],
)
def test_exact_integer_scorer_matches_rounded_half_means(
    factors: list[int], low: int, high: int, missing: float
) -> None:
    data = _survey(sum(factors) + int(missing * 100), (240, sum(factors)), low, high, missing)
    exact, received = _exact_reliability(data, factors, seed=8)
    assert received >= 200  # every varying respondent's responses qualify
    expected = _rounded_reliability(data, factors, seed=8)
    np.testing.assert_array_equal(np.isnan(exact), np.isnan(expected))
    np.testing.assert_allclose(exact, expected, rtol=1e-13, atol=1e-13)
    assert np.isnan(exact[1:3]).all()


def test_exact_correlations_of_two_scales_are_exactly_signed() -> None:
    rng = np.random.default_rng(14)
    left = rng.integers(-9, 10, size=(2, 3, 500)).astype(float) * 2520
    right = rng.integers(-9, 10, size=(2, 3, 500)).astype(float) * 2520
    correlations, usable = reliability_module._exact_correlations(left, right, 2)
    tied = (left[0] == left[1]) | (right[0] == right[1])
    np.testing.assert_array_equal(usable, ~tied)
    expected = np.sign(left[1] - left[0]) * np.sign(right[1] - right[0])
    np.testing.assert_array_equal(correlations[usable], expected[usable])
    assert np.isnan(correlations[tied]).all()


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.int64, np.float32, bool])
def test_exact_integer_scorer_accepts_integer_like_dtypes(dtype: type) -> None:
    data = np.nan_to_num(_survey(5, (120, 12), 0, 1 if dtype is bool else 6, 0.0), nan=1.0)
    data[2] = 0
    typed = data.astype(dtype)
    exact, received = _exact_reliability(typed, [4, 4, 4], seed=3)
    assert received > 0
    expected = _rounded_reliability(typed, [4, 4, 4], seed=3)
    np.testing.assert_array_equal(np.isnan(exact), np.isnan(expected))
    np.testing.assert_allclose(exact, expected, rtol=1e-13, atol=1e-13)


def test_exact_scorer_selects_rows_by_their_own_responses() -> None:
    scorer = _exact_scorer([10, 10])  # half sizes 5: scales 5 complete, 60 with gaps
    assert scorer is not None
    block = np.tile(np.arange(1.0, 21.0) % 5 + 1, (7, 1))
    block[1, 4] = 2.5  # fractional
    block[2, 4] = np.inf  # infinite
    block[3, 4] = np.nan  # missing, small responses
    block[4] *= 2.0**30  # too large even when complete
    block[5] *= 2.0**20  # complete: 2 * 5 * 2**21 fits within 2**26.5
    block[6] = block[5]
    block[6, 0] = np.nan  # the same responses with a gap: 2 * 60 * 2**21 does not
    np.testing.assert_array_equal(
        scorer.eligible(block), [True, False, False, True, False, True, False]
    )
    integers = np.array([[1, 2] * 10, [2**60, 1] * 10], dtype=np.int64)
    np.testing.assert_array_equal(scorer.eligible(integers), [True, False])
    np.testing.assert_array_equal(scorer.eligible(integers[:1] == 1), [True])


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(float).max,
    reason="long double is double precision on this platform",
)
def test_exact_scorer_rejects_extended_precision_beyond_float64() -> None:
    scorer = _exact_scorer([4, 4])
    assert scorer is not None
    huge = np.longdouble(10) ** 400
    block = np.array([[1, 2, 3, 4, 1, 2, 3, 4], [huge, 1, 2, 3, 4, 1, 2, 3]], dtype=np.longdouble)
    np.testing.assert_array_equal(scorer.eligible(block), [True, False])


def test_long_scales_with_missing_responses_use_the_rounded_scorer() -> None:
    # Gaps allow any half size from 1 to 20, whose common multiple is too large.
    scorer = _exact_scorer([40, 40])
    assert scorer is not None
    block = _survey(3, (8, 80), 1, 5, 0.0)[4:]
    block[0, 7] = np.nan
    np.testing.assert_array_equal(scorer.eligible(block), [False, True, True, True])


def test_exact_scorer_is_skipped_when_no_respondent_could_qualify() -> None:
    assert _exact_scorer([5, 1]) is None  # one scale has halves
    assert _exact_scorer([2 * 7919, 2 * 7907]) is None  # lcm(7919, 7907) is too large
    assert _exact_scorer([2 * 709, 2 * 709]) is not None


def test_empty_or_zero_long_scales_stay_unavailable() -> None:
    # Gaps would need lcm(1..709), beyond float range; no row reaches either scorer.
    zeros = np.zeros((3, 2 * 1418))
    zeros[0, 0] = np.nan
    for data in (zeros, np.full((3, 2 * 1418), np.nan)):
        scores = individual_reliability(data, n_splits=3, random_seed=0, factors=[1418, 1418])
        assert np.isnan(scores).all()


def test_scores_depend_only_on_each_respondents_own_responses() -> None:
    data = _survey(33, (200, 24), 1, 5, 0.05)
    data[10:14] = np.nan_to_num(data[10:14], nan=3.0)
    expected = individual_reliability(data, n_splits=30, random_seed=6, factors=[8, 8, 8])
    changed = data.copy()
    changed[10, 3] = 2.5  # fractional: this row takes the rounded scorer
    changed[11, 0] = np.nan  # a first gap in an otherwise complete matrix block
    changed[12, :8] *= 2.0**40  # too large for exact moments
    changed[13, 5] = np.inf
    rescored = individual_reliability(changed, n_splits=30, random_seed=6, factors=[8, 8, 8])
    others = np.ones(len(data), dtype=bool)
    others[10:14] = False
    np.testing.assert_array_equal(rescored[others], expected[others])


def test_exact_scorer_correlates_exact_half_means_far_from_zero() -> None:
    # Half sizes 3, 5, and 6 give fractional half means near 400,000, which double
    # precision rounds coarsely relative to their spread across scales.
    rng = np.random.default_rng(41)
    factors = [6, 10, 12]
    data = 400_000 + rng.integers(0, 4, size=(30, sum(factors))).astype(float)
    actual, received = _exact_reliability(data, factors, seed=5)
    assert received == len(data)
    expected = _factor_reference(
        data, factors, 40, np.random.RandomState(5).permutation, half_mean=_exact_mean
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_exact_scores_do_not_depend_on_row_batches() -> None:
    data = _survey(21, (300, 24), 1, 5, 0.05)
    data[100:150] = np.nan_to_num(data[100:150], nan=3.0)  # a complete batch
    data[200:260] = np.nan  # an empty batch
    expected = individual_reliability(data, n_splits=30, random_seed=4, factors=[8, 8, 8])
    with (
        patch("ier.reliability._ROW_BATCH_ELEMENTS", 997),
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 997),
    ):
        batched = individual_reliability(data, n_splits=30, random_seed=4, factors=[8, 8, 8])
    np.testing.assert_array_equal(batched, expected)
    with patch("ier.reliability._SINGLE_THREAD_PRODUCT", 50):
        stepped = individual_reliability(data, n_splits=30, random_seed=4, factors=[8, 8, 8])
    np.testing.assert_array_equal(stepped, expected)
