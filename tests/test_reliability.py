"""Split-half reliability correctness, numerical stability, and missing scores."""

import math
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
