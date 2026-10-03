"""Exact item profiles and correlations survive extreme floating response units."""

from decimal import Decimal, localcontext
from fractions import Fraction
from importlib import import_module
from itertools import combinations, permutations
from unittest.mock import patch

import numpy as np
import pytest

from ier import person_total
from ier._column_statistics import column_mean_profile


def _exact_person_total(data: np.ndarray) -> np.ndarray:
    """Apply the Pearson definition to exact rational sample item means."""
    profile: list[Fraction | None] = []
    for column in data.T:
        observed = [Fraction(float(value)) for value in column if np.isfinite(value)]
        profile.append(sum(observed, start=Fraction()) / len(observed) if observed else None)
    result = []
    with localcontext() as context:
        context.prec = 800
        for row in data:
            paired = [
                (Fraction(float(value)), mean)
                for value, mean in zip(row, profile, strict=True)
                if np.isfinite(value) and mean is not None
            ]
            if len(paired) < 2:
                result.append(np.nan)
                continue
            values, means = zip(*paired, strict=True)
            center_left = sum(values, start=Fraction()) / len(values)
            center_right = sum(means, start=Fraction()) / len(means)
            left = [value - center_left for value in values]
            right = [mean - center_right for mean in means]
            numerator = sum((a * b for a, b in zip(left, right, strict=True)), start=Fraction())
            squares = sum((value * value for value in left), start=Fraction()) * sum(
                (value * value for value in right), start=Fraction()
            )
            if squares == 0:
                result.append(np.nan)
                continue
            covariance = Decimal(numerator.numerator) / Decimal(numerator.denominator)
            variance = Decimal(squares.numerator) / Decimal(squares.denominator)
            result.append(float(covariance / variance.sqrt()))
    return np.asarray(result)


def _profile_data(structure: str) -> np.ndarray:
    data = np.random.default_rng(38).integers(0, 13, size=(17, 7)).astype(float)
    data[0] = 4
    if structure == "subnormal":
        data *= np.nextafter(0.0, 1.0)
    elif structure == "tiny_baseline":
        data = np.finfo(float).tiny + data * np.nextafter(0.0, 1.0)
    else:
        baseline = {"decimal": 1.1, "negative": -1.1, "large": 1e300}[structure]
        data = baseline + data * abs(np.spacing(baseline))
    return data


@pytest.mark.parametrize(
    "structure", ["subnormal", "tiny_baseline", "decimal", "negative", "large"]
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
def test_extreme_float_profiles_match_exact_pearson_definition(
    structure: str, layout: str, missing: bool
) -> None:
    data = _profile_data(structure)
    if missing:
        data[1::3, 2:4] = np.nan
        data[2] = np.nan
        data[3, 1:] = np.nan
        data[:, -1] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _exact_person_total(data)
    for budget in [14, 2000]:
        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget),
            patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", budget),
        ):
            np.testing.assert_allclose(person_total(data), expected, rtol=3e-14, atol=3e-15)
            strict = person_total(data, na_rm=False)
        if missing:
            assert np.isnan(strict).all()
        else:
            np.testing.assert_allclose(strict, expected, rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("value", [0.0, np.nextafter(0.0, 1.0), 1.1, 1e300])
def test_constant_sample_profiles_remain_unavailable(value: float) -> None:
    data = np.full((17, 7), value)
    data[::3, ::2] = np.nan
    assert np.isnan(person_total(data)).all()


@pytest.mark.parametrize("ignore_nan", [False, True])
def test_profile_repair_preserves_nonfinite_item_means(ignore_nan: bool) -> None:
    data = np.array([[1.1, 1.1, np.inf, np.inf], [1.1, 1.1, -np.inf, np.nan]])
    np.testing.assert_array_equal(
        column_mean_profile(data, ignore_nan=ignore_nan),
        [0.0, 0.0, np.nan, np.inf if ignore_nan else np.nan],
    )


def test_extreme_profile_repairs_keep_bounded_owned_workspaces() -> None:
    data = _profile_data("large")
    data[1, 2] = np.nan
    data.flags.writeable = False
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 14),
        patch("ier._column_statistics.np.ldexp", wraps=np.ldexp) as scales,
    ):
        profile = column_mean_profile(data, ignore_nan=True)
    blocks = [call.args[0] for call in scales.call_args_list if np.ndim(call.args[0]) == 2]
    assert len(blocks) > 1
    assert all(block.size <= 14 and block.flags.writeable for block in blocks)
    np.testing.assert_allclose(
        person_total(data), _exact_person_total(data), rtol=3e-14, atol=3e-15
    )
    assert np.isfinite(profile).all()


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
def test_cancellation_keeps_tiny_responses_among_enormous_observations(
    layout: str, missing: bool
) -> None:
    smallest = np.nextafter(0.0, 1.0)
    data = np.array(
        [
            [1e308, 1e308, 1e308, 1e308],
            [-1e308, -1e308, -1e308, -1e308],
            [smallest, 2 * smallest, 3 * smallest, 4 * smallest],
            [2 * smallest, smallest, 4 * smallest, 3 * smallest],
        ]
    )
    if missing:
        data[2:, 1] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _exact_person_total(data)
    with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 4):
        np.testing.assert_allclose(person_total(data), expected, rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("small_scale", [1.0, np.nextafter(0.0, 1.0)])
def test_large_sample_profile_preserves_independent_small_respondent_variation(
    layout: str, missing: bool, small_scale: float
) -> None:
    baseline = 1e308
    large = baseline + np.arange(7) * np.spacing(baseline)
    data = np.array(
        [large, large, np.arange(1, 8) * small_scale, np.arange(7, 0, -1) * small_scale]
    )
    if missing:
        data[:, 2] = np.nan
        data[2:, 4] = np.nan
        data[:, -1] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _exact_person_total(data)
    for budget in [7, 14, 1000]:
        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget),
            patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", budget),
        ):
            np.testing.assert_allclose(person_total(data), expected, rtol=3e-14, atol=3e-15)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("budget", [24, 48, 2000])
def test_equal_exact_item_means_survive_different_cancellation_order(
    layout: str, budget: int
) -> None:
    data = np.array(list(permutations([1.0, 1e-20, -1.0, 1.1]))).T
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    data.flags.writeable = False
    expected = _exact_person_total(data)
    assert np.isnan(expected).all()
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget),
        patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", budget),
    ):
        np.testing.assert_array_equal(person_total(data), expected)


@pytest.mark.parametrize("baseline", [1e100, 1e200, 1e300, 1e308])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("missing_row", [0, 2])
@pytest.mark.parametrize("small_scale", [1.0, np.nextafter(0.0, 1.0)])
def test_pairwise_item_means_retain_original_sample_denominators(
    baseline: float, layout: str, missing_row: int, small_scale: float
) -> None:
    large = baseline + np.arange(4) * np.spacing(baseline)
    data = np.array([large, large, np.arange(1, 5) * small_scale])
    data[missing_row, 2] = np.nan
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _exact_person_total(data)
    for budget in [4, 8, 1000]:
        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", budget),
            patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", budget),
        ):
            np.testing.assert_allclose(person_total(data), expected, rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("remote_baseline", [1e307, 1e300, 1e100, 1.1, 1e-100, 1e-300])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_pairwise_profiles_resolve_remote_nearconstant_item_groups(
    remote_baseline: float, layout: str
) -> None:
    large = 1e308 + np.arange(3) * np.spacing(1e308)
    remote = remote_baseline + np.arange(3) * np.spacing(remote_baseline)
    profile = np.concatenate((large, remote))
    data = np.array(
        [profile, profile, [1, 2, 3, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, 1, 2, 3]]
    )
    data = data[::-1, ::-1] if layout == "strided" else np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = _exact_person_total(data)
    with (
        patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 7),
        patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", 7),
    ):
        np.testing.assert_allclose(person_total(data), expected, rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(data, original)


def test_repeated_pairwise_masks_reuse_exact_sample_profiles_across_batches() -> None:
    large = 1e308 + np.arange(4) * np.spacing(1e308)
    data = np.vstack((large, large, np.tile([1.0, 2.0, np.nan, 4.0], (31, 1))))
    data.flags.writeable = False
    module = import_module("ier.person_total")
    with (
        patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", 4),
        patch("ier.person_total._exact_mean_profile", wraps=module._exact_mean_profile) as exact,
    ):
        scores = person_total(data)
    np.testing.assert_allclose(scores, _exact_person_total(data), rtol=3e-14, atol=3e-15)
    assert exact.call_count == 1
    assert exact.call_args.args[0] is data


@pytest.mark.parametrize("budget", [9, 90, 9000])
@pytest.mark.parametrize("capacity", [1, 32])
def test_pairwise_profile_eviction_is_bounded_and_preserves_exact_scores(
    budget: int, capacity: int
) -> None:
    large = 1e308 + np.arange(8) * np.spacing(1e308)
    profile = np.concatenate((large, [1e299]))
    masks = list(combinations(range(8), 3)) * 2
    data = np.full((len(masks) + 2, 9), np.nan)
    data[:2] = profile
    for row, columns in enumerate(masks, start=2):
        data[row, list(columns)] = [1, 2, 3]
    original = data.copy()
    data.flags.writeable = False
    module = import_module("ier.person_total")
    repair = module._repair_pairwise_profiles
    observed_cache_sizes: list[tuple[int, int]] = []

    def checked_repair(
        sample: np.ndarray,
        block: np.ndarray,
        means: np.ndarray,
        scores: np.ndarray,
        cache: dict[bytes, np.ndarray],
    ) -> None:
        assert block.size <= max(budget, block.shape[1])
        repair(sample, block, means, scores, cache)
        observed_cache_sizes.append((len(cache), sum(value.size for value in cache.values())))

    with (
        patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", budget),
        patch("ier.person_total._MAX_PROFILE_CACHE", capacity),
        patch("ier.person_total._repair_pairwise_profiles", checked_repair),
    ):
        scores = person_total(data)
    np.testing.assert_allclose(scores, _exact_person_total(data), rtol=3e-14, atol=3e-15)
    assert observed_cache_sizes
    assert all(count <= capacity and size <= budget for count, size in observed_cache_sizes)
    np.testing.assert_array_equal(data, original)
