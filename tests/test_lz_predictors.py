"""Likelihood and ability-estimation proofs across extreme finite IRT units."""

import math
from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
import pytest

from ier import lz_flag
from ier.lz import _estimate_theta


def _decimal_scores(
    data: np.ndarray,
    difficulties: np.ndarray,
    discriminations: np.ndarray,
    abilities: np.ndarray,
    *,
    na_rm: bool,
) -> np.ndarray:
    """Compute Bernoulli likelihood moments without floating predictor products."""
    with localcontext() as context:
        context.prec = 1100
        minimum = Decimal.from_float(1e-10)
        maximum = Decimal.from_float(1 - 1e-10)
        lower = (minimum / (1 - minimum)).ln()
        upper = (maximum / (1 - maximum)).ln()
        result = []
        for row, ability in zip(data, abilities, strict=True):
            if np.isnan(ability) or (not na_rm and np.isnan(row).any()):
                result.append(np.nan)
                continue
            numerator = Decimal(0)
            variance = Decimal(0)
            observed = False
            for response, difficulty, discrimination in zip(
                row, difficulties, discriminations, strict=True
            ):
                if np.isnan(response):
                    continue
                observed = True
                if np.isnan(difficulty) or np.isnan(discrimination):
                    numerator = Decimal("NaN")
                    break
                odds = Decimal.from_float(float(discrimination)) * (
                    Decimal.from_float(float(ability)) - Decimal.from_float(float(difficulty))
                )
                if odds <= lower:
                    probability, odds = minimum, lower
                elif odds >= upper:
                    probability, odds = maximum, upper
                else:
                    probability = 1 / (1 + (-odds).exp())
                numerator += (Decimal.from_float(float(response)) - probability) * odds
                variance += probability * (1 - probability) * odds * odds
            if not observed or numerator.is_nan():
                result.append(np.nan)
            else:
                result.append(float(numerator / variance.sqrt()) if variance else 0.0)
    return np.asarray(result)


PARAMETERS = [
    ([-1e308, -9e307, 0, 1e308], [1e-308] * 4, [1e308, -1e308, 0, 1e308]),
    ([-1e308, 1e308, 0, 0], [0, 0, 0, 0], [1e308, -1e308, 0, 1e308]),
    ([-10, 5, 1, -1], [np.finfo(float).max] * 4, [2, -2, 0, 2]),
    (
        [-1e-300, -2e-300, 1e-300, 2e-300],
        [0.5e-300, 1.5e-300, 0.5e-300, 1.5e-300],
        [0, 1e-300, -1e-300, 0],
    ),
    ([-1e-323, -2e-323, 1e-323, 2e-323], [0.2, 0.3, 0.2, 0.3], [0, 1e-323, -1e-323, 0]),
    ([-1, -1e-300, 1e-300, 1], [1, 1e-300, 1e-300, 1], [0, 1e-300, -1e-300, 0]),
]


@pytest.mark.parametrize(
    "parameters",
    PARAMETERS,
    ids=[
        "difference-overflow",
        "zero-discrimination",
        "product-overflow",
        "product-underflow",
        "subnormal-rounding",
        "mixed-units",
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_extreme_calibrated_predictions_match_decimal(
    parameters: tuple[list[float], list[float], list[float]],
    layout: str,
    na_rm: bool,
) -> None:
    difficulties, discriminations, abilities = [np.asarray(values) for values in parameters]
    data = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [1, np.nan, 0, 1], [np.nan] * 4])
    if layout == "strided":
        backing = np.zeros((8, 8))
        backing[::2, ::2] = data
        data = backing[::2, ::2]
    else:
        data = np.array(data, order=layout)
    inputs = [data, difficulties, discriminations, abilities]
    originals = [value.copy() for value in inputs]
    for value in inputs:
        value.flags.writeable = False
    expected = _decimal_scores(data, difficulties, discriminations, abilities, na_rm=na_rm)
    with np.errstate(all="raise"), patch("ier.lz._LZ_BATCH_ELEMENTS", 5):
        scores, flags = lz_flag(
            data,
            difficulty=difficulties,
            discrimination=discriminations,
            theta=abilities,
            threshold=-1.5,
            na_rm=na_rm,
        )
    np.testing.assert_allclose(scores, expected, rtol=5e-14, atol=2e-15)
    np.testing.assert_array_equal(flags, expected < -1.5)
    for value, original in zip(inputs, originals, strict=True):
        np.testing.assert_array_equal(value, original)


def test_tiny_products_preserve_missing_parameter_availability() -> None:
    data = np.asarray([[1, 0, np.nan], [1, 0, 1], [np.nan] * 3])
    difficulties = np.asarray([-1e-300, 2e-300, np.nan])
    discriminations = np.asarray([1e-300, 1e-300, np.nan])
    abilities = np.zeros(3)
    expected = _decimal_scores(data, difficulties, discriminations, abilities, na_rm=True)
    with np.errstate(all="raise"):
        scores, flags = lz_flag(
            data, difficulty=difficulties, discrimination=discriminations, theta=abilities
        )
    np.testing.assert_allclose(scores, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(flags, expected < -1.96)


def _decimal_theta(
    data: np.ndarray, a: np.ndarray, b: np.ndarray, scale: float, *, na_rm: bool
) -> np.ndarray:
    """Find the monotone likelihood-score root by independent Decimal bisection."""
    with localcontext() as context:
        context.prec = 80
        unit = Decimal.from_float(scale)
        coefficients = [Decimal.from_float(float(value)) / unit for value in a]
        difficulties = [Decimal.from_float(float(value)) * unit for value in b]
        result = []
        for row in data:
            if not na_rm and np.isnan(row).any():
                result.append(np.nan)
                continue
            observations = [
                (Decimal.from_float(float(response)), coefficient, difficulty)
                for response, coefficient, difficulty in zip(
                    row, coefficients, difficulties, strict=True
                )
                if not np.isnan(response)
            ]
            if not observations:
                result.append(np.nan)
                continue
            if all(response == 1 for response, _, _ in observations):
                result.append(3.0)
                continue
            if all(response == 0 for response, _, _ in observations):
                result.append(-3.0)
                continue
            lower, upper = Decimal(-32), Decimal(32)
            for _ in range(100):
                position = (lower + upper) / 2
                gradient = sum(
                    coefficient
                    * (response - 1 / (1 + (coefficient * (difficulty - position)).exp()))
                    for response, coefficient, difficulty in observations
                )
                if gradient > 0:
                    lower = position
                else:
                    upper = position
            result.append(float((lower + upper) / (2 * unit)))
        return np.asarray(result)


@pytest.mark.parametrize("scale", [1e155, 1e200, 1e308, np.finfo(float).max])
@pytest.mark.parametrize("na_rm", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_estimated_steep_models_match_independent_decimal_roots(
    scale: float, na_rm: bool, negative: bool
) -> None:
    data = np.asarray(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [1, np.nan, 1, 0],
            [1] * 4,
            [0] * 4,
            [np.nan] * 4,
        ]
    )
    coefficients = np.asarray([-0.5, 1, 0.75, -0.25] if negative else [0.5, 1, 0.75, 0.25])
    a = coefficients * scale
    b = np.asarray([-0.3, 0.1, 0.4, -0.2]) / scale
    expected_theta = _decimal_theta(data, a, b, scale, na_rm=na_rm)
    expected_scores = _decimal_scores(data, b, a, expected_theta, na_rm=na_rm)
    originals = [value.copy() for value in (data, a, b)]
    for value in (data, a, b):
        value.flags.writeable = False
    with np.errstate(all="raise"), patch("ier.lz._LZ_BATCH_ELEMENTS", 5):
        theta = _estimate_theta(data, a, b, na_rm=na_rm)
        scores, flags = lz_flag(data, difficulty=b, discrimination=a, na_rm=na_rm, threshold=-1.5)
    # Compare ability in model units so enormous slopes cannot hide a poor fit
    # behind an absolute tolerance on tiny ability estimates.
    np.testing.assert_allclose(
        theta[:4] * scale, expected_theta[:4] * scale, rtol=3e-12, atol=3e-12
    )
    np.testing.assert_array_equal(theta[4:], expected_theta[4:])
    np.testing.assert_allclose(scores, expected_scores, rtol=3e-12, atol=3e-12)
    np.testing.assert_array_equal(flags, expected_scores < -1.5)
    for value, original in zip((data, a, b), originals, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_common_steep_discrimination_has_analytic_binomial_ability(
    layout: str, na_rm: bool
) -> None:
    scale = np.finfo(float).max
    data = np.asarray([[1, 1, 1, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, np.nan, 1, 0]])
    if layout == "strided":
        backing = np.zeros((8, 8))
        backing[::2, ::2] = data
        data = backing[::2, ::2]
    else:
        data = np.array(data, order=layout)
    original = data.copy()
    data.flags.writeable = False
    expected = [math.log(3), -math.log(3), 0, math.log(2) if na_rm else np.nan]
    with np.errstate(all="raise"), patch("ier.lz._LZ_BATCH_ELEMENTS", 5):
        abilities = _estimate_theta(data, np.full(4, scale), np.zeros(4), na_rm=na_rm)
    np.testing.assert_allclose(abilities * scale, expected, rtol=3e-15, atol=1e-15)
    np.testing.assert_array_equal(data, original)


def test_saturated_and_missing_huge_items_do_not_hide_moderate_information() -> None:
    data = np.asarray([[1, 1, 1, 0], [np.nan, 1, 1, 0], [1, 0, 0, 1]])
    a = np.asarray([np.finfo(float).max, 1, 1, 1])
    b = np.asarray([-1, 0, 0, 0])
    with np.errstate(all="raise"):
        abilities = _estimate_theta(data, a, b)
        scores, flags = lz_flag(data, difficulty=b, discrimination=a)
    expected_theta = np.asarray([math.log(2), math.log(2), -math.log(2)])
    expected_scores = _decimal_scores(data, b, a, expected_theta, na_rm=True)
    np.testing.assert_allclose(abilities, expected_theta, rtol=3e-12, atol=1e-12)
    np.testing.assert_allclose(scores, expected_scores, rtol=3e-12, atol=1e-12)
    np.testing.assert_array_equal(flags, expected_scores < -1.96)


@pytest.mark.parametrize("observed_huge", [False, True])
def test_huge_item_keeps_varied_moderate_item_calibration(observed_huge: bool) -> None:
    moderate = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    data = np.column_stack((np.ones(3) if observed_huge else np.full(3, np.nan), moderate))
    a = np.asarray([np.finfo(float).max, 0.5, 1.2, 0.8, 1.5])
    b = np.asarray([-2.0, 0.3, -0.8, 1.2, 0.0])
    expected_theta = _decimal_theta(moderate, a[1:], b[1:], 1.0, na_rm=True)
    assert np.all(expected_theta > b[0])
    expected_scores = _decimal_scores(data, b, a, expected_theta, na_rm=True)
    with np.errstate(all="raise"), patch("ier.lz._LZ_BATCH_ELEMENTS", 7):
        theta = _estimate_theta(data, a, b)
        scores, flags = lz_flag(data, difficulty=b, discrimination=a)
    np.testing.assert_allclose(theta, expected_theta, rtol=3e-12, atol=1e-12)
    np.testing.assert_allclose(scores, expected_scores, rtol=3e-12, atol=1e-12)
    np.testing.assert_array_equal(flags, expected_scores < -1.96)


@pytest.mark.parametrize("missing_parameter", ["difficulty", "discrimination"])
@pytest.mark.parametrize("na_rm", [False, True])
def test_estimated_scaled_models_preserve_unavailable_parameters(
    missing_parameter: str, na_rm: bool
) -> None:
    data = np.asarray([[1, 0, np.nan], [1, 0, 1], [np.nan] * 3])
    a = np.full(3, np.finfo(float).max)
    b = np.zeros(3)
    (b if missing_parameter == "difficulty" else a)[2] = np.nan
    expected = _decimal_scores(data, b, a, np.zeros(3), na_rm=na_rm)
    with np.errstate(all="raise"):
        scores, flags = lz_flag(data, difficulty=b, discrimination=a, na_rm=na_rm)
    np.testing.assert_array_equal(scores, expected)
    assert not flags.any()


@pytest.mark.parametrize("slope", [1e-323, 1e-300, 1e-155])
def test_flat_models_retain_initial_ability_and_missing_policy(slope: float) -> None:
    data = np.asarray([[1, 1, 1, 0], [1, 0, 1, 0], [1, np.nan, 1, 0], [np.nan] * 4])
    expected = [math.log(3), 0, math.log(2), np.nan]
    with np.errstate(all="raise"):
        theta = _estimate_theta(data, np.full(4, slope), np.zeros(4))
        scores, flags = lz_flag(data, difficulty=np.zeros(4), discrimination=np.full(4, slope))
    np.testing.assert_allclose(theta, expected, rtol=3e-15, atol=1e-15)
    assert np.isfinite(scores[:3]).all()
    assert np.isnan(scores[3])
    assert not flags.any()


@pytest.mark.parametrize("slope", [3.0, np.finfo(float).max])
def test_saturated_predictions_retain_ability_bounds(slope: float) -> None:
    data = np.asarray([[1, 1, 1, 0], [0, 0, 0, 1], [1] * 4, [0] * 4])
    limit = np.finfo(float).max
    b = np.asarray([-limit, limit, -limit, limit])
    with np.errstate(all="raise"):
        theta = _estimate_theta(data, np.full(4, slope), b)
        scores, flags = lz_flag(data, difficulty=b, discrimination=np.full(4, slope))
    np.testing.assert_allclose(theta, [4, -4, 3, -3], rtol=0, atol=1e-12)
    expected = _decimal_scores(data, b, np.full(4, slope), np.asarray([4, -4, 3, -3]), na_rm=True)
    np.testing.assert_allclose(scores, expected, rtol=3e-14, atol=1e-14)
    np.testing.assert_array_equal(flags, expected < -1.96)
