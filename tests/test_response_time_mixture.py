"""Independent numerical checks for Gaussian response-time mixture fitting."""

import math
from decimal import Decimal, localcontext
from unittest.mock import Mock, patch

import numpy as np
import pytest

from ier import response_time_mixture
from ier.response_time import _em_gaussian_mixture, _mixture_expectation, _weighted_deviation


def _decimal_expectation(
    data: np.ndarray, weights: np.ndarray, means: np.ndarray, deviations: np.ndarray
) -> tuple[np.ndarray, float]:
    with localcontext() as context:
        context.prec = 800
        half_log_two_pi = Decimal.from_float(2 * math.pi).ln() / 2
        probabilities = []
        likelihood = Decimal(0)
        for value in data:
            joints = []
            for weight, mean, deviation in zip(weights, means, deviations, strict=True):
                if weight == 0:
                    joints.append(Decimal("-Infinity"))
                    continue
                sd = Decimal.from_float(float(deviation))
                distance = (Decimal.from_float(float(value)) - Decimal.from_float(float(mean))) / sd
                joints.append(
                    Decimal.from_float(float(weight)).ln()
                    - sd.ln()
                    - half_log_two_pi
                    - distance * distance / 2
                )
            maximum = max(joints)
            densities = [
                (value - maximum).exp() if value - maximum > -1000 else Decimal(0)
                for value in joints
            ]
            total = sum(densities)
            probabilities.append([float(value / total) for value in densities])
            likelihood += maximum + total.ln()
        return np.asarray(probabilities), float(likelihood)


@pytest.mark.parametrize(
    ("observations", "centers", "scales", "masses"),
    [
        ([-1e308, 0, 1e308], [-1e308, 1e308], [1e307, 1e307], [0.25, 0.75]),
        ([0, 1e-313, 1], [0, 1e-313], [1e-313, 1e-313], [0.25, 0.75]),
        ([1e308], [0, 1], [1e-5, 1e-5], [0.5, 0.5]),
        ([1e308], [0, 0], [1e-5, 1e-5], [0.25, 0.75]),
        ([1e308], [0, 1], [1e-5, 1e-5], [0, 1]),
        ([1e154] * 10, [0, 1], [1, 1], [0.5, 0.5]),
        ([1e6], [0, 1e-8], [1, 1], [0.5, 0.5]),
        ([4e-299], [0, 0], [1e-300, 1e50], [0.5, 0.5]),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_extreme_expectations_match_decimal(
    observations: list[float],
    centers: list[float],
    scales: list[float],
    masses: list[float],
    order: str,
) -> None:
    data, means, deviations, weights = map(np.asarray, (observations, centers, scales, masses))
    expected, expected_likelihood = _decimal_expectation(data, weights, means, deviations)
    responsibilities = np.empty(expected.shape, order=order)
    likelihood = _mixture_expectation(
        data, weights, means, deviations, responsibilities, np.empty(len(data))
    )
    np.testing.assert_allclose(responsibilities, expected, rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1, rtol=0, atol=3e-16)
    np.testing.assert_allclose(likelihood, expected_likelihood, rtol=3e-14, atol=3e-14)


@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1, 1e200, 1e300])
def test_weighted_deviations_match_decimal(scale: float) -> None:
    data = np.array([1, 2, 3, 4], dtype=float) * scale
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    mean = 2 * scale
    with localcontext() as context:
        context.prec = 800
        variance = sum(
            Decimal.from_float(float(weight))
            * (Decimal.from_float(float(value)) - Decimal.from_float(mean)) ** 2
            for weight, value in zip(weights, data, strict=True)
        )
        expected = float(variance.sqrt())
    actual = _weighted_deviation(data, mean, weights, 1.0, np.empty(len(data)))
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)


def test_zero_mass_outliers_do_not_destroy_tiny_weighted_variation() -> None:
    data = np.array([1e300, 1e-200, 2e-200, 3e-200])
    weights = np.array([0, 0.25, 0.5, 0.25])
    actual = _weighted_deviation(data, 2e-200, weights, 1.0, np.empty(len(data)))
    np.testing.assert_allclose(actual, 1e-200 / math.sqrt(2), rtol=3e-15, atol=0)


@pytest.mark.parametrize("value", [1e-300, 1.1, 1e300, 1e308])
@pytest.mark.parametrize("components", [2, 3])
@pytest.mark.parametrize("log_transform", [False, True])
def test_constant_timings_have_equal_component_probabilities(
    value: float, components: int, log_transform: bool
) -> None:
    data = np.full((9, 7), value)
    data[1, 0] = np.nan
    data[2] = np.nan
    result = response_time_mixture(
        data, n_components=components, log_transform=log_transform, random_seed=42
    )
    expected = np.full(9, 1 / components)
    expected[2] = np.nan
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("scale", [1e-300, 1e-100, 1e-10])
def test_variance_floor_remains_in_original_units(scale: float) -> None:
    data = np.arange(1, 31, dtype=float)[:, None] * scale
    result = response_time_mixture(data, log_transform=False, random_seed=42)
    np.testing.assert_allclose(result, 0.5, rtol=0, atol=1e-7)


@pytest.mark.parametrize("scale", [1e200, 1e300, 1e307])
@pytest.mark.parametrize("order", ["C", "F"])
def test_large_finite_timings_retain_fast_group_and_availability(scale: float, order: str) -> None:
    times = np.repeat(np.array([1, 1.1, 1.2, 5, 5.1, 5.2, 0, -1])[:, None], 3, axis=1) * scale
    times = np.array(np.vstack((times, [np.nan] * 3, [np.inf] * 3)), order=order)
    times[1, 1:] = np.nan
    original = times.copy()
    times.flags.writeable = False
    result = response_time_mixture(times, log_transform=False, random_seed=42)
    assert np.all(result[:3] > 0.99)
    assert np.all(result[3:6] < 0.01)
    assert np.isnan(result[6:]).all()
    np.testing.assert_array_equal(times, original)


@pytest.mark.parametrize("offset", [1.1, 1e100, 1e300])
def test_nearly_constant_fit_preserves_translation_and_scale(offset: float) -> None:
    coefficients = np.array([1, 2, 3, 8, 9, 10], dtype=float)
    data = offset + coefficients * np.spacing(offset)
    # Removing jitter isolates the affine invariance of the fitted distributions.
    rng = Mock(spec=np.random.Generator)
    rng.normal.return_value = np.zeros(2)
    expected = _em_gaussian_mixture(coefficients, 2, rng)
    if offset == 1.1:
        # The absolute variance floor overwhelms sub-ulp-scale separation here.
        expected = np.full(6, 0.5)
    with patch("ier.response_time.np.random.default_rng", return_value=rng):
        result = response_time_mixture(data[:, None], log_transform=False)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_abandoned_component_cannot_be_selected_as_fast() -> None:
    rng = Mock(spec=np.random.Generator)
    rng.normal.return_value = np.array([-1e6, 0])
    probabilities = _em_gaussian_mixture(np.arange(1.0, 7), 2, rng)
    np.testing.assert_array_equal(probabilities, np.ones(6))
