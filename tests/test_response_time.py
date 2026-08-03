"""Unit tests for response-time IER helpers."""

import unittest
from unittest.mock import patch

import numpy as np

from ier.response_time import (
    _em_gaussian_mixture,
    _mixture_expectation,
    response_time,
    response_time_consistency,
    response_time_flag,
    response_time_mixture,
)


class TestResponseTime(unittest.TestCase):
    """Tests for response time functions."""

    def test_basic_metrics(self) -> None:
        """Test basic response time metrics."""
        times = [[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]]
        self.assertAlmostEqual(response_time(times, metric="mean")[0], 3.0)
        self.assertAlmostEqual(response_time(times, metric="median")[0], 3.0)
        self.assertAlmostEqual(response_time(times, metric="min")[0], 2.0)

    def test_invalid_metric_raises(self) -> None:
        """Test that invalid metric raises ValueError."""
        times = [[1.0, 2.0, 3.0]]
        with self.assertRaises(ValueError):
            response_time(times, metric="invalid")

    def test_flag_function(self) -> None:
        """Test response time flagging."""
        times = [[2.0, 3.0, 4.0], [0.1, 0.1, 0.1], [2.5, 2.5, 2.5]]
        flags = response_time_flag(times, threshold=0.5)
        self.assertTrue(flags[1])
        self.assertFalse(flags[0])

    def test_consistency(self) -> None:
        """Test response time consistency (CV)."""
        times = [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]
        cv = response_time_consistency(times)
        self.assertGreater(cv[0], cv[1])

    def test_bounded_metrics_match_numpy_without_mutating_input(self) -> None:
        rng = np.random.default_rng(20260803)
        times = rng.lognormal(mean=1.0, sigma=0.5, size=(53, 11))
        times[rng.random(times.shape) < 0.1] = np.nan
        original = times.copy()

        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 30):
            means = response_time(times, metric="mean")
            medians = response_time(times, metric="median")
            deviations = response_time(times, metric="sd")
            consistency = response_time_consistency(times)

        expected_means = np.nanmean(times, axis=1)
        expected_deviations = np.nanstd(times, axis=1)
        np.testing.assert_allclose(means, expected_means, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            medians,
            np.nanmedian(times, axis=1),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            deviations,
            expected_deviations,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            consistency,
            expected_deviations / expected_means,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(times, original)

    def test_all_missing_rows_are_unavailable_without_warning(self) -> None:
        times = np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]])

        for metric in ["mean", "median", "sd", "min"]:
            with self.subTest(metric=metric):
                result = response_time(times, metric=metric)
                self.assertTrue(np.isnan(result[0]))
                self.assertTrue(np.isfinite(result[1]))

        consistency = response_time_consistency(times)
        self.assertTrue(np.isnan(consistency[0]))
        self.assertTrue(np.isfinite(consistency[1]))


class TestResponseTimeMixture(unittest.TestCase):
    """Tests for response time mixture model."""

    def test_basic_functionality(self) -> None:
        """Test basic mixture model computation."""
        rng = np.random.default_rng(42)
        fast = rng.uniform(0.3, 0.8, size=(10, 5))
        slow = rng.uniform(3.0, 8.0, size=(10, 5))
        times = np.vstack([fast, slow])
        result = response_time_mixture(times, random_seed=42)
        self.assertEqual(len(result), 20)

    def test_fast_high_probability(self) -> None:
        """Test that fast responders get high P(fast)."""
        rng = np.random.default_rng(42)
        fast = rng.uniform(0.1, 0.5, size=(15, 5))
        slow = rng.uniform(5.0, 10.0, size=(15, 5))
        times = np.vstack([fast, slow])
        result = response_time_mixture(times, random_seed=42)
        fast_mean = np.mean(result[:15])
        slow_mean = np.mean(result[15:])
        self.assertGreater(fast_mean, slow_mean)

    def test_slow_low_probability(self) -> None:
        """Test that slow responders get low P(fast)."""
        rng = np.random.default_rng(42)
        fast = rng.uniform(0.1, 0.5, size=(15, 5))
        slow = rng.uniform(5.0, 10.0, size=(15, 5))
        times = np.vstack([fast, slow])
        result = response_time_mixture(times, random_seed=42)
        slow_mean = np.mean(result[15:])
        self.assertLess(slow_mean, 0.5)

    def test_range_0_1(self) -> None:
        """Test that all probabilities are in [0, 1]."""
        rng = np.random.default_rng(42)
        times = rng.uniform(0.5, 10.0, size=(20, 5))
        result = response_time_mixture(times, random_seed=42)
        valid = result[~np.isnan(result)]
        self.assertTrue(np.all(valid >= 0.0))
        self.assertTrue(np.all(valid <= 1.0))

    def test_no_log_transform(self) -> None:
        """Test without log transform."""
        rng = np.random.default_rng(42)
        times = rng.uniform(0.5, 10.0, size=(20, 5))
        result = response_time_mixture(times, log_transform=False, random_seed=42)
        self.assertEqual(len(result), 20)

    def test_reproducibility(self) -> None:
        """Test that same seed gives same results."""
        rng = np.random.default_rng(42)
        times = rng.uniform(0.5, 10.0, size=(20, 5))
        r1 = response_time_mixture(times, random_seed=123)
        r2 = response_time_mixture(times, random_seed=123)
        np.testing.assert_array_almost_equal(r1, r2)

    def test_em_uses_pre_normalization_log_likelihood(self) -> None:
        """Test EM does not converge just because normalized responsibilities sum to one."""
        rng = np.random.default_rng(123)
        data = np.concatenate(
            [
                rng.normal(-1.0, 0.45, 30),
                rng.normal(0.2, 0.55, 35),
                rng.normal(2.0, 0.7, 30),
            ]
        )

        short_run = _em_gaussian_mixture(data, 3, np.random.default_rng(42), max_iter=2, tol=1e-12)
        full_run = _em_gaussian_mixture(data, 3, np.random.default_rng(42), max_iter=100, tol=1e-12)

        self.assertGreater(float(np.max(np.abs(short_run - full_run))), 0.05)

    def test_expectation_matches_probability_space_reference(self) -> None:
        data = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
        weights = np.array([0.4, 0.6])
        means = np.array([-1.0, 2.0])
        variances = np.array([0.5, 1.25])
        responsibilities = np.empty((len(data), len(weights)))
        scratch = np.empty(len(data))

        expected = np.column_stack(
            [
                weight
                * np.exp(-0.5 * (data - mean) ** 2 / variance)
                / np.sqrt(2.0 * np.pi * variance)
                for weight, mean, variance in zip(weights, means, variances, strict=True)
            ]
        )
        row_sums = np.sum(expected, axis=1)
        expected_log_likelihood = float(np.sum(np.log(row_sums)))
        expected /= row_sums[:, None]

        actual_log_likelihood = _mixture_expectation(
            data,
            weights,
            means,
            variances,
            responsibilities,
            scratch,
        )

        self.assertAlmostEqual(actual_log_likelihood, expected_log_likelihood)
        np.testing.assert_allclose(responsibilities, expected, rtol=1e-15, atol=1e-15)

    def test_expectation_uses_log_domain_for_underflowed_rows(self) -> None:
        data = np.array([0.0, 1_000_000.0])
        weights = np.array([0.5, 0.5])
        means = np.array([0.0, 1.0])
        variances = np.array([1e-10, 1e-10])
        responsibilities = np.empty((len(data), len(weights)))
        scratch = np.empty(len(data))

        log_likelihood = _mixture_expectation(
            data,
            weights,
            means,
            variances,
            responsibilities,
            scratch,
        )

        self.assertTrue(np.isfinite(log_likelihood))
        np.testing.assert_array_equal(responsibilities, [[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(np.sum(responsibilities, axis=1), [1.0, 1.0])

    def test_insufficient_data_raises(self) -> None:
        """Test that insufficient data raises ValueError."""
        times = [[1.0, 2.0, 3.0]]
        with self.assertRaises(ValueError):
            response_time_mixture(times, n_components=2)

    def test_with_nan(self) -> None:
        """Test handling of NaN values in response times."""
        rng = np.random.default_rng(42)
        times = rng.uniform(0.5, 10.0, size=(20, 5))
        times[0, :] = np.nan
        result = response_time_mixture(times, random_seed=42)
        self.assertTrue(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[1]))

    def test_non_finite_medians_are_excluded(self) -> None:
        times = np.array(
            [
                [np.inf, np.inf],
                [0.5, 0.6],
                [0.7, 0.8],
                [4.0, 5.0],
                [6.0, 7.0],
            ]
        )

        result = response_time_mixture(times, random_seed=42)

        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isfinite(result[1:]).all())

    def test_bounded_medians_preserve_mixture_probabilities(self) -> None:
        rng = np.random.default_rng(20260803)
        times = np.vstack(
            [
                rng.lognormal(mean=-0.7, sigma=0.2, size=(20, 11)),
                rng.lognormal(mean=1.2, sigma=0.35, size=(40, 11)),
            ]
        )
        times[rng.random(times.shape) < 0.1] = np.nan
        medians = np.nanmedian(times, axis=1)
        expected = _em_gaussian_mixture(
            np.log(medians),
            2,
            np.random.default_rng(42),
        )

        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 30):
            result = response_time_mixture(times, random_seed=42)

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
