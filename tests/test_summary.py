"""Tests for shared summary statistics."""

import unittest
import warnings

import numpy as np

from ier._summary import calculate_summary_stats


class TestCalculateSummaryStats(unittest.TestCase):
    """Verify missing-value behavior shared by public summary helpers."""

    def test_all_unavailable_values_return_nan_without_warning(self) -> None:
        """An unavailable cohort has defined NaN statistics and emits no warning."""
        values = np.full(4, np.nan)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = calculate_summary_stats(values, suffix="_score")

        self.assertEqual(
            set(result),
            {"mean_score", "std_score", "min_score", "max_score", "median_score"},
        )
        self.assertTrue(all(np.isnan(value) for value in result.values()))

    def test_partial_missing_values_match_available_case_statistics(self) -> None:
        """Missing values are compacted once without changing available-case results."""
        values = np.array([np.nan, 1.0, 4.0, 9.0, np.nan])
        original = values.copy()
        valid_values = np.array([1.0, 4.0, 9.0])

        result = calculate_summary_stats(values)

        expected = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "median": float(np.median(valid_values)),
        }
        self.assertEqual(result, expected)
        np.testing.assert_array_equal(values, original)
