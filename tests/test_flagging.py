"""Tests for shared threshold and percentile validation."""

import unittest
from typing import Any, cast

import numpy as np

from ier import (
    acquiescence_flag,
    composite_flag,
    mad_flag,
    markov_flag,
    response_time_flag,
)
from ier._flagging import (
    resolve_threshold,
    threshold_flags,
    validate_percentile,
    validate_threshold,
)


class TestFlaggingValidation(unittest.TestCase):
    def test_percentile_validation(self) -> None:
        self.assertEqual(validate_percentile(95), 95.0)
        self.assertEqual(validate_percentile(cast("Any", "5")), 5.0)
        for value in [False, -0.1, 100.1, np.nan, np.inf, "bad", None]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "percentile"):
                validate_percentile(cast("Any", value))

    def test_threshold_validation(self) -> None:
        self.assertIsNone(validate_threshold(None))
        self.assertEqual(validate_threshold(1), 1.0)
        self.assertEqual(validate_threshold(cast("Any", "1.5")), 1.5)
        for value in [False, np.nan, np.inf, -np.inf, "bad"]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "threshold"):
                validate_threshold(cast("Any", value))

    def test_explicit_threshold_still_requires_a_valid_percentile(self) -> None:
        with self.assertRaisesRegex(ValueError, "percentile"):
            resolve_threshold(np.array([1.0, 2.0]), threshold=1.0, percentile=101.0)

    def test_all_missing_scores_produce_no_flags(self) -> None:
        scores = np.array([np.nan, np.nan])
        self.assertEqual(resolve_threshold(scores, threshold=None, percentile=95.0), 0.0)
        np.testing.assert_array_equal(
            threshold_flags(
                scores,
                threshold=None,
                percentile=95.0,
                direction="high",
            ),
            [False, False],
        )

    def test_public_percentile_flaggers_share_validation(self) -> None:
        data = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 3.0, 3.0, 3.0]])
        calls = {
            "acquiescence": lambda: acquiescence_flag(
                data, scale_min=1, scale_max=5, percentile=101.0
            ),
            "composite": lambda: composite_flag(data, indices=["irv"], percentile=101.0),
            "mad": lambda: mad_flag(
                data, item_pairs=[(0, 1), (2, 3)], scale_max=5, percentile=101.0
            ),
            "markov": lambda: markov_flag(data, percentile=101.0),
            "response_time": lambda: response_time_flag(data, cutoff_percentile=101.0),
        }
        for name, call in calls.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "percentile"):
                call()

    def test_public_flagger_rejects_non_finite_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold"):
            response_time_flag([[1.0, 2.0], [2.0, 3.0]], threshold=np.nan)
