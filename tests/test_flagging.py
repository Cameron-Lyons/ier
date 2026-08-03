"""Tests for shared threshold and percentile validation."""

import unittest
from typing import Any, cast
from unittest.mock import patch

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

    def test_fixed_cutoffs_include_equality_and_percentile_cutoffs_exclude_ties(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, np.nan])
        np.testing.assert_array_equal(
            threshold_flags(scores, threshold=2.0, percentile=50.0, direction="high"),
            [False, True, True, False],
        )
        np.testing.assert_array_equal(
            threshold_flags(scores, threshold=2.0, percentile=50.0, direction="low"),
            [True, True, False, False],
        )
        np.testing.assert_array_equal(
            threshold_flags(scores, threshold=None, percentile=50.0, direction="high"),
            [False, False, True, False],
        )
        np.testing.assert_array_equal(
            threshold_flags(scores, threshold=None, percentile=50.0, direction="low"),
            [True, False, False, False],
        )

    def test_percentile_threshold_matches_filtered_reference(self) -> None:
        rng = np.random.default_rng(20260803)
        scores = rng.normal(size=257)
        scores[rng.random(scores.size) < 0.2] = np.nan
        valid_scores = scores[~np.isnan(scores)]

        for percentile in [0.0, 1.0, 50.0, 95.0, 100.0]:
            with self.subTest(percentile=percentile):
                self.assertEqual(
                    resolve_threshold(scores, threshold=None, percentile=percentile),
                    float(np.percentile(valid_scores, percentile)),
                )

    def test_explicit_inclusive_override_is_preserved(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, np.nan])
        np.testing.assert_array_equal(
            threshold_flags(
                scores,
                threshold=2.0,
                percentile=50.0,
                direction="high",
                inclusive=False,
            ),
            [False, False, True, False],
        )

    def test_public_flaggers_follow_shared_cutoff_boundaries(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, np.nan])
        high_calls = [
            ("ier.acquiescence.acquiescence", lambda: acquiescence_flag([[1.0]], threshold=2.0)),
            ("ier.composite.composite", lambda: composite_flag([[1.0]], threshold=2.0)),
            ("ier.mad.mad", lambda: mad_flag([[1.0]], threshold=2.0)),
        ]
        low_calls = [
            ("ier.markov.markov", lambda: markov_flag([[1.0]], threshold=2.0)),
            (
                "ier.response_time.response_time",
                lambda: (scores, response_time_flag([[1.0]], threshold=2.0)),
            ),
        ]

        for target, call in high_calls:
            with self.subTest(target=target), patch(target, return_value=scores):
                _, flags = call()
                np.testing.assert_array_equal(flags, [False, True, True, False])
        for target, call in low_calls:
            with self.subTest(target=target), patch(target, return_value=scores):
                _, flags = call()
                np.testing.assert_array_equal(flags, [True, True, False, False])

    def test_public_percentile_flaggers_exclude_cutoff_ties(self) -> None:
        scores = np.full(4, 2.0)
        calls = [
            ("ier.acquiescence.acquiescence", lambda: acquiescence_flag([[1.0]])),
            ("ier.composite.composite", lambda: composite_flag([[1.0]])),
            ("ier.mad.mad", lambda: mad_flag([[1.0]])),
            ("ier.markov.markov", lambda: markov_flag([[1.0]], percentile=50.0)),
            (
                "ier.response_time.response_time",
                lambda: (scores, response_time_flag([[1.0]], cutoff_percentile=50.0)),
            ),
        ]

        for target, call in calls:
            with self.subTest(target=target), patch(target, return_value=scores):
                _, flags = call()
                self.assertFalse(np.any(flags))

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
