"""Tests for shared threshold and percentile validation."""

import unittest
from fractions import Fraction
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from ier import (
    acquiescence_flag,
    composite_flag,
    individual_reliability_flag,
    lz_flag,
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
from ier.composite import _CompositeRun


class TestFlaggingValidation(unittest.TestCase):
    def test_percentile_validation(self) -> None:
        self.assertEqual(validate_percentile(95), 95.0)
        self.assertEqual(validate_percentile(cast("Any", "5")), 5.0)
        self.assertEqual(validate_percentile(cast("Any", np.array(95.0))), 95.0)
        for value in [
            False,
            np.bool_(True),
            -0.1,
            100.1,
            np.nan,
            np.inf,
            "bad",
            None,
            np.array([95.0]),
            Fraction(10**400),
        ]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "percentile"):
                validate_percentile(cast("Any", value))

    def test_threshold_validation(self) -> None:
        self.assertIsNone(validate_threshold(None))
        self.assertEqual(validate_threshold(1), 1.0)
        self.assertEqual(validate_threshold(cast("Any", "1.5")), 1.5)
        self.assertEqual(validate_threshold(cast("Any", np.array(1.5))), 1.5)
        for value in [
            False,
            np.bool_(True),
            np.nan,
            np.inf,
            -np.inf,
            "bad",
            np.array([1.5]),
            Fraction(10**400),
        ]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "threshold"):
                validate_threshold(cast("Any", value))

    def test_fixed_cutoff_helpers_validate_before_scoring(self) -> None:
        calls = [
            (
                "ier.lz.lz",
                lambda value: lz_flag([[1, 1, 0, 0]], threshold=value),
            ),
            (
                "ier.reliability.individual_reliability",
                lambda value: individual_reliability_flag(
                    [[1, 2, 3, 4]], threshold=value, n_splits=1, random_seed=0
                ),
            ),
        ]
        invalid = [np.nan, np.inf, True, np.bool_(True), "bad", None, Fraction(10**400)]
        for target, call in calls:
            for value in invalid:
                with (
                    self.subTest(target=target, value=value),
                    patch(target) as scorer,
                    self.assertRaisesRegex(ValueError, "threshold must be a finite number"),
                ):
                    call(cast("Any", value))
                scorer.assert_not_called()

    def test_fixed_cutoff_helpers_accept_numeric_strings(self) -> None:
        with patch("ier.lz.lz", return_value=np.array([-0.5, 0.5])):
            _, flags = lz_flag([[1, 1, 0, 0]], threshold=cast("Any", "0"))
        np.testing.assert_array_equal(flags, [True, False])

        with patch(
            "ier.reliability.individual_reliability", return_value=np.array([0.2, 0.4, np.nan])
        ):
            flags = individual_reliability_flag([[1, 2, 3, 4]], threshold=cast("Any", "0.3"))
        np.testing.assert_array_equal(flags, [True, False, True])

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
            ("ier.composite._run_composite", lambda: composite_flag([[1.0]], threshold=2.0)),
            ("ier.mad.mad", lambda: mad_flag([[1.0]], threshold=2.0)),
        ]
        low_calls = [
            ("ier.markov.markov", lambda: markov_flag([[1.0]], threshold=2.0)),
            (
                "ier.response_time.response_time",
                lambda: (scores, response_time_flag([[1.0]], threshold=2.0)),
            ),
        ]

        # Composite flagging thresholds the reduced vector of one shared composite run.
        returns = {
            "ier.composite._run_composite": _CompositeRun(
                scores, {}, {}, "mean", True, {}, None, None
            )
        }
        for target, call in high_calls:
            with (
                self.subTest(target=target),
                patch(target, return_value=returns.get(target, scores)),
            ):
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
