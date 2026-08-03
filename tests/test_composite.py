"""Unit tests for composite IER scorers."""

import unittest
import warnings
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from ier import IndexOptions
from ier.composite import (
    _combine_scores,
    composite,
    composite_flag,
    composite_probability,
    composite_summary,
)
from ier.irv import irv


class TestComposite(unittest.TestCase):
    """Tests for composite IER index functions."""

    def test_basic_functionality(self) -> None:
        """Test basic composite calculation."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data)
        self.assertEqual(len(result), 3)

    def test_straightliner_highest_score(self) -> None:
        """Test that straightliners get highest composite score."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data)
        self.assertEqual(np.argmax(result), 1)

    def test_specific_indices(self) -> None:
        """Test composite with specific indices."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        result = composite(data, indices=["irv", "longstring"])
        self.assertEqual(len(result), 3)

    def test_irv_sections_flow_through_raw_composites(self) -> None:
        data = np.array(
            [[1, 2, 5, 5, 4, 1], [3, 3, 3, 1, 3, 5], [5, 4, 3, 2, 1, 2]],
            dtype=float,
        )
        expected = -irv(data, split=True, split_points=[0, 2, 6])

        result = composite(
            data,
            indices=["irv"],
            options=IndexOptions(irv_split_points=[0, 2, 6]),
            standardize=False,
        )

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-15)

    def test_sum_method(self) -> None:
        """Test composite with sum method."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = composite(data, method="sum")
        self.assertEqual(len(result), 2)

    def test_max_method(self) -> None:
        """Test composite with max method."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = composite(data, method="max")
        self.assertEqual(len(result), 2)

    def test_no_standardize(self) -> None:
        """Test composite without standardization."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = composite(data, standardize=False)
        self.assertEqual(len(result), 2)

    def test_parallel_workers_propagate_across_composite_apis(self) -> None:
        data = np.array(
            [
                [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
                [2, 4, 1, 5, 3, 2, 5, 1, 4, 3],
            ],
            dtype=float,
        )
        indices = ["irv", "longstring", "person_total"]

        sequential = composite(data, indices=indices)
        parallel = composite(data, indices=indices, workers=3)
        np.testing.assert_allclose(parallel, sequential, equal_nan=True)

        sequential_scores, sequential_flags = composite_flag(data, indices=indices)
        parallel_scores, parallel_flags = composite_flag(data, indices=indices, workers=3)
        np.testing.assert_allclose(parallel_scores, sequential_scores, equal_nan=True)
        np.testing.assert_array_equal(parallel_flags, sequential_flags)

        sequential_summary = composite_summary(data, indices=indices)
        parallel_summary = composite_summary(data, indices=indices, workers=3)
        self.assertEqual(parallel_summary["indices_used"], indices)
        self.assertEqual(parallel_summary["errors"], sequential_summary["errors"])
        np.testing.assert_allclose(
            parallel_summary["composite"],
            sequential_summary["composite"],
            equal_nan=True,
        )
        for name in indices:
            np.testing.assert_array_equal(
                parallel_summary["indices"][name],
                sequential_summary["indices"][name],
            )

        sequential_probability = composite_probability(data, indices=indices)
        parallel_probability = composite_probability(data, indices=indices, workers=3)
        np.testing.assert_allclose(parallel_probability, sequential_probability, equal_nan=True)

    def test_with_evenodd(self) -> None:
        """Test composite with evenodd index."""
        data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
        result = composite(
            data,
            indices=["irv", "evenodd"],
            options=IndexOptions(evenodd_factors=[5, 5]),
        )
        self.assertEqual(len(result), 2)

    def test_flag_function(self) -> None:
        """Test composite flagging."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        scores, flags = composite_flag(data, threshold=1.0)
        self.assertEqual(len(flags), 3)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )
        self.assertTrue(flags[1])

    def test_flag_with_percentile(self) -> None:
        """Test composite flagging with percentile."""
        data = [
            [1, 2, 3, 4, 5],
            [3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1],
            [2, 3, 4, 5, 1],
        ]
        scores, flags = composite_flag(data, percentile=75.0)
        self.assertEqual(len(flags), 4)

    def test_summary_function(self) -> None:
        """Test composite summary."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3], [5, 4, 3, 2, 1]]
        summary = composite_summary(data)
        self.assertIn("composite", summary)
        self.assertIn("indices", summary)
        self.assertIn("indices_used", summary)
        self.assertIn("mean", summary)
        self.assertIn("std", summary)
        self.assertIsNone(summary["min_valid_indices"])
        self.assertEqual(summary["valid_index_counts"].shape, (3,))

    def test_invalid_index_raises(self) -> None:
        """Test that invalid index raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, indices=["invalid_index"])

    def test_evenodd_without_factors_soft_fails(self) -> None:
        """Missing evenodd config soft-fails like screen(); no indices succeed."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaisesRegex(ValueError, "no valid indices"):
            composite(data, indices=["evenodd"])
        _, diagnostics = composite(data, indices=["irv", "evenodd"], return_diagnostics=True)
        self.assertIn("evenodd", diagnostics)

    def test_strict_mode_propagates_across_composite_apis(self) -> None:
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        calls = {
            "composite": lambda: composite(data, indices=["irv", "mad"], strict=True),
            "flag": lambda: composite_flag(data, indices=["irv", "mad"], strict=True),
            "summary": lambda: composite_summary(data, indices=["irv", "mad"], strict=True),
            "probability": lambda: composite_probability(data, indices=["irv", "mad"], strict=True),
        }
        for name, call in calls.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError,
                    "index 'mad' failed: mad_positive_items",
                ),
            ):
                call()

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, method=cast("Any", "invalid"))

    def test_return_diagnostics(self) -> None:
        """Test composite can return index diagnostics."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        scores, diagnostics = composite(data, indices=["irv", "mad"], return_diagnostics=True)
        self.assertEqual(len(scores), 2)
        self.assertIn("mad", diagnostics)

    def test_flag_return_diagnostics(self) -> None:
        """Test composite_flag returns diagnostics when requested."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        scores, flags, diagnostics = composite_flag(
            data, indices=["irv", "mad"], return_diagnostics=True
        )
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(flags), 2)
        self.assertIn("mad", diagnostics)

    def test_no_valid_indices_raises_with_failures(self) -> None:
        """Test composite reports collected handler failures when no index succeeds."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        with self.assertRaisesRegex(ValueError, "no valid indices"):
            composite(data, indices=["mad"])

    def test_constant_index_standardizes_to_zero(self) -> None:
        """Test zero-variance standardized index scores become zero."""
        data = [[1, 1, 2, 2], [3, 3, 4, 4]]
        result = composite(data, indices=["longstring"], standardize=True)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_noninteger_longstring_not_truncated(self) -> None:
        """Test longstring index does not truncate non-integer values."""
        data = [[1.1, 1.9, 1.1, 1.9]]
        result = composite(data, indices=["longstring"], standardize=False)
        np.testing.assert_array_equal(result, np.array([1.0]))

    def test_score_combination_matches_matrix_reductions_without_stacking(self) -> None:
        scores = {
            "first": np.array([1.0, np.nan, 3.0, np.nan, 5.0]),
            "constant": np.array([2.0, 2.0, np.nan, np.nan, 2.0]),
            "third": np.array([np.nan, 4.0, 6.0, np.nan, 8.0]),
        }

        for standardize in (False, True):
            prepared: dict[str, np.ndarray] = {}
            for name, values in scores.items():
                if not standardize:
                    prepared[name] = values
                    continue
                valid = ~np.isnan(values)
                if np.sum(valid) <= 1:
                    prepared[name] = values
                    continue
                mean = np.nanmean(values)
                std = np.nanstd(values)
                if std > 0:
                    prepared[name] = (values - mean) / std
                else:
                    prepared[name] = np.zeros_like(values)
                    prepared[name][np.isnan(values)] = np.nan
            matrix = np.column_stack(list(prepared.values()))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                expected = {
                    "mean": np.nanmean(matrix, axis=1),
                    "sum": np.nansum(matrix, axis=1),
                    "max": np.nanmax(matrix, axis=1),
                }

            for method in ("mean", "sum", "max"):
                with (
                    self.subTest(method=method, standardize=standardize),
                    patch(
                        "ier.composite.np.column_stack",
                        side_effect=AssertionError("score matrix was constructed"),
                    ),
                ):
                    actual = _combine_scores(
                        scores,
                        {},
                        cast("Any", method),
                        standardize,
                    )
                np.testing.assert_allclose(
                    actual,
                    expected[method],
                    rtol=1e-14,
                    atol=1e-14,
                    equal_nan=True,
                )

    def test_weighted_reductions_renormalize_available_scores(self) -> None:
        scores = {
            "first": np.array([1.0, np.nan, 3.0, np.nan]),
            "second": np.array([5.0, 7.0, np.nan, np.nan]),
        }
        weights = {"first": 2.0, "second": 1.0}

        np.testing.assert_allclose(
            _combine_scores(scores, {}, "mean", False, weights),
            np.array([7.0 / 3.0, 7.0, 3.0, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _combine_scores(scores, {}, "mean", False, {"first": 2.0}),
            _combine_scores(scores, {}, "mean", False, weights),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _combine_scores(scores, {}, "sum", False, weights),
            np.array([7.0, 7.0, 6.0, 0.0]),
        )
        np.testing.assert_allclose(
            _combine_scores(scores, {}, "max", False, weights),
            np.array([5.0, 7.0, 6.0, np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _combine_scores(scores, {}, "mean", False, weights),
            _combine_scores(
                scores,
                {},
                "mean",
                False,
                {name: weight * 10 for name, weight in weights.items()},
            ),
            equal_nan=True,
        )

    def test_minimum_valid_indices_filter_every_reduction_method(self) -> None:
        scores = {
            "first": np.array([1.0, np.nan, 3.0, np.nan]),
            "second": np.array([5.0, 7.0, np.nan, np.nan]),
        }
        expected = {
            "mean": np.array([3.0, np.nan, np.nan, np.nan]),
            "sum": np.array([6.0, np.nan, np.nan, np.nan]),
            "max": np.array([5.0, np.nan, np.nan, np.nan]),
        }

        for method in ("mean", "sum", "max"):
            with self.subTest(method=method):
                counts = np.empty(4, dtype=np.int_)
                actual = _combine_scores(
                    scores,
                    {},
                    cast("Any", method),
                    False,
                    min_valid_indices=2,
                    valid_counts_out=counts,
                )
                np.testing.assert_allclose(actual, expected[method], equal_nan=True)
                np.testing.assert_array_equal(counts, [2, 1, 1, 0])

        default_sum = _combine_scores(scores, {}, "sum", False)
        filtered_sum = _combine_scores(scores, {}, "sum", False, min_valid_indices=1)
        self.assertEqual(default_sum[-1], 0.0)
        self.assertTrue(np.isnan(filtered_sum[-1]))

        invalid_count_outputs = [
            np.empty((4, 1), dtype=np.int_),
            np.empty(4, dtype=float),
        ]
        for invalid in invalid_count_outputs:
            with (
                self.subTest(shape=invalid.shape, dtype=invalid.dtype),
                self.assertRaisesRegex(
                    ValueError,
                    "respondent-length integer array",
                ),
            ):
                _combine_scores(scores, {}, "mean", False, valid_counts_out=invalid)

    def test_weighted_completeness_uses_component_count_not_weight_sum(self) -> None:
        scores = {
            "first": np.array([1.0, 2.0, np.nan]),
            "second": np.array([5.0, np.nan, np.nan]),
        }
        weights = {"first": 100.0, "second": 0.5}

        result = _combine_scores(
            scores,
            {},
            "mean",
            False,
            weights,
            min_valid_indices=2,
        )

        np.testing.assert_allclose(result, [(100.0 + 2.5) / 100.5, np.nan, np.nan], equal_nan=True)

    @patch("ier.composite.score_registered_indices")
    def test_completeness_propagates_and_summary_reports_counts(self, score_mock: Any) -> None:
        def score_indices(
            *_args: Any, **kwargs: Any
        ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
            irv = np.array([1.0, np.nan, 3.0, np.nan])
            if not kwargs.get("apply_composite_direction", False):
                irv *= -1.0
            return {
                "irv": irv,
                "longstring": np.array([5.0, 7.0, np.nan, np.nan]),
            }, {}

        score_mock.side_effect = score_indices
        data = np.ones((4, 3), dtype=float)
        kwargs = {
            "indices": ["irv", "longstring"],
            "standardize": False,
            "min_valid_indices": 2,
        }

        scores = composite(data, **cast("Any", kwargs))
        flag_scores, flags = composite_flag(data, threshold=2.0, **cast("Any", kwargs))
        summary = composite_summary(data, **cast("Any", kwargs))
        probabilities = composite_probability(
            data,
            indices=["irv", "longstring"],
            min_valid_indices=2,
        )

        expected = np.array([3.0, np.nan, np.nan, np.nan])
        np.testing.assert_allclose(scores, expected, equal_nan=True)
        np.testing.assert_allclose(flag_scores, expected, equal_nan=True)
        np.testing.assert_array_equal(flags, [True, False, False, False])
        np.testing.assert_allclose(summary["composite"], expected, equal_nan=True)
        np.testing.assert_array_equal(summary["valid_index_counts"], [2, 1, 1, 0])
        self.assertEqual(summary["min_valid_indices"], 2)
        self.assertEqual(summary["n_valid"], 1)
        self.assertTrue(np.isfinite(probabilities[0]))
        self.assertTrue(np.isnan(probabilities[1:]).all())

    @patch("ier.composite.score_registered_indices")
    def test_soft_failed_indices_do_not_relax_requested_minimum(
        self,
        score_mock: Any,
    ) -> None:
        score_mock.return_value = (
            {"irv": np.array([1.0, 2.0])},
            {"longstring": "unavailable"},
        )

        scores, diagnostics = composite(
            np.ones((2, 3)),
            indices=["irv", "longstring"],
            standardize=False,
            min_valid_indices=2,
            return_diagnostics=True,
        )

        self.assertTrue(np.isnan(scores).all())
        self.assertEqual(diagnostics, {"longstring": "unavailable"})

    def test_minimum_valid_indices_validation(self) -> None:
        data = [[1, 2, 3], [3, 2, 1]]
        cases = [
            (0, "positive integer"),
            (-1, "positive integer"),
            (True, "positive integer"),
            (cast("Any", 1.5), "positive integer"),
            (3, "cannot exceed"),
        ]

        for minimum, message in cases:
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, message):
                composite(
                    data,
                    indices=["irv", "longstring"],
                    min_valid_indices=minimum,
                )

    def test_constant_standardization_preserves_missing_values(self) -> None:
        scores = {
            "constant": np.array([2.0, 2.0, np.nan]),
            "varying": np.array([1.0, 3.0, 5.0]),
        }

        actual = _combine_scores(scores, {}, "mean", True)
        expected_varying = (scores["varying"] - 3.0) / np.std(scores["varying"])
        np.testing.assert_allclose(
            actual,
            np.array([expected_varying[0] / 2.0, expected_varying[1] / 2.0, expected_varying[2]]),
        )

    def test_weights_propagate_across_composite_helpers(self) -> None:
        data = np.array(
            [
                [1, 2, 3, 4, 5],
                [3, 3, 3, 3, 3],
                [5, 4, 3, 2, 1],
                [2, 5, 1, 4, 3],
            ],
            dtype=float,
        )
        indices = ["irv", "longstring"]
        weights = {"irv": 3.0}

        scores = composite(data, indices=indices, weights=weights)
        flag_scores, _ = composite_flag(data, indices=indices, weights=weights)
        summary = composite_summary(data, indices=indices, weights=weights)
        probabilities = composite_probability(data, indices=indices, weights=weights)

        np.testing.assert_allclose(flag_scores, scores, equal_nan=True)
        np.testing.assert_allclose(summary["composite"], scores, equal_nan=True)
        np.testing.assert_allclose(probabilities, 1.0 / (1.0 + np.exp(-scores)))
        self.assertEqual(summary["weights"], {"irv": 3.0, "longstring": 1.0})

    def test_weight_validation_rejects_unsafe_or_unselected_values(self) -> None:
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        cases = [
            ({"irv": 0.0}, "positive finite"),
            ({"irv": -1.0}, "positive finite"),
            ({"irv": np.nan}, "positive finite"),
            ({"irv": np.inf}, "positive finite"),
            ({"irv": True}, "positive finite"),
            ({"irv": cast("Any", "invalid")}, "positive finite"),
            ({"mahad": 2.0}, "not selected"),
        ]

        for weights, message in cases:
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, message):
                composite(
                    data,
                    indices=["irv", "longstring"],
                    weights=weights,
                )


class TestCompositeProbability(unittest.TestCase):
    """Tests for composite_probability function."""

    def test_basic_functionality(self) -> None:
        """Test basic probability computation."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite_probability(data)
        self.assertEqual(len(result), 3)

    def test_range_0_1(self) -> None:
        """Test that probabilities are in [0, 1]."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite_probability(data)
        valid = result[~np.isnan(result)]
        self.assertTrue(np.all(valid >= 0.0))
        self.assertTrue(np.all(valid <= 1.0))

    def test_high_composite_high_probability(self) -> None:
        """Test that high composite scores map to high probabilities."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        probs = composite_probability(data)
        scores = composite(data)
        max_idx = int(np.argmax(scores))
        self.assertEqual(int(np.argmax(probs)), max_idx)

    def test_low_composite_low_probability(self) -> None:
        """Test that low composite scores map to low probabilities."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        probs = composite_probability(data)
        scores = composite(data)
        min_idx = int(np.argmin(scores))
        self.assertEqual(int(np.argmin(probs)), min_idx)

    def test_specific_indices(self) -> None:
        """Test with specific indices."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ]
        result = composite_probability(data, indices=["irv", "longstring"])
        self.assertEqual(len(result), 2)

    def test_extreme_scores_are_transformed_without_overflow(self) -> None:
        scores = np.array([-np.inf, -1000.0, -1.0, 0.0, 1.0, 1000.0, np.inf, np.nan])

        with patch("ier.composite.composite", return_value=scores):
            result = composite_probability([[1.0]])

        expected = np.array(
            [
                0.0,
                0.0,
                1.0 / (1.0 + np.e),
                0.5,
                np.e / (1.0 + np.e),
                1.0,
                1.0,
                np.nan,
            ]
        )
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_return_diagnostics_preserves_probabilities(self) -> None:
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]

        result, diagnostics = composite_probability(
            data,
            indices=["irv", "mad"],
            return_diagnostics=True,
        )

        expected = composite_probability(data, indices=["irv"])
        np.testing.assert_allclose(result, expected)
        self.assertIn("mad", diagnostics)
        self.assertIn("mad_positive_items", diagnostics["mad"])

    @patch("ier.composite.composite", return_value=np.array([0.0]))
    def test_diagnostics_requires_diagnostic_pair(self, _composite_mock: Any) -> None:
        with self.assertRaisesRegex(TypeError, "expected .* diagnostics"):
            composite_probability([[1.0]], return_diagnostics=True)

    @patch(
        "ier.composite.composite",
        return_value=(np.array([0.0]), {"irv": "unavailable"}),
    )
    def test_default_rejects_unexpected_diagnostic_pair(self, _composite_mock: Any) -> None:
        with self.assertRaisesRegex(TypeError, "unexpected diagnostics"):
            composite_probability([[1.0]])


class TestCompositeBestSubset(unittest.TestCase):
    """Tests for composite best_subset method."""

    def test_works(self) -> None:
        """Test that best_subset method works."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data, method="best_subset")
        self.assertEqual(len(result), 3)

    def test_overrides_indices(self) -> None:
        """Test that best_subset overrides user-specified indices."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data, method="best_subset", indices=["mahad"])
        self.assertEqual(len(result), 3)

    def test_with_mad(self) -> None:
        """Test best_subset with MAD item info provided."""
        data = [
            [5, 1, 5, 1, 5, 1, 5, 1, 5, 1],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(
            data,
            method="best_subset",
            options=IndexOptions(
                mad_positive_items=[0, 2, 4, 6, 8],
                mad_negative_items=[1, 3, 5, 7, 9],
                mad_scale_max=5,
            ),
        )
        self.assertEqual(len(result), 3)

    def test_without_mad(self) -> None:
        """Test best_subset without MAD falls back to irv/longstring/lz."""
        data = [
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [5, 4, 3, 2, 1, 2, 3, 4, 5, 4],
        ]
        result = composite(data, method="best_subset")
        summary = composite_summary(data, method="best_subset")
        self.assertNotIn("mad", summary["indices_used"])
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
