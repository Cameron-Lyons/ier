"""Unit tests for specialized IER indices and composite scorers."""

import unittest
from unittest.mock import patch

import numpy as np

from ier.composite import composite, composite_flag, composite_probability, composite_summary
from ier.guttman import guttman, guttman_flag
from ier.infrequency import infrequency, infrequency_flag
from ier.longstring import longstring_pattern
from ier.lz import lz, lz_flag
from ier.mad import mad, mad_flag
from ier.mahad import mahad_qqplot
from ier.markov import _transition_entropy, markov, markov_flag, markov_summary
from ier.onset import onset, onset_flag
from ier.person_total import person_total
from ier.reliability import individual_reliability, individual_reliability_flag
from ier.response_time import (
    _em_gaussian_mixture,
    response_time,
    response_time_consistency,
    response_time_flag,
    response_time_mixture,
)
from ier.semantic import semantic_ant, semantic_syn
from ier.u3_poly import midpoint_responding, response_pattern, u3_poly


class TestPersonTotal(unittest.TestCase):
    """Tests for person-total correlation."""

    def test_basic_functionality(self) -> None:
        """Test basic person-total correlation."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = person_total(data)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.0, places=5)
        self.assertAlmostEqual(result[2], 1.0, places=5)

    def test_reversed_pattern(self) -> None:
        """Test that reversed patterns get lower correlation."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = person_total(data)
        self.assertLess(result[1], result[0])

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[1, 2, np.nan, 4, 5], [1, 2, 3, 4, 5]]
        result = person_total(data, na_rm=True)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))


class TestSemanticSyn(unittest.TestCase):
    """Tests for semantic synonym/antonym functions."""

    def test_basic_synonym(self) -> None:
        """Test basic semantic synonym detection."""
        data = [[1, 1, 5, 5], [1, 2, 5, 4], [3, 3, 3, 3]]
        pairs = [(0, 1), (2, 3)]
        result = semantic_syn(data, pairs)
        self.assertEqual(len(result), 3)

    def test_empty_pairs_raises(self) -> None:
        """Test that empty pairs raises ValueError."""
        data = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            semantic_syn(data, [])

    def test_invalid_indices_raises(self) -> None:
        """Test that invalid indices raise ValueError."""
        data = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            semantic_syn(data, [(0, 10)])

    def test_semantic_ant_wrapper(self) -> None:
        """Test semantic_ant is a proper wrapper."""
        data = [[1, 5, 2, 4], [1, 5, 1, 5]]
        pairs = [(0, 1), (2, 3)]
        result = semantic_ant(data, pairs)
        self.assertEqual(len(result), 2)

    def test_zero_variance_rows_are_handled(self) -> None:
        """Test semantic consistency handles zero-variance respondents deterministically."""
        data = [[3, 3, 3, 3], [3, 3, 3, 3]]
        pairs = [(0, 1), (2, 3)]
        syn = semantic_syn(data, pairs)
        ant = semantic_ant(data, pairs)
        np.testing.assert_array_almost_equal(syn, [1.0, 1.0])
        np.testing.assert_array_almost_equal(ant, [-1.0, -1.0])


class TestGuttman(unittest.TestCase):
    """Tests for Guttman error functions."""

    def test_basic_functionality(self) -> None:
        """Test basic Guttman error calculation."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        result = guttman(data)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(0 <= r <= 1 or np.isnan(r) for r in result))

    def test_normalized_range(self) -> None:
        """Test that normalized Guttman errors are in valid range."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        result = guttman(data, normalize=True)
        for r in result:
            if not np.isnan(r):
                self.assertGreaterEqual(r, 0)
                self.assertLessEqual(r, 1)

    def test_flag_function(self) -> None:
        """Test Guttman flagging."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
        flags = guttman_flag(data, threshold=0.3)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
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


class TestIndividualReliability(unittest.TestCase):
    """Tests for individual reliability functions."""

    def test_basic_functionality(self) -> None:
        """Test basic individual reliability."""
        data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 3, 3], [3, 3, 3, 3, 3, 3]]
        result = individual_reliability(data, n_splits=10, random_seed=42)
        self.assertEqual(len(result), 3)

    def test_too_few_items_raises(self) -> None:
        """Test that too few items raises ValueError."""
        data = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            individual_reliability(data)

    def test_flag_function(self) -> None:
        """Test individual reliability flagging."""
        data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 3, 3]]
        flags = individual_reliability_flag(data, n_splits=10, random_seed=42)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )


class TestU3Poly(unittest.TestCase):
    """Tests for U3 polytomous and response pattern functions."""

    def test_basic_u3(self) -> None:
        """Test basic U3 calculation."""
        data = [[1, 5, 1, 5, 1], [3, 3, 3, 3, 3], [1, 2, 3, 4, 5]]
        result = u3_poly(data, scale_min=1, scale_max=5)
        self.assertEqual(len(result), 3)
        self.assertGreater(result[0], result[1])

    def test_midpoint_responding(self) -> None:
        """Test midpoint responding calculation."""
        data = [[1, 5, 1, 5, 1], [3, 3, 3, 3, 3], [1, 2, 3, 4, 5]]
        result = midpoint_responding(data, scale_min=1, scale_max=5)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1], 1.0)

    def test_response_pattern(self) -> None:
        """Test response pattern returns all indices."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        result = response_pattern(data, scale_min=1, scale_max=5)
        self.assertIn("extreme", result)
        self.assertIn("midpoint", result)
        self.assertIn("acquiescence", result)
        self.assertIn("variability", result)


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

    def test_with_evenodd(self) -> None:
        """Test composite with evenodd index."""
        data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
        result = composite(data, indices=["irv", "evenodd"], evenodd_factors=[5, 5])
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

    def test_invalid_index_raises(self) -> None:
        """Test that invalid index raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, indices=["invalid_index"])

    def test_evenodd_without_factors_raises(self) -> None:
        """Test that evenodd without factors raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, indices=["evenodd"])

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises ValueError."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            composite(data, method="invalid")  # type: ignore[arg-type]

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


class TestMAD(unittest.TestCase):
    """Tests for Mean Absolute Difference functions."""

    def test_basic_functionality(self) -> None:
        """Test basic MAD calculation."""
        data = [
            [5, 1, 5, 1],
            [5, 5, 5, 5],
            [3, 3, 3, 3],
        ]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 4.0)
        self.assertAlmostEqual(result[2], 0.0)

    def test_attentive_responder(self) -> None:
        """Test that attentive responders get low MAD."""
        data = [[5, 1, 4, 2], [4, 2, 5, 1]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        for score in result:
            self.assertLess(score, 1.0)

    def test_careless_responder_high_mad(self) -> None:
        """Test that careless responders ignoring item direction get high MAD."""
        data = [[5, 5, 5, 5], [1, 1, 1, 1]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5)
        for score in result:
            self.assertGreater(score, 3.0)

    def test_item_pairs_input(self) -> None:
        """Test using item_pairs instead of positive/negative items."""
        data = [[5, 1, 4, 2], [3, 3, 3, 3]]
        pairs = [(0, 1), (2, 3)]
        result = mad(data, item_pairs=pairs, scale_max=5)
        self.assertEqual(len(result), 2)

    def test_flag_function(self) -> None:
        """Test MAD flagging."""
        data = [
            [5, 1, 5, 1],
            [5, 5, 5, 5],
            [3, 3, 3, 3],
        ]
        scores, flags = mad_flag(
            data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5, threshold=3.0
        )
        self.assertEqual(len(flags), 3)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )
        self.assertTrue(flags[1])
        self.assertFalse(flags[0])

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[5, 1, np.nan, 1], [5, 5, 5, 5]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3], scale_max=5, na_rm=True)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))

    def test_invalid_no_items_raises(self) -> None:
        """Test that missing item specification raises ValueError."""
        data = [[1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            mad(data, positive_items=[0, 1])

    def test_invalid_index_raises(self) -> None:
        """Test that out-of-bounds index raises ValueError."""
        data = [[1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            mad(data, positive_items=[0, 10], negative_items=[1, 2], scale_max=5)

    def test_both_item_specs_raises(self) -> None:
        """Test that specifying both item_pairs and positive/negative raises."""
        data = [[1, 2, 3, 4]]
        with self.assertRaises(ValueError):
            mad(data, positive_items=[0], negative_items=[1], item_pairs=[(0, 1)], scale_max=5)

    def test_scale_max_inference(self) -> None:
        """Test that scale_max is inferred from data when not provided."""
        data = [[5, 1, 5, 1], [3, 3, 3, 3]]
        result = mad(data, positive_items=[0, 2], negative_items=[1, 3])
        self.assertEqual(len(result), 2)


class TestLz(unittest.TestCase):
    """Tests for standardized log-likelihood (lz) functions."""

    def test_basic_functionality(self) -> None:
        """Test basic lz calculation."""
        data = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        result = lz(data)
        self.assertEqual(len(result), 3)

    def test_normal_pattern_positive_lz(self) -> None:
        """Test that normal response patterns get non-negative lz."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]
        result = lz(data)
        for score in result:
            self.assertGreater(score, -2.0)

    def test_aberrant_pattern_negative_lz(self) -> None:
        """Test that aberrant patterns tend toward negative lz."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        lz_scores = lz(data)
        self.assertGreater(lz_scores[0], lz_scores[3])

    def test_1pl_model(self) -> None:
        """Test lz with 1PL (Rasch) model."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        result = lz(data, model="1pl")
        self.assertEqual(len(result), 2)

    def test_2pl_model(self) -> None:
        """Test lz with 2PL model (default)."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        result = lz(data, model="2pl")
        self.assertEqual(len(result), 2)

    def test_custom_parameters(self) -> None:
        """Test lz with user-specified item parameters."""
        data = [[1, 1, 0, 0], [0, 1, 1, 0]]
        difficulty = [-1.0, -0.5, 0.5, 1.0]
        discrimination = [1.0, 1.0, 1.0, 1.0]
        result = lz(data, difficulty=difficulty, discrimination=discrimination)
        self.assertEqual(len(result), 2)

    def test_custom_theta(self) -> None:
        """Test lz with user-specified theta values."""
        data = [[1, 1, 0, 0], [0, 1, 1, 0]]
        theta = [0.0, 0.5]
        result = lz(data, theta=theta)
        self.assertEqual(len(result), 2)

    def test_flag_function(self) -> None:
        """Test lz flagging."""
        data = [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
        scores, flags = lz_flag(data, threshold=-1.5)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )

    def test_flag_with_custom_threshold(self) -> None:
        """Test lz flagging with custom threshold."""
        data = [[1, 1, 0, 0], [1, 0, 1, 0]]
        scores, flags = lz_flag(data, threshold=0.0)
        self.assertEqual(len(flags), 2)

    def test_with_nan(self) -> None:
        """Test handling of missing values."""
        data = [[1, 1, np.nan, 0], [1, 0, 1, 0]]
        result = lz(data, na_rm=True)
        self.assertEqual(len(result), 2)

    def test_all_correct_responses(self) -> None:
        """Test handling of all correct responses."""
        data = [[1, 1, 1, 1], [0, 0, 0, 0]]
        result = lz(data)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))

    def test_polytomous_dichotomization(self) -> None:
        """Test that polytomous data is dichotomized."""
        data = [[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]]
        result = lz(data)
        self.assertEqual(len(result), 2)

    def test_invalid_model_raises(self) -> None:
        """Test that invalid model raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, model="invalid")

    def test_mismatched_difficulty_raises(self) -> None:
        """Test that mismatched difficulty length raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, difficulty=[1.0, 2.0])

    def test_mismatched_discrimination_raises(self) -> None:
        """Test that mismatched discrimination length raises ValueError."""
        data = [[1, 1, 0, 0]]
        with self.assertRaises(ValueError):
            lz(data, discrimination=[1.0, 2.0])

    def test_mismatched_theta_raises(self) -> None:
        """Test that mismatched theta length raises ValueError."""
        data = [[1, 1, 0, 0], [0, 0, 1, 1]]
        with self.assertRaises(ValueError):
            lz(data, theta=[0.0])


class TestInfrequency(unittest.TestCase):
    """Tests for infrequency/bogus item scoring."""

    def test_basic_functionality(self) -> None:
        """Test basic infrequency counting."""
        data = [[5, 3, 1], [5, 5, 5], [1, 3, 5]]
        result = infrequency(data, item_indices=[0, 2], expected_responses=[5, 1])
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertAlmostEqual(result[2], 2.0)

    def test_all_correct(self) -> None:
        """Test that all-correct responses yield 0."""
        data = [[5, 1], [5, 1]]
        result = infrequency(data, item_indices=[0, 1], expected_responses=[5, 1])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_all_failed(self) -> None:
        """Test that all-failed responses yield max count."""
        data = [[1, 5], [2, 4]]
        result = infrequency(data, item_indices=[0, 1], expected_responses=[5, 1])
        np.testing.assert_array_equal(result, [2.0, 2.0])

    def test_proportion(self) -> None:
        """Test proportion mode."""
        data = [[5, 5], [1, 1]]
        result = infrequency(data, item_indices=[0, 1], expected_responses=[5, 1], proportion=True)
        self.assertAlmostEqual(result[0], 0.5)

    def test_flag_function(self) -> None:
        """Test infrequency flagging."""
        data = [[5, 3, 1], [1, 3, 5]]
        scores, flags = infrequency_flag(data, [0, 2], [5, 1], threshold=2)
        self.assertFalse(flags[0])
        self.assertTrue(flags[1])

    def test_flag_threshold(self) -> None:
        """Test infrequency flag with different thresholds."""
        data = [[5, 5], [1, 1]]
        _, flags_t1 = infrequency_flag(data, [0, 1], [5, 1], threshold=1)
        _, flags_t2 = infrequency_flag(data, [0, 1], [5, 1], threshold=2)
        self.assertTrue(flags_t1[0])
        self.assertFalse(flags_t2[0])

    def test_empty_indices_raises(self) -> None:
        """Test that empty item_indices raises ValueError."""
        data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            infrequency(data, item_indices=[], expected_responses=[])

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            infrequency(data, item_indices=[0, 1], expected_responses=[5])

    def test_out_of_bounds_raises(self) -> None:
        """Test that out-of-bounds index raises ValueError."""
        data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            infrequency(data, item_indices=[10], expected_responses=[5])

    def test_with_nan(self) -> None:
        """Test NaN values are treated as non-matching but not counted as failures."""
        data = [[np.nan, 1], [5, 1]]
        result = infrequency(data, item_indices=[0, 1], expected_responses=[5, 1])
        self.assertEqual(len(result), 2)


class TestLongstringPattern(unittest.TestCase):
    """Tests for longstring_pattern function."""

    def test_basic_functionality(self) -> None:
        """Test basic longstring pattern detection."""
        data = [[1, 2, 1, 2, 1, 2], [1, 2, 3, 4, 5, 6]]
        result = longstring_pattern(data)
        self.assertEqual(len(result), 2)

    def test_repeating_detected(self) -> None:
        """Test that repeating pattern (seesaw) is detected."""
        data = [[1, 2, 1, 2, 1, 2, 1, 2], [1, 3, 5, 2, 4, 1, 3, 5]]
        result = longstring_pattern(data)
        self.assertGreater(result[0], 0)

    def test_no_pattern(self) -> None:
        """Test that non-repeating data returns 0."""
        data = [[1, 2, 3, 4, 5, 6, 7, 8]]
        result = longstring_pattern(data)
        self.assertAlmostEqual(result[0], 0.0)

    def test_seesaw_pattern(self) -> None:
        """Test seesaw (1-2-1-2) detection."""
        data = [[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]
        result = longstring_pattern(data)
        self.assertGreater(result[0], 0)

    def test_straight_line_excluded(self) -> None:
        """Test that straight-line (all same) yields 0 since it's not a repeating pattern."""
        data = [[3, 3, 3, 3, 3, 3, 3, 3]]
        result = longstring_pattern(data)
        self.assertAlmostEqual(result[0], 0.0)

    def test_with_nan(self) -> None:
        """Test handling of NaN values."""
        data = [[1, 2, np.nan, 1, 2, 1, 2, 1]]
        result = longstring_pattern(data, na_rm=True)
        self.assertEqual(len(result), 1)

    def test_min_columns(self) -> None:
        """Test minimum columns validation."""
        data = [[1]]
        with self.assertRaises(ValueError):
            longstring_pattern(data)


class TestMahadQQPlot(unittest.TestCase):
    """Tests for Mahalanobis Q-Q plot function."""

    def test_basic_functionality(self) -> None:
        """Test basic Q-Q plot computation."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(30, 3))
        theoretical, observed = mahad_qqplot(data)
        self.assertEqual(len(theoretical), 30)
        self.assertEqual(len(observed), 30)

    def test_shapes_match(self) -> None:
        """Test that output shapes match."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20, 4))
        theoretical, observed = mahad_qqplot(data)
        self.assertEqual(theoretical.shape, observed.shape)

    def test_sorted_ascending(self) -> None:
        """Test that outputs are sorted in ascending order."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(25, 3))
        theoretical, observed = mahad_qqplot(data)
        np.testing.assert_array_equal(theoretical, np.sort(theoretical))
        np.testing.assert_array_equal(observed, np.sort(observed))

    def test_positive_theoretical_quantiles(self) -> None:
        """Test that theoretical quantiles are positive."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20, 3))
        theoretical, _ = mahad_qqplot(data)
        self.assertTrue(np.all(theoretical > 0))

    def test_non_negative_observed(self) -> None:
        """Test that observed squared distances are non-negative."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20, 3))
        _, observed = mahad_qqplot(data)
        self.assertTrue(np.all(observed >= 0))

    def test_plot_false(self) -> None:
        """Test that plot=False returns without error."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(15, 3))
        theoretical, observed = mahad_qqplot(data, plot=False)
        self.assertIsInstance(theoretical, np.ndarray)
        self.assertIsInstance(observed, np.ndarray)

    def test_with_nan(self) -> None:
        """Test Q-Q plot with na_rm=True and NaN values."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(20, 3))
        data[0, 1] = np.nan
        theoretical, observed = mahad_qqplot(data, na_rm=True)
        self.assertEqual(len(theoretical), 19)

    def test_insufficient_observations_raises(self) -> None:
        """Test error for too few observations."""
        data = [[1, 2, 3, 4, 5]]
        with self.assertRaises(ValueError):
            mahad_qqplot(data)


class TestMarkov(unittest.TestCase):
    """Tests for Markov chain transition entropy."""

    def test_basic_functionality(self) -> None:
        """Test basic Markov entropy computation."""
        data = [[1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [1, 2, 1, 2, 1]]
        result = markov(data)
        self.assertEqual(len(result), 3)

    def test_constant_zero_entropy(self) -> None:
        """Test that constant responses yield zero entropy."""
        data = [[3, 3, 3, 3, 3, 3]]
        result = markov(data)
        self.assertAlmostEqual(result[0], 0.0)

    def test_varied_greater_than_constant(self) -> None:
        """Test that varied responses have higher entropy than constant."""
        data = [
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [1, 2, 3, 1, 3, 2, 1, 3, 2, 1],
        ]
        result = markov(data)
        self.assertGreater(result[1], result[0])

    def test_seesaw_low_entropy(self) -> None:
        """Test that seesaw pattern has lower entropy than varied."""
        data = [
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 3, 1, 3, 2, 1, 3, 2, 1],
        ]
        result = markov(data)
        self.assertLess(result[0], result[1])

    def test_flag_function(self) -> None:
        """Test Markov flagging."""
        data = [[3, 3, 3, 3, 3], [1, 3, 5, 2, 4], [1, 2, 1, 2, 1]]
        scores, flags = markov_flag(data, threshold=0.1)
        self.assertEqual(len(flags), 3)
        self.assertTrue(flags[0])

    def test_summary_function(self) -> None:
        """Test Markov summary."""
        data = [[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]]
        summary = markov_summary(data)
        self.assertIn("mean", summary)
        self.assertIn("n_total", summary)

    def test_with_nan(self) -> None:
        """Test handling of NaN values."""
        data = [[1, np.nan, 3, 4, 5], [1, 2, 3, 4, 5]]
        result = markov(data, na_rm=True)
        self.assertEqual(len(result), 2)
        self.assertFalse(np.isnan(result[0]))

    def test_all_missing_rows_return_nan(self) -> None:
        """Test all-missing data returns missing entropy scores."""
        data = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        result = markov(data, na_rm=True)
        self.assertTrue(np.all(np.isnan(result)))

    def test_na_rm_false_rejects_missing_values(self) -> None:
        """Test Markov raises when missing data is present and na_rm=False."""
        data = [[1, np.nan, 3], [1, 2, 3]]
        with self.assertRaises(ValueError):
            markov(data, na_rm=False)

    def test_na_rm_false_uses_vectorized_path(self) -> None:
        """Test complete data with na_rm=False uses vectorized transition entropy."""
        data = [[1, 2, 1, 2], [1, 2, 3, 1]]
        vectorized = markov(data, na_rm=False)
        rowwise = markov(data, na_rm=True)
        np.testing.assert_allclose(vectorized, rowwise)

    def test_rows_with_too_few_nonmissing_values_return_nan(self) -> None:
        """Test rows reduced below one transition return NaN."""
        data = [[np.nan, 1, np.nan], [1, 2, 3]]
        result = markov(data, na_rm=True)
        self.assertTrue(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[1]))

    def test_zero_transition_matrix_entropy(self) -> None:
        """Test empty transition matrices have zero entropy."""
        trans = np.zeros((2, 2), dtype=float)
        self.assertEqual(_transition_entropy(trans), 0.0)

    def test_min_columns_raises(self) -> None:
        """Test that too few columns raises ValueError."""
        data = [[1, 2], [3, 4]]
        with self.assertRaises(ValueError):
            markov(data)


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

    def test_insufficient_data_raises(self) -> None:
        """Test that insufficient data raises ValueError."""
        times = [[1.0, 2.0, 3.0]]
        with self.assertRaises(ValueError):
            response_time_mixture(times, n_components=2)

    def test_requires_scipy(self) -> None:
        """Test mixture model reports missing optional SciPy dependency."""
        times = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        with patch("ier.response_time.SCIPY_AVAILABLE", False), self.assertRaises(RuntimeError):
            response_time_mixture(times)

    def test_with_nan(self) -> None:
        """Test handling of NaN values in response times."""
        rng = np.random.default_rng(42)
        times = rng.uniform(0.5, 10.0, size=(20, 5))
        times[0, :] = np.nan
        result = response_time_mixture(times, random_seed=42)
        self.assertTrue(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[1]))


class TestOnset(unittest.TestCase):
    """Tests for carelessness onset detection."""

    def test_basic_functionality(self) -> None:
        """Test basic onset detection."""
        rng = np.random.default_rng(42)
        data = rng.choice([1, 2, 3, 4, 5], size=(3, 30))
        result = onset(data, window_size=5, min_items=10)
        self.assertEqual(len(result), 3)

    def test_attentive_then_careless(self) -> None:
        """Test detection when switching from attentive to careless."""
        rng = np.random.default_rng(42)
        attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 20))
        careless = np.full((1, 20), 3)
        data = np.hstack([attentive, careless])
        result = onset(data, window_size=5, min_items=10)
        self.assertEqual(len(result), 1)

    def test_consistently_attentive(self) -> None:
        """Test that consistently attentive respondent may not trigger."""
        rng = np.random.default_rng(42)
        data = rng.choice([1, 2, 3, 4, 5], size=(1, 40))
        result = onset(data, window_size=5, min_items=10)
        self.assertEqual(len(result), 1)

    def test_flag_function(self) -> None:
        """Test onset flagging."""
        rng = np.random.default_rng(42)
        attentive = rng.choice([1, 2, 3, 4, 5], size=(1, 20))
        careless = np.full((1, 20), 3)
        data = np.hstack([attentive, careless])
        flags = onset_flag(data, window_size=5, min_items=10)
        self.assertEqual(len(flags), 1)
        self.assertTrue(np.issubdtype(flags.dtype, np.bool_))

    def test_min_items(self) -> None:
        """Test that short surveys return NaN."""
        data = [[1, 2, 3, 4, 5]]
        result = onset(data, window_size=3, min_items=10)
        self.assertTrue(np.isnan(result[0]))

    def test_with_nan(self) -> None:
        """Test handling of NaN values."""
        rng = np.random.default_rng(42)
        data = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=(1, 30))
        data[0, 5] = np.nan
        result = onset(data, window_size=5, min_items=10, na_rm=True)
        self.assertEqual(len(result), 1)


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
            mad_positive_items=[0, 2, 4, 6, 8],
            mad_negative_items=[1, 3, 5, 7, 9],
            mad_scale_max=5,
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
