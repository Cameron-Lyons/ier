"""Unit tests for specialized IER indices."""

import unittest
from unittest.mock import patch

import numpy as np

from ier.guttman import guttman, guttman_flag
from ier.infrequency import infrequency, infrequency_flag
from ier.longstring import longstring_pattern
from ier.lz import (
    _compute_lz,
    _compute_lz_row,
    _estimate_theta,
    _ml_theta,
    lz,
    lz_flag,
)
from ier.mad import mad, mad_flag
from ier.mahad import mahad_qqplot
from ier.markov import _transition_entropy, markov, markov_flag, markov_summary
from ier.onset import onset, onset_flag
from ier.person_total import person_total
from ier.reliability import individual_reliability, individual_reliability_flag
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

    def test_semantic_ant_reverse_scores_pairs(self) -> None:
        """Antonym pairs are compared after reflection around the response scale."""
        data = [[1, 5, 2, 4], [1, 1, 2, 2]]
        pairs = [(0, 1), (2, 3)]
        result = semantic_ant(data, pairs, scale_min=1, scale_max=5)
        np.testing.assert_array_almost_equal(result, [1.0, -1.0])

    def test_semantic_ant_infers_scale_bounds(self) -> None:
        """Antonym scoring infers omitted response-scale bounds from the matrix."""
        data = [[1, 5, 2, 4], [5, 1, 4, 2]]
        result = semantic_ant(data, [(0, 1), (2, 3)])
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_semantic_ant_rejects_inverted_scale(self) -> None:
        """Explicit antonym scale bounds must be ordered."""
        with self.assertRaisesRegex(ValueError, "scale_max"):
            semantic_ant([[1, 5]], [(0, 1)], scale_min=5, scale_max=1)

    def test_zero_variance_rows_are_handled(self) -> None:
        """Test semantic consistency handles zero-variance respondents deterministically."""
        data = [[3, 3, 3, 3], [3, 3, 3, 3]]
        pairs = [(0, 1), (2, 3)]
        syn = semantic_syn(data, pairs)
        ant = semantic_ant(data, pairs)
        np.testing.assert_array_almost_equal(syn, [1.0, 1.0])
        np.testing.assert_array_almost_equal(ant, [1.0, 1.0])


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

    def test_missing_values_exclude_only_unavailable_pairs(self) -> None:
        """Test missing responses leave all remaining ordered pairs available."""
        data = [[1, np.nan, 3, 2], [3, 2, np.nan, 1]]
        result = guttman(data)
        np.testing.assert_allclose(result, [2.0 / 3.0, 2.0 / 3.0])

    def test_sparse_categorical_fast_path_matches_pairwise_definition(self) -> None:
        """Sparse integer codes retain exact ordered-pair counts with missing data."""
        rng = np.random.default_rng(17)
        data = rng.choice([-2.0, 0.0, 5.0], size=(40, 12))
        data[rng.random(data.shape) < 0.1] = np.nan

        item_difficulty = np.nanmean(data, axis=0)
        ordered = data[:, np.argsort(item_difficulty)]
        expected = np.zeros(data.shape[0])
        for column in range(1, data.shape[1]):
            expected += np.count_nonzero(
                ordered[:, :column] < ordered[:, column, np.newaxis], axis=1
            )

        np.testing.assert_array_equal(guttman(data, normalize=False), expected)

    def test_high_cardinality_raw_counts(self) -> None:
        """Test the bounded-memory fallback on continuous-style response data."""
        n_items = 70
        data = np.arange(n_items, dtype=float).reshape(1, -1)
        result = guttman(data, normalize=False)
        np.testing.assert_array_equal(result, [n_items * (n_items - 1) / 2])

    def test_all_missing_data_returns_nan_without_warnings(self) -> None:
        """Test respondents with no comparable items return missing scores."""
        data = np.full((2, 4), np.nan)
        result = guttman(data)
        self.assertTrue(np.all(np.isnan(result)))

    def test_flag_function(self) -> None:
        """Test Guttman flagging."""
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
        flags = guttman_flag(data, threshold=0.3)
        self.assertEqual(len(flags), 2)
        self.assertTrue(
            np.issubdtype(flags.dtype, np.bool_),
            msg=f"Expected boolean dtype, got {flags.dtype}",
        )


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

    def test_seeded_complete_and_missing_scores_are_stable(self) -> None:
        """Streaming aggregation preserves the established seeded sequence."""
        complete = np.array(
            [
                [1, 2, 1, 2, 1, 2],
                [1, 5, 2, 4, 1, 5],
                [3, 3, 3, 3, 3, 3],
                [5, 4, 3, 2, 1, 2],
            ],
            dtype=float,
        )
        missing = complete.copy()
        missing[0, 1] = np.nan
        missing[1, 4] = np.nan
        missing[3, 0] = np.nan

        np.testing.assert_allclose(
            individual_reliability(complete, n_splits=25, random_seed=17),
            [
                0.0833333333333335,
                0.22658493155154946,
                np.nan,
                -0.11400136903312826,
            ],
            rtol=0,
            atol=1e-15,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            individual_reliability(missing, n_splits=25, random_seed=17),
            [
                0.16666666666666652,
                0.1818181818181818,
                np.nan,
                0.16666666666666655,
            ],
            rtol=0,
            atol=1e-15,
            equal_nan=True,
        )

    def test_seeded_scoring_does_not_modify_global_random_state(self) -> None:
        """A local seed must not change unrelated NumPy random draws."""
        data = [[1, 2, 1, 2, 1, 2], [1, 5, 2, 4, 3, 3]]
        original_state = np.random.get_state()
        try:
            np.random.seed(1234)
            expected_next = np.random.random()

            np.random.seed(1234)
            individual_reliability(data, n_splits=10, random_seed=99)

            self.assertEqual(np.random.random(), expected_next)
        finally:
            np.random.set_state(original_state)

    def test_n_splits_must_be_a_positive_integer(self) -> None:
        """Reject invalid split counts before allocating workspaces."""
        data = [[1, 2, 1, 2], [2, 1, 2, 1]]
        for n_splits in [True, 0, -1, 1.5]:
            with (
                self.subTest(n_splits=n_splits),
                self.assertRaisesRegex(
                    ValueError,
                    "n_splits must be a positive integer",
                ),
            ):
                individual_reliability(data, n_splits=n_splits)  # type: ignore[arg-type]


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

    def test_local_theta_solver_matches_scipy_reference_values(self) -> None:
        cases = [
            ([1, 1, 0, 0], [1, 1, 1, 1], [-1, -0.5, 0.5, 1]),
            ([1, 0, 1, 0], [0.5, 1, 1.5, 2], [-2, -0.5, 0.5, 2]),
            ([1, 1, 1, 0, 0], [3, 0.2, 1.2, 2, 0.7], [-3, -1, 0, 1, 3]),
            ([0, 1, 0, 1, 1], [1, -0.5, 2, 1.5, 0.2], [-2, -1, 0, 1, 2]),
        ]
        expected = [0.0, 0.5359047366047544, 0.3859716606867468, -0.6358889853624815]
        actual = [
            _ml_theta(np.asarray(responses), np.asarray(discrimination), np.asarray(difficulty))
            for responses, discrimination, difficulty in cases
        ]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-6)

        for theta, (responses, discrimination, difficulty) in zip(actual, cases, strict=True):
            response_array = np.asarray(responses)
            discrimination_array = np.asarray(discrimination)
            difficulty_array = np.asarray(difficulty)
            probabilities = 1.0 / (1.0 + np.exp(-discrimination_array * (theta - difficulty_array)))
            score = np.sum(discrimination_array * (response_array - probabilities))
            self.assertAlmostEqual(float(score), 0.0, places=11)

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

    def test_complete_batch_kernels_match_scalar_rows(self) -> None:
        """Batched complete-data kernels preserve exact scalar results."""
        rng = np.random.default_rng(29)
        for n_items in (4, 5, 17, 80):
            with self.subTest(n_items=n_items):
                data = rng.integers(0, 2, size=(257, n_items)).astype(float)
                data[0] = 0.0
                data[1] = 1.0
                discrimination = rng.uniform(0.2, 3.0, n_items)
                difficulty = rng.uniform(-3.0, 3.0, n_items)

                with patch("ier.lz._LZ_BATCH_ELEMENTS", 512):
                    theta = _estimate_theta(data, discrimination, difficulty)
                    scores = _compute_lz(
                        data,
                        discrimination,
                        difficulty,
                        theta,
                    )

                expected_theta = np.array(
                    [
                        -3.0
                        if np.all(row == 0)
                        else 3.0
                        if np.all(row == 1)
                        else _ml_theta(row, discrimination, difficulty)
                        for row in data
                    ]
                )
                expected_scores = np.array(
                    [
                        _compute_lz_row(row, discrimination, difficulty, row_theta)
                        for row, row_theta in zip(data, expected_theta, strict=True)
                    ]
                )
                np.testing.assert_array_equal(theta, expected_theta)
                np.testing.assert_array_equal(scores, expected_scores)

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
        self.assertEqual(result[0], 10.0)

    def test_embedded_and_partial_repetitions(self) -> None:
        """Test patterns with prefixes and incomplete final repetitions."""
        data = [
            [9, 9, 1, 2, 1, 2, 1, 2, 8],
            [1, 2, 3, 4, 1, 2, 3, 9, 8],
            [1, 1, 1, 2, 1, 2, 1, 2, 9],
        ]
        result = longstring_pattern(data)
        np.testing.assert_array_equal(result, [6.0, 7.0, 6.0])

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

    def test_complete_matrix_fast_path_matches_row_scoring(self) -> None:
        """Test complete matrix scoring against the missing-data row path."""
        rng = np.random.default_rng(42)
        for n_items in [4, 5, 12, 24]:
            for max_pattern_length in [2, 3, 5, 8, 20]:
                with self.subTest(n_items=n_items, max_pattern_length=max_pattern_length):
                    data = rng.integers(1, 6, size=(100, n_items)).astype(float)
                    result = longstring_pattern(
                        data,
                        max_pattern_length=max_pattern_length,
                    )
                    row_path_data = np.column_stack((data, np.full(data.shape[0], np.nan)))
                    row_path_result = longstring_pattern(
                        row_path_data,
                        max_pattern_length=max_pattern_length,
                    )
                    np.testing.assert_array_equal(result, row_path_result)

    def test_complete_matrix_fast_path_handles_wide_sequences(self) -> None:
        """Pattern counts remain exact after the workspace widens past uint8."""
        rng = np.random.default_rng(7)
        data = rng.integers(1, 6, size=(20, 300)).astype(float)
        data[0] = np.tile([1.0, 5.0], 150)

        result = longstring_pattern(data, max_pattern_length=8)
        row_path_data = np.column_stack((data, np.full(data.shape[0], np.nan)))
        row_path_result = longstring_pattern(row_path_data, max_pattern_length=8)

        self.assertEqual(result[0], 300.0)
        np.testing.assert_array_equal(result, row_path_result)

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

    def test_complete_data_matches_for_both_missing_value_policies(self) -> None:
        """Test complete data returns the same scores for either missing-value policy."""
        data = [[1, 2, 1, 2], [1, 2, 3, 1]]
        default_scores = markov(data, na_rm=True)
        strict_scores = markov(data, na_rm=False)
        np.testing.assert_allclose(default_scores, strict_scores)

    def test_categorical_fast_path_matches_missing_data_row_path(self) -> None:
        """Test bounded integer encoding against the established row scorer."""
        rng = np.random.default_rng(42)
        data = rng.choice([-2.0, 0.0, 5.0], size=(200, 24))

        scores = markov(data)
        row_path_data = np.column_stack((data, np.full(data.shape[0], np.nan)))
        row_path_scores = markov(row_path_data)

        np.testing.assert_allclose(scores, row_path_scores, rtol=0, atol=1e-12)

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

    def test_complete_fast_path_matches_missing_data_row_path(self) -> None:
        """Complete batches preserve established row-wise changepoints."""
        rng = np.random.default_rng(17)
        for n_items in [5, 10, 20, 41, 80]:
            data = rng.integers(1, 6, size=(24, n_items)).astype(float)
            if n_items >= 20:
                data[0, n_items // 2 :] = 3.0
            row_path_data = np.column_stack((data, np.full(data.shape[0], np.nan)))

            for window_size in [2, 3, 5, 10]:
                if window_size > n_items:
                    continue
                with self.subTest(n_items=n_items, window_size=window_size):
                    result = onset(data, window_size=window_size, min_items=window_size)
                    row_path_result = onset(
                        row_path_data,
                        window_size=window_size,
                        min_items=window_size,
                    )
                    np.testing.assert_array_equal(result, row_path_result)

    def test_complete_fast_path_chunks_workspace(self) -> None:
        """Bounded batches preserve results when the workspace forces chunks."""
        rng = np.random.default_rng(23)
        data = rng.integers(1, 6, size=(250, 60)).astype(float)
        expected = onset(data, window_size=7, min_items=20)

        with patch("ier.onset._ONSET_BATCH_WORKSPACE_BYTES", 4096):
            result = onset(data, window_size=7, min_items=20)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
