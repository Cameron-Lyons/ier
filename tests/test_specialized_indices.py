"""Unit tests for specialized IER indices."""

import unittest
from unittest.mock import patch

import numpy as np

from ier.guttman import guttman, guttman_flag
from ier.infrequency import infrequency, infrequency_flag
from ier.longstring import _longest_repeating_pattern, longstring_pattern
from ier.lz import (
    _compute_lz,
    _compute_lz_row,
    _estimate_discrimination,
    _estimate_theta,
    _ml_theta,
    lz,
    lz_flag,
)
from ier.mad import mad, mad_flag
from ier.mahad import mahad_qqplot
from ier.markov import (
    _transition_entropy,
    _transition_entropy_row,
    markov,
    markov_flag,
    markov_summary,
)
from ier.onset import onset, onset_flag
from ier.person_total import person_total
from ier.reliability import individual_reliability, individual_reliability_flag
from ier.semantic import semantic_ant, semantic_ant_flag, semantic_syn, semantic_syn_flag
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

    def test_batched_scoring_matches_expanded_formula(self) -> None:
        """Bounded means and correlations preserve the expanded definition."""
        from ier._correlation import row_correlations

        rng = np.random.default_rng(20260803)
        data = rng.normal(size=(513, 30))
        data[rng.random(data.shape) < 0.05] = np.nan
        original = data.copy()

        item_means = np.nanmean(data, axis=0)
        valid = ~np.isnan(data)
        mean_matrix = np.where(valid, item_means, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            centered_data = data - np.nanmean(data, axis=1, keepdims=True)
            centered_means = mean_matrix - np.nanmean(mean_matrix, axis=1, keepdims=True)
            covariance = np.nansum(centered_data * centered_means, axis=1)
            data_norm = np.sqrt(np.nansum(centered_data**2, axis=1))
            mean_norm = np.sqrt(np.nansum(centered_means**2, axis=1))
            expected = covariance / (data_norm * mean_norm)
        expected[np.sum(valid, axis=1) < 2] = np.nan

        with (
            patch("ier.person_total._PERSON_TOTAL_BATCH_ELEMENTS", 300),
            patch("ier.person_total.row_correlations", wraps=row_correlations) as correlations,
        ):
            result = person_total(data)

        self.assertGreater(correlations.call_count, 2)
        self.assertTrue(all(call.args[0].size <= 300 for call in correlations.call_args_list))
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-14, equal_nan=True)
        np.testing.assert_array_equal(data, original)

    def test_zero_variance_profiles_remain_unavailable(self) -> None:
        """Undefined correlations are NaN rather than synthetic zero scores."""
        data = [[3, 3, 3, 3], [1, 2, 4, 5], [2, 4, 5, 5]]
        result = person_total(data)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isfinite(result[1:]).all())

    def test_all_missing_item_is_ignored_without_warning(self) -> None:
        """An unavailable item does not warn or contaminate pairwise scores."""
        data = [[1, np.nan, 5], [2, np.nan, 3], [3, np.nan, 1]]
        result = person_total(data, na_rm=True)
        np.testing.assert_allclose(result, [1.0, 1.0, -1.0], rtol=0.0, atol=1e-15)

    def test_strict_missing_policy_propagates_unavailable_item(self) -> None:
        """Disabling missing removal leaves every affected correlation unavailable."""
        data = [[1, np.nan, 5], [2, np.nan, 3], [3, np.nan, 1]]
        result = person_total(data, na_rm=False)
        self.assertTrue(np.isnan(result).all())

    def test_strict_policy_matches_complete_data_scores(self) -> None:
        """Strict and pairwise policies agree when every response is present."""
        data = [[1, 2, 4, 5], [2, 3, 5, 6], [5, 4, 2, 1]]
        np.testing.assert_allclose(
            person_total(data, na_rm=False),
            person_total(data, na_rm=True),
            rtol=0.0,
            atol=1e-15,
        )


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

    def test_batched_scoring_matches_expanded_formula_without_mutation(self) -> None:
        """Bounded pair differences preserve the expanded semantic definition."""
        from ier._row_statistics import row_mean

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(503, 40)).astype(float)
        data[rng.random(data.shape) < 0.1] = np.nan
        original = data.copy()
        pairs = [(index, index + 20) for index in range(20)]
        pair_differences = np.abs(data[:, :20] - data[:, 20:])
        mean_differences = np.nanmean(pair_differences, axis=1)
        deviations = np.nanstd(data, axis=1)
        expected = np.full(len(data), np.nan)
        nonzero = deviations > 0
        expected[nonzero] = 1.0 - mean_differences[nonzero] / deviations[nonzero]
        zero = deviations == 0
        expected[zero] = np.where(np.isclose(mean_differences[zero], 0.0), 1.0, -1.0)
        np.clip(expected, -1.0, 1.0, out=expected)

        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 120),
            patch("ier._pair_statistics.row_mean", wraps=row_mean) as reductions,
        ):
            result = semantic_syn(data, pairs)

        self.assertGreater(reductions.call_count, 2)
        self.assertTrue(all(call.args[0].size <= 120 for call in reductions.call_args_list))
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-15, equal_nan=True)
        np.testing.assert_array_equal(data, original)

    def test_all_missing_data_returns_unavailable_scores_without_warning(self) -> None:
        data = np.full((3, 4), np.nan)
        pairs = [(0, 1), (2, 3)]

        self.assertTrue(np.isnan(semantic_syn(data, pairs)).all())
        self.assertTrue(np.isnan(semantic_ant(data, pairs)).all())

    def test_direct_flag_helpers_use_low_consistency_direction(self) -> None:
        synonym_data = [[1, 1, 5, 5], [1, 5, 5, 1]]
        antonym_data = [[1, 5, 2, 4], [1, 1, 2, 2]]
        pairs = [(0, 1), (2, 3)]

        synonym_scores, synonym_flags = semantic_syn_flag(
            synonym_data,
            pairs,
            threshold=0.0,
        )
        antonym_scores, antonym_flags = semantic_ant_flag(
            antonym_data,
            pairs,
            threshold=0.0,
            scale_min=1,
            scale_max=5,
        )

        np.testing.assert_array_equal(synonym_flags, [False, True])
        np.testing.assert_array_equal(antonym_flags, [False, True])
        self.assertGreater(synonym_scores[0], synonym_scores[1])
        self.assertGreater(antonym_scores[0], antonym_scores[1])


class TestGuttman(unittest.TestCase):
    """Tests for Guttman error functions."""

    @staticmethod
    def _expanded_counts(data: np.ndarray, *, na_rm: bool = True) -> np.ndarray:
        """Evaluate the direct item-pair definition for regression checks."""
        item_difficulty = np.nanmean(data, axis=0) if na_rm else np.mean(data, axis=0)
        ordered = data[:, np.argsort(item_difficulty)]
        expected = np.zeros(data.shape[0])
        for column in range(1, data.shape[1]):
            expected += np.count_nonzero(
                ordered[:, :column] < ordered[:, column, np.newaxis], axis=1
            )
        return expected

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

        expected = self._expanded_counts(data)

        np.testing.assert_array_equal(guttman(data, normalize=False), expected)

    def test_fractional_categories_are_scored_in_bounded_batches(self) -> None:
        """Fractional scales preserve pair counts while limiting each workspace."""
        from ier.guttman import _count_categorical_errors

        rng = np.random.default_rng(20260803)
        data = rng.choice([0.25, 1.5, 9.75], size=(503, 40))
        data[rng.random(data.shape) < 0.1] = np.nan
        original = data.copy()
        expected = self._expanded_counts(data)

        with (
            patch("ier.guttman._GUTTMAN_BATCH_CELLS", 120),
            patch(
                "ier.guttman._count_categorical_errors",
                wraps=_count_categorical_errors,
            ) as counter,
        ):
            result = guttman(data, normalize=False)

        self.assertGreater(counter.call_count, 2)
        self.assertTrue(all(call.args[0].size <= 120 for call in counter.call_args_list))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)

    def test_high_cardinality_raw_counts(self) -> None:
        """Test the bounded-memory fallback on continuous-style response data."""
        n_items = 70
        data = np.arange(n_items, dtype=float).reshape(1, -1)
        result = guttman(data, normalize=False)
        np.testing.assert_array_equal(result, [n_items * (n_items - 1) / 2])

    def test_high_cardinality_fallback_is_batched(self) -> None:
        """Continuous response data uses the same bounded row batches."""
        from ier.guttman import _count_pairwise_errors

        rng = np.random.default_rng(20260803)
        data = rng.normal(size=(53, 70))
        data[rng.random(data.shape) < 0.05] = np.nan
        expected = self._expanded_counts(data)

        with (
            patch("ier.guttman._GUTTMAN_BATCH_CELLS", 140),
            patch(
                "ier.guttman._count_pairwise_errors",
                wraps=_count_pairwise_errors,
            ) as counter,
        ):
            result = guttman(data, normalize=False)

        self.assertGreater(counter.call_count, 2)
        self.assertTrue(all(call.args[0].size <= 140 for call in counter.call_args_list))
        np.testing.assert_array_equal(result, expected)

    def test_strict_missing_policy_matches_direct_definition(self) -> None:
        """Strict scoring keeps the fixed denominator and direct pair semantics."""
        data = np.array([[1.0, np.nan, 3.0, 2.0], [3.0, 2.0, np.nan, 1.0]])
        expected = self._expanded_counts(data, na_rm=False) / 6.0
        np.testing.assert_array_equal(guttman(data, na_rm=False), expected)

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
            atol=2e-15,
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
            atol=2e-15,
            equal_nan=True,
        )

    def test_raw_moment_path_avoids_stable_centering_for_ordinary_rows(self) -> None:
        """Ordinary survey-scale rows stay on the allocation-light path."""
        rng = np.random.default_rng(61)
        data = rng.normal(loc=3.0, scale=0.8, size=(53, 12))
        data[rng.random(data.shape) < 0.05] = np.nan

        with patch(
            "ier.reliability._stable_paired_split_correlations",
            side_effect=AssertionError("stable centering fallback was called"),
        ):
            result = individual_reliability(data, n_splits=9, random_seed=17)

        self.assertTrue(np.isfinite(result).all())

    def test_large_offset_rows_use_stable_centering_reference(self) -> None:
        """Cancellation-prone raw moments retain the centered definition."""
        from ier.reliability import _stable_paired_split_correlations

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(53, 12)).astype(float) + 1e12
        data[rng.random(data.shape) < 0.05] = np.nan

        def centered_reference(
            half1: np.ndarray,
            half2: np.ndarray,
            has_missing: bool,
        ) -> tuple[np.ndarray, np.ndarray]:
            valid = ~np.isnan(half1) & ~np.isnan(half2) if has_missing else None
            return _stable_paired_split_correlations(half1, half2, valid)

        with patch(
            "ier.reliability._paired_split_correlations",
            side_effect=centered_reference,
        ):
            expected = individual_reliability(data, n_splits=9, random_seed=17)

        with patch(
            "ier.reliability._stable_paired_split_correlations",
            wraps=_stable_paired_split_correlations,
        ) as stable_fallback:
            actual = individual_reliability(data, n_splits=9, random_seed=17)

        self.assertGreater(stable_fallback.call_count, 0)
        np.testing.assert_array_equal(actual, expected)

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

    def test_unseeded_scoring_consumes_only_the_expected_permutations(self) -> None:
        """The global stream advances once per requested item split."""
        data = [[1, 2, 1, 2, 1, 2, 1], [1, 5, 2, 4, 3, 3, 2]]
        original_state = np.random.get_state()
        try:
            np.random.seed(20260803)
            individual_reliability(data, n_splits=7)
            actual_next = np.random.random()

            np.random.seed(20260803)
            for _ in range(7):
                np.random.permutation(7)
            expected_next = np.random.random()

            self.assertEqual(actual_next, expected_next)
        finally:
            np.random.set_state(original_state)

    def test_scoring_batches_rows_without_changing_seeded_results(self) -> None:
        """Forced tiny workspaces preserve missing-data scores and input values."""
        from ier.reliability import _paired_split_correlations

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(53, 12)).astype(float)
        data[rng.random(data.shape) < 0.1] = np.nan
        original = data.copy()
        expected = individual_reliability(data, n_splits=7, random_seed=17)

        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 18),
            patch(
                "ier.reliability._paired_split_correlations",
                wraps=_paired_split_correlations,
            ) as correlations,
        ):
            result = individual_reliability(data, n_splits=7, random_seed=17)

        self.assertGreater(correlations.call_count, 7)
        self.assertTrue(
            all(
                call.args[0].size <= 18 and call.args[1].size <= 18
                for call in correlations.call_args_list
            )
        )
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)

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

    def test_response_pattern_batches_and_handles_all_missing_rows(self) -> None:
        """Combined response summaries stay bounded and preserve missing semantics."""
        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(53, 11)).astype(float)
        data[rng.random(data.shape) < 0.1] = np.nan
        data[0] = np.nan
        original = data.copy()
        valid = ~np.isnan(data)
        counts = np.sum(valid, axis=1)
        expected_extreme = np.divide(
            np.sum(((data == 1) | (data == 5)) & valid, axis=1),
            counts,
            out=np.full(len(data), np.nan),
            where=counts > 0,
        )
        expected_midpoint = np.divide(
            np.sum((data == 3) & valid, axis=1),
            counts,
            out=np.full(len(data), np.nan),
            where=counts > 0,
        )

        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 30):
            result = response_pattern(data, scale_min=1, scale_max=5)

        np.testing.assert_allclose(
            result["extreme"], expected_extreme, rtol=0.0, atol=1e-15, equal_nan=True
        )
        np.testing.assert_allclose(
            result["midpoint"], expected_midpoint, rtol=0.0, atol=1e-15, equal_nan=True
        )
        assert np.isnan(result["acquiescence"][0])
        assert np.isnan(result["variability"][0])
        np.testing.assert_allclose(
            result["acquiescence"][1:],
            np.nanmean(data[1:], axis=1),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result["variability"][1:],
            np.nanstd(data[1:], axis=1),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_array_equal(data, original)


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

    def test_batched_scoring_matches_expanded_missing_policies(self) -> None:
        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(503, 40)).astype(float)
        data[rng.random(data.shape) < 0.1] = np.nan
        original = data.copy()
        positive_items = list(range(20))
        negative_items = list(range(20, 40))
        differences = np.abs(data[:, :20] - (6.0 - data[:, 20:]))

        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 120):
            missing_aware = mad(
                data,
                positive_items=positive_items,
                negative_items=negative_items,
                scale_min=1,
                scale_max=5,
            )
            strict = mad(
                data,
                positive_items=positive_items,
                negative_items=negative_items,
                scale_min=1,
                scale_max=5,
                na_rm=False,
            )

        np.testing.assert_allclose(
            missing_aware,
            np.nanmean(differences, axis=1),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            strict,
            np.mean(differences, axis=1),
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        )
        np.testing.assert_array_equal(data, original)

    def test_fractional_scale_bounds_are_preserved(self) -> None:
        data = [[1.5, 0.5], [0.5, 1.5]]
        result = mad(
            data,
            item_pairs=[(0, 1)],
            scale_min=0.5,
            scale_max=1.5,
        )
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_all_missing_data_returns_nan_without_warning(self) -> None:
        data = np.full((3, 2), np.nan)
        result = mad(data, item_pairs=[(0, 1)])
        self.assertTrue(np.isnan(result).all())

    def test_inverted_scale_bounds_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "scale_max"):
            mad(
                [[1.0, 5.0]],
                item_pairs=[(0, 1)],
                scale_min=5.0,
                scale_max=1.0,
            )


def _reference_discrimination(x: np.ndarray, *, na_rm: bool) -> np.ndarray:
    """Evaluate the former per-item point-biserial implementation."""
    result = np.ones(x.shape[1])
    total_score = np.nansum(x, axis=1) if na_rm else np.sum(x, axis=1)
    if np.std(total_score) == 0:
        return result

    for item_index in range(x.shape[1]):
        if na_rm:
            valid = ~np.isnan(x[:, item_index])
            item_response = x[valid, item_index]
            scores = total_score[valid]
        else:
            item_response = x[:, item_index]
            scores = total_score
        if len(np.unique(item_response)) < 2 or np.std(scores) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.corrcoef(item_response, scores)[0, 1]
        if np.isnan(correlation):
            continue
        correlation = np.clip(correlation, -0.99, 0.99)
        result[item_index] = correlation * 1.7 / np.sqrt(1 - correlation**2)
        result[item_index] = np.clip(result[item_index], 0.2, 3.0)
    return result


def _reference_missing_theta(
    x: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    *,
    na_rm: bool,
) -> np.ndarray:
    """Evaluate the former scalar missing-response ability path."""
    result = np.zeros(len(x))
    for row_index, row in enumerate(x):
        if na_rm:
            valid = ~np.isnan(row)
            responses = row[valid]
            item_discrimination = discrimination[valid]
            item_difficulty = difficulty[valid]
        else:
            responses = row
            item_discrimination = discrimination
            item_difficulty = difficulty

        if len(responses) == 0:
            result[row_index] = np.nan
        elif np.all(responses == 1):
            result[row_index] = 3.0
        elif np.all(responses == 0):
            result[row_index] = -3.0
        else:
            result[row_index] = _ml_theta(
                responses,
                item_discrimination,
                item_difficulty,
            )
    return result


def _reference_missing_lz(
    x: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    theta: np.ndarray,
    *,
    na_rm: bool,
) -> np.ndarray:
    """Evaluate the former scalar missing-response likelihood path."""
    result = np.zeros(len(x))
    for row_index, (row, row_theta) in enumerate(zip(x, theta, strict=True)):
        if np.isnan(row_theta):
            result[row_index] = np.nan
            continue

        if na_rm:
            valid = ~np.isnan(row)
            responses = row[valid]
            item_discrimination = discrimination[valid]
            item_difficulty = difficulty[valid]
        else:
            responses = row
            item_discrimination = discrimination
            item_difficulty = difficulty

        if len(responses) == 0:
            result[row_index] = np.nan
        else:
            result[row_index] = _compute_lz_row(
                responses,
                item_discrimination,
                item_difficulty,
                row_theta,
            )
    return result


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

    def test_complete_discrimination_contraction_matches_itemwise_reference(self) -> None:
        """Complete binary items share one contraction without changing estimates."""
        rng = np.random.default_rng(20260803)
        data = rng.integers(0, 2, size=(257, 37)).astype(float)
        data[:, 0] = 0.0
        data[:, 1] = 1.0
        expected = _reference_discrimination(data, na_rm=True)

        with patch(
            "ier.lz.np.corrcoef",
            side_effect=AssertionError("per-item correlations were constructed"),
        ):
            actual = _estimate_discrimination(data)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)
        self.assertEqual(actual[0], 1.0)
        self.assertEqual(actual[1], 1.0)

    def test_missing_discrimination_contraction_matches_itemwise_reference(self) -> None:
        """Missing-aware column blocks preserve established item estimates."""
        rng = np.random.default_rng(43)
        data = rng.integers(0, 2, size=(101, 13)).astype(float)
        data[rng.random(data.shape) < 0.08] = np.nan

        for na_rm in (True, False):
            with self.subTest(na_rm=na_rm):
                expected = _reference_discrimination(data, na_rm=na_rm)
                with (
                    patch("ier.lz._LZ_CALIBRATION_BLOCK_ELEMENTS", 101),
                    patch(
                        "ier.lz.np.corrcoef",
                        side_effect=AssertionError("per-item correlations were constructed"),
                    ),
                ):
                    actual = _estimate_discrimination(data, na_rm=na_rm)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)

    def test_missing_batch_kernels_match_scalar_rows(self) -> None:
        """Masked batches preserve ability and likelihood results for both policies."""
        from ier.lz import _compute_lz_masked_batch, _ml_theta_masked_batch

        rng = np.random.default_rng(20260803)
        data = rng.integers(0, 2, size=(257, 37)).astype(float)
        data[rng.random(data.shape) < 0.08] = np.nan
        data[0] = np.nan
        data[1] = 0.0
        data[1, ::5] = np.nan
        data[2] = 1.0
        data[2, ::7] = np.nan
        data[3] = rng.integers(0, 2, size=data.shape[1])
        discrimination = rng.uniform(0.2, 3.0, data.shape[1])
        difficulty = rng.uniform(-3.0, 3.0, data.shape[1])

        for na_rm in (True, False):
            with self.subTest(na_rm=na_rm):
                expected_theta = _reference_missing_theta(
                    data,
                    discrimination,
                    difficulty,
                    na_rm=na_rm,
                )
                expected_scores = _reference_missing_lz(
                    data,
                    discrimination,
                    difficulty,
                    expected_theta,
                    na_rm=na_rm,
                )

                with (
                    patch("ier.lz._LZ_MISSING_BATCH_ELEMENTS", 111),
                    patch(
                        "ier.lz._ml_theta",
                        side_effect=AssertionError("scalar ability solver was called"),
                    ),
                    patch(
                        "ier.lz._compute_lz_row",
                        side_effect=AssertionError("scalar likelihood scorer was called"),
                    ),
                    patch(
                        "ier.lz._ml_theta_masked_batch",
                        wraps=_ml_theta_masked_batch,
                    ) as theta_batches,
                    patch(
                        "ier.lz._compute_lz_masked_batch",
                        wraps=_compute_lz_masked_batch,
                    ) as score_batches,
                ):
                    actual_theta = _estimate_theta(
                        data,
                        discrimination,
                        difficulty,
                        na_rm=na_rm,
                    )
                    actual_scores = _compute_lz(
                        data,
                        discrimination,
                        difficulty,
                        actual_theta,
                        na_rm=na_rm,
                    )

                self.assertGreater(theta_batches.call_count, 1)
                self.assertGreater(score_batches.call_count, 1)
                self.assertTrue(
                    all(call.args[0].size <= 111 for call in theta_batches.call_args_list)
                )
                self.assertTrue(
                    all(call.args[0].size <= 111 for call in score_batches.call_args_list)
                )
                np.testing.assert_allclose(
                    actual_theta,
                    expected_theta,
                    rtol=0.0,
                    atol=2e-15,
                    equal_nan=True,
                )
                np.testing.assert_allclose(
                    actual_scores,
                    expected_scores,
                    rtol=0.0,
                    atol=5e-14,
                    equal_nan=True,
                )

    def test_strict_missing_policy_keeps_scores_unavailable(self) -> None:
        """Strict handling preserves unavailable public scores after batching."""
        data = np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, np.nan, 0.0],
            ]
        )

        self.assertTrue(np.isnan(lz(data, na_rm=False)).all())

    def test_missing_batch_preserves_infinite_custom_theta(self) -> None:
        """Infinite custom abilities retain the scalar likelihood semantics."""
        data = np.asarray(
            [
                [1.0, np.nan, 1.0, 0.0],
                [0.0, 1.0, np.nan, 1.0],
            ]
        )
        discrimination = np.asarray([0.5, 1.0, 1.5, 2.0])
        difficulty = np.asarray([-2.0, -0.5, 0.5, 2.0])
        theta = np.asarray([np.inf, -np.inf])
        expected = _reference_missing_lz(
            data,
            discrimination,
            difficulty,
            theta,
            na_rm=True,
        )

        actual = _compute_lz(
            data,
            discrimination,
            difficulty,
            theta,
            na_rm=True,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0.0)

    def test_constant_total_scores_keep_default_discrimination(self) -> None:
        """Undefined correlations retain the established default of one."""
        data = np.asarray(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
            ]
        )

        np.testing.assert_array_equal(_estimate_discrimination(data), np.ones(4))

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

    def test_extreme_item_parameters_do_not_overflow(self) -> None:
        data = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        difficulty = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
        discrimination = np.ones(4)
        theta = np.zeros(2)

        complete = lz(
            data,
            difficulty=difficulty,
            discrimination=discrimination,
            theta=theta,
        )
        data[0, 0] = np.nan
        missing = lz(
            data,
            difficulty=difficulty,
            discrimination=discrimination,
            theta=theta,
        )

        self.assertTrue(np.isfinite(complete).all())
        self.assertTrue(np.isfinite(missing).all())

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

    def test_proportion_flagging_uses_missing_policy(self) -> None:
        """Test proportional cutoffs and unavailable rows share scoring semantics."""
        data = [[5, 5], [1, 5], [np.nan, np.nan]]
        scores, flags = infrequency_flag(
            data,
            [0, 1],
            [5, 1],
            threshold=0.5,
            proportion=True,
            missing="omit",
        )

        np.testing.assert_allclose(scores, [0.5, 1.0, np.nan], equal_nan=True)
        np.testing.assert_array_equal(flags, [True, True, False])

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

    def test_missing_response_policies(self) -> None:
        """Test legacy, conservative, available-case, and strict missing policies."""
        data = [
            [np.nan, 1],
            [5, np.nan],
            [np.nan, np.nan],
            [1, 5],
        ]
        kwargs = {"item_indices": [0, 1], "expected_responses": [5, 1]}

        np.testing.assert_array_equal(infrequency(data, **kwargs), [0.0, 0.0, 0.0, 2.0])
        np.testing.assert_array_equal(
            infrequency(data, **kwargs, missing="fail"),
            [1.0, 1.0, 2.0, 2.0],
        )
        np.testing.assert_allclose(
            infrequency(data, **kwargs, missing="omit"),
            [0.0, 0.0, np.nan, 2.0],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            infrequency(data, **kwargs, missing="omit", proportion=True),
            [0.0, 0.0, np.nan, 1.0],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            infrequency(data, **kwargs, missing="propagate"),
            [np.nan, np.nan, np.nan, 2.0],
            equal_nan=True,
        )

    def test_configuration_validation(self) -> None:
        """Test invalid policies, selections, expectations, and thresholds."""
        data = [[5, 1]]
        cases = [
            ({"item_indices": [True], "expected_responses": [5]}, "integer column"),
            ({"item_indices": [0.5], "expected_responses": [5]}, "integer column"),
            ({"item_indices": [0, 0], "expected_responses": [5, 5]}, "duplicates"),
            ({"item_indices": [0], "expected_responses": [np.nan]}, "finite numeric"),
            ({"item_indices": [0], "expected_responses": [[5]]}, "finite numeric"),
            (
                {"item_indices": [0], "expected_responses": [5], "missing": "unknown"},
                "missing must be one of",
            ),
            (
                {"item_indices": [0], "expected_responses": [5], "proportion": 1},
                "proportion must be a boolean",
            ),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                infrequency(data, **kwargs)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            infrequency_flag(data, [0], [5], threshold=-0.1)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            infrequency_flag(data, [0], [5], threshold=1.1, proportion=True)
        with self.assertRaisesRegex(ValueError, "proportion must be a boolean"):
            infrequency_flag(data, [0], [5], proportion=1)  # type: ignore[arg-type]


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

    def test_missing_rows_use_bounded_compressed_batches(self) -> None:
        """Missing-response patterns retain scalar semantics in bounded groups."""
        from ier.longstring import _longest_repeating_patterns

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(257, 60)).astype(float)
        data[rng.random(data.shape) < 0.15] = np.nan
        data[0] = np.nan
        data[1, 3:] = np.nan
        original = data.copy()
        expected = np.zeros(len(data))
        for row_index, row in enumerate(data):
            retained = row[~np.isnan(row)]
            if retained.size >= 4:
                expected[row_index] = _longest_repeating_pattern(retained, 5)

        with (
            patch("ier.longstring._MISSING_COMPRESSION_BATCH_ELEMENTS", 300),
            patch(
                "ier.longstring._longest_repeating_patterns",
                wraps=_longest_repeating_patterns,
            ) as grouped,
        ):
            result = longstring_pattern(data, max_pattern_length=5)

        self.assertGreater(grouped.call_count, 1)
        self.assertTrue(all(call.args[0].size <= 300 for call in grouped.call_args_list))
        self.assertTrue(all(not np.isnan(call.args[0]).any() for call in grouped.call_args_list))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)

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

    def test_missing_rows_are_grouped_in_bounded_compressed_batches(self) -> None:
        """Missing removal retains order while avoiding respondent-wise scoring."""
        from ier.markov import _markov_complete

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(257, 60)).astype(float)
        data[rng.random(data.shape) < 0.15] = np.nan
        data[0] = np.nan
        data[1, 1:] = np.nan
        original = data.copy()
        expected = np.full(len(data), np.nan)
        for row_index, row in enumerate(data):
            retained = row[~np.isnan(row)]
            if len(retained) >= 2:
                expected[row_index] = _transition_entropy_row(retained)

        with (
            patch("ier.markov._MISSING_COMPRESSION_BATCH_CELLS", 300),
            patch("ier.markov._markov_complete", wraps=_markov_complete) as grouped_batches,
        ):
            result = markov(data)

        self.assertGreater(grouped_batches.call_count, 1)
        self.assertTrue(all(call.args[0].size <= 300 for call in grouped_batches.call_args_list))
        self.assertTrue(
            all(not np.isnan(call.args[0]).any() for call in grouped_batches.call_args_list)
        )
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(data, original)

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

    def test_noninteger_categories_match_missing_data_row_path(self) -> None:
        """Sorted-category encoding preserves the sparse row calculation."""
        rng = np.random.default_rng(17)
        data = rng.choice([0.25, 1.5, 9.75], size=(200, 24))

        scores = markov(data)
        row_path_data = np.column_stack((data, np.full(data.shape[0], np.nan)))
        row_path_scores = markov(row_path_data)

        np.testing.assert_allclose(scores, row_path_scores, rtol=0, atol=1e-12)

    def test_complete_fast_path_chunks_workspace(self) -> None:
        """Bounded dense batches preserve results when forced into small chunks."""
        rng = np.random.default_rng(23)
        data = rng.integers(1, 6, size=(250, 60)).astype(float)
        expected = markov(data)

        with patch("ier.markov._TRANSITION_BATCH_WORKSPACE_BYTES", 4096):
            result = markov(data)

        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)

    def test_high_cardinality_rows_use_sparse_observed_counts(self) -> None:
        """Distinct response values do not require a dense global state square."""
        data = np.arange(40_000, dtype=float).reshape(2, 20_000)

        scores = markov(data)
        row_path_data = np.column_stack((data, np.full(data.shape[0], np.nan)))
        row_path_scores = markov(row_path_data)

        np.testing.assert_array_equal(scores, [0.0, 0.0])
        np.testing.assert_array_equal(row_path_scores, scores)

    def test_high_cardinality_sparse_fallback_batches_rows(self) -> None:
        """Sorted sparse transition counts honor the bounded cell budget."""
        from ier.markov import _transition_entropy_sparse_batch

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 1001, size=(257, 40)).astype(float)
        original = data.copy()
        expected = np.array([_transition_entropy_row(row) for row in data])

        with (
            patch("ier.markov._SPARSE_TRANSITION_BATCH_CELLS", 240),
            patch(
                "ier.markov._transition_entropy_sparse_batch",
                wraps=_transition_entropy_sparse_batch,
            ) as sparse_batches,
        ):
            result = markov(data)

        self.assertGreater(sparse_batches.call_count, 2)
        self.assertTrue(
            all(call.args[0][:, :-1].size <= 240 for call in sparse_batches.call_args_list)
        )
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(data, original)

    def test_high_cardinality_sparse_entropy_matches_hand_calculation(self) -> None:
        """Observed-pair counts retain branching entropy above the dense-state limit."""
        row = np.concatenate((np.array([0.0, 1.0, 0.0, 2.0]), np.arange(3.0, 70.0)))

        result = markov(row[None, :])

        self.assertAlmostEqual(result[0], 2.0 / (len(row) - 1))

    def test_infinite_categories_are_scored_without_warnings(self) -> None:
        """Infinite response categories remain valid categorical labels."""
        data = [[np.inf, np.inf, 1.0, 1.0], [-np.inf, 0.0, np.inf, 0.0]]

        result = markov(data)

        self.assertTrue(np.isfinite(result).all())

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

    @staticmethod
    def _expanded_missing_onset(
        data: np.ndarray,
        *,
        window_size: int,
        min_items: int,
    ) -> np.ndarray:
        """Evaluate the established row-wise definition for regression checks."""
        from ier.onset import _SHAO_ZHANG_CRITICAL_VALUE

        result = np.full(len(data), np.nan)
        for row_index, raw_row in enumerate(data):
            row = raw_row[~np.isnan(raw_row)]
            if len(row) < min_items:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(row, window_size)
            series = np.std(windows, axis=1)
            n_observations = len(series)
            if n_observations < 3:
                continue

            prefix_sum = np.cumsum(series)
            prefix_square_sum = np.cumsum(series * series)
            centered_prefix = np.cumsum(series - np.mean(series))
            trim = max(1, n_observations // 10)
            candidates = np.arange(trim, n_observations - trim)
            if len(candidates) == 0:
                continue

            prefix_positions = candidates - 1
            variances = (
                prefix_square_sum[prefix_positions] - prefix_sum[prefix_positions] ** 2 / candidates
            )
            variances = np.maximum(variances, 1e-10)
            statistics = centered_prefix[candidates] ** 2 / variances
            offset = int(np.argmax(statistics))
            if statistics[offset] > _SHAO_ZHANG_CRITICAL_VALUE:
                result[row_index] = float(trim + offset + window_size - 1)
        return result

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

    def test_missing_rows_are_grouped_in_bounded_batches(self) -> None:
        """Compressed groups preserve the scalar definition and input values."""
        from ier.onset import _onset_complete

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(503, 40)).astype(float)
        data[rng.random(data.shape) < 0.2] = np.nan
        data[0] = np.nan
        data[1, 15:] = np.nan
        data[2] = rng.integers(1, 6, size=40)
        original = data.copy()
        expected = self._expanded_missing_onset(
            data,
            window_size=7,
            min_items=20,
        )

        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 120),
            patch("ier.onset._onset_complete", wraps=_onset_complete) as grouped,
        ):
            result = onset(data, window_size=7, min_items=20)

        self.assertGreater(grouped.call_count, 2)
        self.assertTrue(all(call.args[0].size <= 120 for call in grouped.call_args_list))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)

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

    def test_rolling_variability_matches_expanded_windows(self) -> None:
        """Rolling deviations retain stable sliding-window statistics."""
        from ier.onset import _running_inconsistency_complete

        rng = np.random.default_rng(20260803)
        data = 1e9 + rng.normal(size=(53, 41))
        windows = np.lib.stride_tricks.sliding_window_view(data, 7, axis=1)
        expected = np.std(windows, axis=2)

        result = _running_inconsistency_complete(data, 7)

        np.testing.assert_allclose(result, expected, rtol=2e-7, atol=1e-10)

    def test_batched_changepoints_match_expanded_formula(self) -> None:
        """Candidate-only workspaces preserve the changepoint definition."""
        from ier.onset import _SHAO_ZHANG_CRITICAL_VALUE, _shao_zhang_changepoints

        rng = np.random.default_rng(20260803)
        series = rng.normal(size=(37, 41))
        n_observations = series.shape[1]
        prefix_sum = np.cumsum(series, axis=1)
        prefix_square_sum = np.cumsum(series * series, axis=1)
        centered_prefix = np.cumsum(
            series - np.mean(series, axis=1, keepdims=True),
            axis=1,
        )
        trim = max(1, n_observations // 10)
        candidates = np.arange(trim, n_observations - trim)
        prefix_positions = candidates - 1
        variances = (
            prefix_square_sum[:, prefix_positions]
            - prefix_sum[:, prefix_positions] ** 2 / candidates
        )
        variances = np.maximum(variances, 1e-10)
        statistics = centered_prefix[:, candidates] ** 2 / variances
        offsets = np.argmax(statistics, axis=1)
        max_statistics = np.take_along_axis(statistics, offsets[:, None], axis=1)[:, 0]
        expected = np.full(len(series), np.nan)
        detected = max_statistics > _SHAO_ZHANG_CRITICAL_VALUE
        expected[detected] = trim + offsets[detected]

        result = _shao_zhang_changepoints(series.copy())

        np.testing.assert_array_equal(result, expected)

    def test_complete_fast_path_chunks_workspace(self) -> None:
        """Bounded batches preserve results when the workspace forces chunks."""
        from ier.onset import _running_inconsistency_complete

        rng = np.random.default_rng(23)
        data = rng.integers(1, 6, size=(250, 60)).astype(float)
        original = data.copy()
        expected = onset(data, window_size=7, min_items=20)

        with (
            patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 120),
            patch(
                "ier.onset._running_inconsistency_complete",
                wraps=_running_inconsistency_complete,
            ) as reductions,
        ):
            result = onset(data, window_size=7, min_items=20)

        self.assertGreater(reductions.call_count, 2)
        self.assertTrue(all(call.args[0].size <= 120 for call in reductions.call_args_list))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)


if __name__ == "__main__":
    unittest.main()
