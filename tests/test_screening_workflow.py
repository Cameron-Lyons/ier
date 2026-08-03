"""Unit tests for acquiescence, screening, and visualization functions."""

import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexOptions, PsychsynModel, psychsyn_model_scores
from ier.acquiescence import acquiescence, acquiescence_flag
from ier.irv import irv
from ier.mahad import mahad
from ier.psychsyn import psychant, psychsyn
from ier.screen import _reduce_screen_results, screen
from ier.visualize import plot_distributions, plot_flag_counts, plot_flagged_heatmap


class TestAcquiescence(unittest.TestCase):
    """Tests for acquiescence index."""

    def test_simple_mode_all_agree(self) -> None:
        data = [[5, 5, 5, 5], [5, 5, 5, 5]]
        scores = acquiescence(data, scale_min=1, scale_max=5)
        np.testing.assert_array_almost_equal(scores, [1.0, 1.0])

    def test_simple_mode_all_disagree(self) -> None:
        data = [[1, 1, 1, 1], [1, 1, 1, 1]]
        scores = acquiescence(data, scale_min=1, scale_max=5)
        np.testing.assert_array_almost_equal(scores, [0.0, 0.0])

    def test_simple_mode_midpoint(self) -> None:
        data = [[3, 3, 3, 3]]
        scores = acquiescence(data, scale_min=1, scale_max=5)
        np.testing.assert_array_almost_equal(scores, [0.5])

    def test_simple_mode_range(self) -> None:
        data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]
        scores = acquiescence(data, scale_min=1, scale_max=5)
        np.testing.assert_array_almost_equal(scores, [0.5, 0.5])

    def test_balanced_pair_mode(self) -> None:
        data = [[5, 1, 4, 2], [3, 3, 3, 3]]
        scores = acquiescence(
            data, scale_min=1, scale_max=5, positive_items=[0, 2], negative_items=[1, 3]
        )
        self.assertEqual(len(scores), 2)
        self.assertTrue(np.all(scores >= 0))
        self.assertTrue(np.all(scores <= 1))

    def test_balanced_pair_acquiescent(self) -> None:
        data = [[5, 4, 5, 4]]
        scores = acquiescence(
            data, scale_min=1, scale_max=5, positive_items=[0, 2], negative_items=[1, 3]
        )
        self.assertGreater(scores[0], 0.5)

    def test_balanced_pair_mode_batches_without_mutating_input(self) -> None:
        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(51, 8)).astype(float)
        data[rng.random(data.shape) < 0.1] = np.nan
        original = data.copy()
        positive_items = [0, 2, 4, 6]
        negative_items = [1, 3, 5, 7]
        pairs = (data[:, positive_items] + (6.0 - data[:, negative_items])) * 0.5
        expected = np.clip((np.nanmean(pairs, axis=1) - 1.0) / 4.0, 0.0, 1.0)

        with patch("ier._row_statistics._ROW_BATCH_ELEMENTS", 24):
            scores = acquiescence(
                data,
                scale_min=1,
                scale_max=5,
                positive_items=positive_items,
                negative_items=negative_items,
            )

        np.testing.assert_allclose(scores, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_array_equal(data, original)

    def test_scale_inference(self) -> None:
        data = [[1, 2, 3, 4, 5]]
        scores = acquiescence(data)
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 0.5)

    def test_nan_handling(self) -> None:
        data = [[1, np.nan, 3, 4, 5], [1, 2, 3, 4, 5]]
        scores = acquiescence(data, scale_min=1, scale_max=5, na_rm=True)
        self.assertEqual(len(scores), 2)
        self.assertFalse(np.isnan(scores[0]))

    def test_only_positive_items_raises(self) -> None:
        data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            acquiescence(data, positive_items=[0])

    def test_empty_items_raises(self) -> None:
        data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            acquiescence(data, positive_items=[], negative_items=[])

    def test_out_of_bounds_index_raises(self) -> None:
        data = [[1, 2, 3]]
        with self.assertRaises(ValueError):
            acquiescence(data, positive_items=[0], negative_items=[10])

    def test_equal_scale_returns_half(self) -> None:
        data = [[3, 3, 3]]
        scores = acquiescence(data, scale_min=3, scale_max=3)
        self.assertAlmostEqual(scores[0], 0.5)

    def test_inverted_scale_raises(self) -> None:
        data = [[3, 3, 3]]
        with self.assertRaises(ValueError):
            acquiescence(data, scale_min=5, scale_max=1)


class TestAcquiescenceFlag(unittest.TestCase):
    """Tests for acquiescence_flag function."""

    def test_returns_tuple(self) -> None:
        data = [[5, 5, 5, 5], [3, 3, 3, 3], [1, 1, 1, 1]]
        scores, flags = acquiescence_flag(data, scale_min=1, scale_max=5)
        self.assertEqual(len(scores), 3)
        self.assertEqual(len(flags), 3)
        self.assertEqual(flags.dtype, bool)

    def test_threshold_override(self) -> None:
        data = [[5, 5, 5, 5], [3, 3, 3, 3], [1, 1, 1, 1]]
        scores, flags = acquiescence_flag(data, scale_min=1, scale_max=5, threshold=0.9)
        self.assertTrue(flags[0])
        self.assertFalse(flags[1])
        self.assertFalse(flags[2])


class TestScreen(unittest.TestCase):
    """Tests for screen function."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.data = rng.integers(1, 6, size=(30, 10)).astype(float)
        self.data[0, :] = 3.0

    def test_basic_output_structure(self) -> None:
        result = screen(self.data)
        self.assertIn("scores", result)
        self.assertIn("flags", result)
        self.assertIn("thresholds", result)
        self.assertIn("threshold_sources", result)
        self.assertIn("percentiles", result)
        self.assertIn("flag_counts", result)
        self.assertIn("valid_index_counts", result)
        self.assertIn("consensus_eligible", result)
        self.assertIn("consensus_flags", result)
        self.assertIn("min_flags", result)
        self.assertIn("min_valid_indices", result)
        self.assertIn("n_indices", result)
        self.assertIn("indices_used", result)
        self.assertIn("errors", result)
        self.assertIn("n_respondents", result)
        self.assertIn("summary", result)
        self.assertEqual(result["n_respondents"], 30)

    def test_scores_and_flags_same_keys(self) -> None:
        result = screen(self.data)
        self.assertEqual(set(result["scores"].keys()), set(result["flags"].keys()))

    def test_flag_counts_bounded(self) -> None:
        result = screen(self.data)
        self.assertTrue(np.all(result["flag_counts"] >= 0))
        self.assertTrue(np.all(result["flag_counts"] <= result["n_indices"]))

    def test_flag_counts_length(self) -> None:
        result = screen(self.data)
        self.assertEqual(len(result["flag_counts"]), 30)

    def test_screen_reductions_accumulate_without_stacking(self) -> None:
        scores = {
            "first": np.array([1.0, np.nan, 3.0, np.nan]),
            "second": np.array([np.nan, 2.0, 3.0, np.nan]),
            "third": np.array([1.0, 2.0, np.nan, np.nan]),
        }
        flags = {
            "first": np.array([True, False, True, False]),
            "second": np.array([False, True, False, False]),
            "third": np.array([False, True, False, False]),
        }

        with patch(
            "ier.screen.np.column_stack",
            side_effect=AssertionError("screen matrix was constructed"),
        ):
            flag_counts, valid_counts, summary = _reduce_screen_results(scores, flags, 4)

        np.testing.assert_array_equal(flag_counts, np.array([1, 2, 1, 0]))
        np.testing.assert_array_equal(valid_counts, np.array([2, 2, 2, 0]))
        self.assertEqual(flag_counts.dtype, np.dtype(np.int_))
        self.assertEqual(valid_counts.dtype, np.dtype(np.int_))
        self.assertEqual(summary["first"]["n_valid"], 2)
        self.assertEqual(summary["first"]["n_unavailable"], 2)
        self.assertEqual(summary["first"]["n_flagged"], 2)
        self.assertEqual(summary["first"]["flag_rate"], 1.0)
        self.assertEqual(summary["second"]["flag_rate"], 0.5)

    def test_screen_reductions_report_unavailable_index(self) -> None:
        scores = {"missing": np.array([np.nan, np.nan])}
        flags = {"missing": np.array([False, False])}

        flag_counts, valid_counts, summary = _reduce_screen_results(scores, flags, 2)

        np.testing.assert_array_equal(flag_counts, [0, 0])
        np.testing.assert_array_equal(valid_counts, [0, 0])
        self.assertEqual(summary["missing"]["n_valid"], 0)
        self.assertEqual(summary["missing"]["n_unavailable"], 2)
        self.assertEqual(summary["missing"]["n_flagged"], 0)
        self.assertTrue(np.isnan(summary["missing"]["flag_rate"]))
        self.assertTrue(np.isnan(summary["missing"]["mean"]))

    def test_default_consensus_requires_two_index_flags(self) -> None:
        result = screen(self.data)
        np.testing.assert_array_equal(
            result["consensus_flags"],
            result["flag_counts"] >= 2,
        )
        self.assertEqual(result["min_flags"], 2)
        self.assertIsNone(result["min_valid_indices"])
        np.testing.assert_array_equal(result["consensus_eligible"], True)

    def test_minimum_valid_indices_suppresses_partial_consensus(self) -> None:
        data = np.array(
            [
                [5.0, 1.0, 1.0, 1.0],
                [np.nan, np.nan, 1.0, 2.0],
                [np.nan, np.nan, 3.0, 3.0],
            ]
        )
        result = screen(
            data,
            indices=["infrequency", "longstring"],
            options=IndexOptions(
                infrequency_item_indices=[0, 1],
                infrequency_expected_responses=[5, 1],
                infrequency_missing="omit",
            ),
            thresholds={"infrequency": 1.0, "longstring": 2.0},
            min_flags=1,
            min_valid_indices=2,
        )

        np.testing.assert_array_equal(result["valid_index_counts"], [2, 1, 1])
        np.testing.assert_array_equal(result["consensus_eligible"], [True, False, False])
        np.testing.assert_array_equal(result["flag_counts"], [1, 0, 1])
        np.testing.assert_array_equal(result["consensus_flags"], [True, False, False])
        self.assertEqual(result["min_valid_indices"], 2)

    def test_custom_consensus_threshold(self) -> None:
        result = screen(self.data, indices=["irv", "longstring"], min_flags=1)
        np.testing.assert_array_equal(
            result["consensus_flags"],
            result["flag_counts"] >= 1,
        )

    def test_fixed_thresholds_are_inclusive_and_direction_aware(self) -> None:
        baseline = screen(self.data, indices=["irv", "longstring"])
        thresholds = {
            "irv": float(baseline["scores"]["irv"][1]),
            "longstring": float(baseline["scores"]["longstring"][1]),
        }

        result = screen(
            self.data,
            indices=["irv", "longstring"],
            thresholds=thresholds,
        )

        self.assertEqual(result["thresholds"], thresholds)
        self.assertEqual(result["threshold_sources"], {"irv": "fixed", "longstring": "fixed"})
        self.assertEqual(result["percentiles"], {"irv": None, "longstring": None})
        np.testing.assert_array_equal(
            result["flags"]["irv"],
            result["scores"]["irv"] <= thresholds["irv"],
        )
        np.testing.assert_array_equal(
            result["flags"]["longstring"],
            result["scores"]["longstring"] >= thresholds["longstring"],
        )
        self.assertTrue(result["flags"]["irv"][1])
        self.assertTrue(result["flags"]["longstring"][1])

    def test_percentile_thresholds_are_returned(self) -> None:
        result = screen(self.data, indices=["irv", "longstring"])

        self.assertAlmostEqual(
            result["thresholds"]["irv"],
            float(np.percentile(result["scores"]["irv"], 5)),
        )
        self.assertAlmostEqual(
            result["thresholds"]["longstring"],
            float(np.percentile(result["scores"]["longstring"], 95)),
        )
        self.assertEqual(
            result["threshold_sources"],
            {"irv": "percentile", "longstring": "percentile"},
        )
        self.assertEqual(result["percentiles"], {"irv": 95.0, "longstring": 95.0})

    def test_per_index_tail_percentiles_override_global_setting(self) -> None:
        result = screen(
            self.data,
            indices=["irv", "longstring"],
            percentile=95,
            percentiles={"irv": 80, "longstring": 99},
        )

        irv_cutoff = float(np.percentile(result["scores"]["irv"], 20))
        longstring_cutoff = float(np.percentile(result["scores"]["longstring"], 99))
        self.assertAlmostEqual(result["thresholds"]["irv"], irv_cutoff)
        self.assertAlmostEqual(result["thresholds"]["longstring"], longstring_cutoff)
        self.assertEqual(result["percentiles"], {"irv": 80.0, "longstring": 99.0})
        np.testing.assert_array_equal(
            result["flags"]["irv"],
            result["scores"]["irv"] < irv_cutoff,
        )
        np.testing.assert_array_equal(
            result["flags"]["longstring"],
            result["scores"]["longstring"] > longstring_cutoff,
        )

    def test_invalid_percentile_raises_with_fixed_thresholds(self) -> None:
        for percentile in [-1, 101, float("nan"), True]:
            with (
                self.subTest(percentile=percentile),
                self.assertRaisesRegex(ValueError, "percentile"),
            ):
                screen(
                    self.data,
                    indices=["irv"],
                    thresholds={"irv": 0.5},
                    percentile=percentile,
                )

    def test_invalid_fixed_thresholds_raise(self) -> None:
        cases = [
            ({"nonexistent": 1.0}, ["irv"], "unknown threshold index"),
            ({"longstring": 1.0}, ["irv"], "not selected"),
            ({"onset": 1.0}, ["onset"], "presence flagging"),
            ({"irv": float("nan")}, ["irv"], "finite number"),
            ({"irv": True}, ["irv"], "finite number"),
        ]
        for thresholds, indices, message in cases:
            with self.subTest(thresholds=thresholds), self.assertRaisesRegex(ValueError, message):
                screen(self.data, indices=indices, thresholds=thresholds)

    def test_invalid_percentile_overrides_raise(self) -> None:
        cases = [
            ({"nonexistent": 95.0}, ["irv"], None, "unknown percentile index"),
            ({"longstring": 95.0}, ["irv"], None, "not selected"),
            ({"onset": 95.0}, ["onset"], None, "presence flagging"),
            ({"irv": float("nan")}, ["irv"], None, "finite number"),
            ({"irv": True}, ["irv"], None, "finite number"),
            ({"irv": -1.0}, ["irv"], None, "between 0 and 100"),
            ({"irv": 101.0}, ["irv"], None, "between 0 and 100"),
            ({"irv": 90.0}, ["irv"], {"irv": 0.5}, "both a threshold and percentile"),
        ]
        for percentiles, indices, thresholds, message in cases:
            with (
                self.subTest(percentiles=percentiles),
                self.assertRaisesRegex(ValueError, message),
            ):
                screen(
                    self.data,
                    indices=indices,
                    thresholds=thresholds,
                    percentiles=percentiles,
                )

    def test_invalid_consensus_threshold_raises(self) -> None:
        for value in [0, -1, 1.5, True]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "min_flags"):
                screen(self.data, min_flags=value)

    def test_invalid_minimum_valid_indices_raises(self) -> None:
        for value in [0, -1, 1.5, True, 3]:
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "min_valid_indices"),
            ):
                screen(
                    self.data,
                    indices=["irv", "longstring"],
                    min_valid_indices=value,  # type: ignore[arg-type]
                )

    def test_parallel_screen_matches_sequential_results_and_order(self) -> None:
        indices = ["irv", "longstring", "markov", "guttman", "individual_reliability"]
        options = IndexOptions(reliability_n_splits=10, reliability_random_seed=123)
        sequential = screen(self.data, indices=indices, options=options)
        parallel = screen(self.data, indices=indices, options=options, workers=4)

        self.assertEqual(parallel["indices_used"], indices)
        self.assertEqual(parallel["thresholds"], sequential["thresholds"])
        self.assertEqual(parallel["summary"], sequential["summary"])
        for name in indices:
            np.testing.assert_array_equal(parallel["scores"][name], sequential["scores"][name])
            np.testing.assert_array_equal(parallel["flags"][name], sequential["flags"][name])
        np.testing.assert_array_equal(parallel["flag_counts"], sequential["flag_counts"])
        np.testing.assert_array_equal(
            parallel["valid_index_counts"], sequential["valid_index_counts"]
        )
        np.testing.assert_array_equal(
            parallel["consensus_eligible"], sequential["consensus_eligible"]
        )
        np.testing.assert_array_equal(parallel["consensus_flags"], sequential["consensus_flags"])

    def test_invalid_worker_count_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers must be a positive integer"):
            screen(self.data, workers=0)

    def test_straightliner_flagged(self) -> None:
        result = screen(self.data)
        self.assertGreater(result["flag_counts"][0], 0)

    def test_specific_indices(self) -> None:
        result = screen(self.data, indices=["irv", "longstring"])
        self.assertEqual(set(result["indices_used"]), {"irv", "longstring"})

    def test_balanced_acquiescence_options_match_direct_scores(self) -> None:
        expected = acquiescence(
            self.data,
            scale_min=1,
            scale_max=5,
            positive_items=[0, 2, 4],
            negative_items=[1, 3, 5],
        )

        result = screen(
            self.data,
            indices=["acquiescence"],
            options=IndexOptions(
                scale_min=1,
                scale_max=5,
                acquiescence_positive_items=[0, 2, 4],
                acquiescence_negative_items=[1, 3, 5],
            ),
        )

        np.testing.assert_array_equal(result["scores"]["acquiescence"], expected)

    def test_irv_section_options_match_direct_scores(self) -> None:
        expected = irv(self.data, split=True, num_split=3)

        result = screen(
            self.data,
            indices=["irv"],
            options=IndexOptions(irv_num_split=3),
        )

        np.testing.assert_allclose(result["scores"]["irv"], expected, rtol=0.0, atol=1e-15)

    def test_irv_split_points_enable_custom_sections(self) -> None:
        split_points = [0, 2, 7, self.data.shape[1]]
        expected = irv(self.data, split=True, split_points=split_points)

        result = screen(
            self.data,
            indices=["irv"],
            options=IndexOptions(irv_split_points=split_points),
        )

        np.testing.assert_allclose(result["scores"]["irv"], expected, rtol=0.0, atol=1e-15)

    def test_psychometric_retry_seeds_match_direct_scores(self) -> None:
        data = np.array(
            [
                [1.0, 1.0, 2.0, 3.0],
                [1.0, np.nan, 2.0, 3.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        item_pairs = np.array([[0, 1], [0, 2], [0, 3]])

        with patch("ier.psychsyn._discover_item_pairs", return_value=item_pairs):
            expected_syn = psychsyn(data, resample_na=True, random_seed=17)
            expected_ant = psychant(data, resample_na=True, random_seed=29)
            synonym_result = screen(
                data,
                indices=["psychsyn"],
                options=IndexOptions(psychsyn_random_seed=17),
            )
            antonym_result = screen(
                data,
                indices=["psychant"],
                options=IndexOptions(psychant_random_seed=29),
            )

        np.testing.assert_array_equal(synonym_result["scores"]["psychsyn"], expected_syn)
        np.testing.assert_array_equal(antonym_result["scores"]["psychant"], expected_ant)

    def test_fixed_psychometric_model_matches_direct_scoring_without_discovery(self) -> None:
        data = np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 1.0, 3.0, 2.0],
                [2.0, 4.0, 1.0, 3.0],
            ]
        )
        model = PsychsynModel(
            np.asarray([[1, 0], [2, 0], [2, 1]]),
            n_items=4,
        )
        expected = psychsyn_model_scores(data, model)

        with patch("ier._registry.psychsyn", side_effect=AssertionError("rediscovered pairs")):
            result = screen(
                data,
                indices=["psychsyn"],
                options=IndexOptions(psychsyn_model=model),
            )

        np.testing.assert_array_equal(result["scores"]["psychsyn"], expected)

    def test_fixed_psychometric_item_contract_obeys_soft_and_strict_policy(self) -> None:
        model = PsychsynModel(
            np.asarray([[1, 0], [2, 0], [2, 1]]),
            n_items=3,
        )
        options = IndexOptions(psychsyn_model=model)

        result = screen(self.data, indices=["psychsyn"], options=options)

        self.assertEqual(result["scores"], {})
        self.assertIn("model requires 3", result["errors"]["psychsyn"])
        with self.assertRaisesRegex(ValueError, "model requires 3"):
            screen(
                self.data,
                indices=["psychsyn"],
                options=options,
                strict=True,
            )

    def test_invalid_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            screen(self.data, indices=["nonexistent"])

    def test_evenodd_included_with_factors(self) -> None:
        result = screen(
            self.data,
            indices=["evenodd"],
            options=IndexOptions(evenodd_factors=[5, 5]),
        )
        self.assertIn("evenodd", result["indices_used"])

    def test_mad_included_with_items(self) -> None:
        result = screen(
            self.data,
            indices=["mad"],
            options=IndexOptions(
                mad_positive_items=[0, 1, 2],
                mad_negative_items=[3, 4, 5],
            ),
        )
        self.assertIn("mad", result["indices_used"])

    def test_missing_optional_config_recorded_in_errors(self) -> None:
        result = screen(self.data, indices=["mad", "evenodd"])
        self.assertIn("mad", result["errors"])
        self.assertIn("evenodd", result["errors"])

    def test_strict_mode_raises_on_index_failure(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "index 'mad' failed: mad_positive_items",
        ):
            screen(self.data, indices=["irv", "mad"], strict=True)

        with self.assertRaisesRegex(ValueError, "strict must be a boolean"):
            screen(self.data, indices=["irv"], strict=cast("Any", 1))

    def test_summary_stats(self) -> None:
        result = screen(self.data, indices=["irv"])
        self.assertIn("irv", result["summary"])
        stats = result["summary"]["irv"]
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertEqual(stats["n_valid"], len(self.data))
        self.assertEqual(stats["n_unavailable"], 0)
        self.assertIn("n_flagged", stats)
        self.assertEqual(stats["flag_rate"], stats["n_flagged"] / len(self.data))

    def test_all_score_lengths_match(self) -> None:
        result = screen(self.data)
        for name, scores in result["scores"].items():
            self.assertEqual(len(scores), 30, f"score length mismatch for {name}")

    def test_all_flag_lengths_match(self) -> None:
        result = screen(self.data)
        for name, flags in result["flags"].items():
            self.assertEqual(len(flags), 30, f"flag length mismatch for {name}")

    def test_default_includes_guttman_and_mahad(self) -> None:
        result = screen(self.data)
        self.assertIn("guttman", result["indices_used"])
        self.assertIn("mahad", result["indices_used"])
        self.assertEqual(result["errors"], {})

    def test_new_registered_indices_with_config(self) -> None:
        result = screen(
            self.data,
            indices=["guttman", "semantic_syn", "infrequency", "individual_reliability"],
            options=IndexOptions(
                semantic_item_pairs=[(0, 1), (2, 3)],
                infrequency_item_indices=[0],
                infrequency_expected_responses=[3.0],
                reliability_n_splits=5,
                reliability_random_seed=0,
            ),
        )
        self.assertEqual(
            set(result["indices_used"]),
            {"guttman", "semantic_syn", "infrequency", "individual_reliability"},
        )
        self.assertEqual(result["errors"], {})

    def test_missing_semantic_and_infrequency_config_in_errors(self) -> None:
        result = screen(self.data, indices=["semantic_syn", "infrequency"])
        self.assertIn("semantic_syn", result["errors"])
        self.assertIn("infrequency", result["errors"])

    def test_semantic_ant_uses_configured_scale_bounds(self) -> None:
        data = np.array([[1, 5, 2, 4], [1, 1, 2, 2]], dtype=float)
        result = screen(
            data,
            indices=["semantic_ant"],
            options=IndexOptions(
                semantic_item_pairs=[(0, 1), (2, 3)],
                scale_min=1,
                scale_max=5,
            ),
        )
        np.testing.assert_array_almost_equal(result["scores"]["semantic_ant"], [1.0, -1.0])

    def test_onset_present_flag_mode(self) -> None:
        rng = np.random.default_rng(0)
        attentive = rng.choice([1, 2, 3, 4, 5], size=(5, 20))
        careless = np.full((5, 20), 3.0)
        data = np.hstack([attentive, careless])
        result = screen(
            data,
            indices=["onset"],
            options=IndexOptions(onset_window_size=5, onset_min_items=10),
        )
        self.assertIn("onset", result["indices_used"])
        self.assertEqual(result["flags"]["onset"].dtype, bool)

    def test_mahad_chi2_distances_are_dependency_free(self) -> None:
        distances = mahad(self.data, method="chi2")
        self.assertEqual(len(distances), 30)


class TestDataFrameInputs(unittest.TestCase):
    """Smoke tests for pandas/polars array-compatible inputs."""

    def test_pandas_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        from ier import irv, screen

        df = pd.DataFrame(np.array([[1, 2, 3, 4, 5], [3, 3, 3, 3, 3]], dtype=float))
        scores = irv(df)
        self.assertEqual(len(scores), 2)
        result = screen(df, indices=["irv", "longstring"])
        self.assertEqual(result["n_respondents"], 2)

    def test_polars_dataframe(self) -> None:
        pl = pytest.importorskip("polars")
        from ier import irv

        df = pl.DataFrame({"a": [1.0, 3.0], "b": [2.0, 3.0], "c": [3.0, 3.0], "d": [4.0, 3.0]})
        scores = irv(df.to_numpy())
        self.assertEqual(len(scores), 2)


class TestPlotDistributions(unittest.TestCase):
    """Tests for plot_distributions."""

    def setUp(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        rng = np.random.default_rng(42)
        self.data = rng.integers(1, 6, size=(20, 8)).astype(float)
        self.result = screen(self.data, indices=["irv", "longstring"])

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        fig = plot_distributions(self.result)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_correct_subplot_count(self) -> None:
        import matplotlib.pyplot as plt

        fig = plot_distributions(self.result)
        visible_axes = [ax for ax in fig.get_axes() if ax.get_visible()]
        self.assertEqual(len(visible_axes), 2)
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        import matplotlib.pyplot as plt

        fig = plot_distributions(self.result, figsize=(10, 5))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_scores(self) -> None:
        import matplotlib.pyplot as plt

        empty_result = {
            "scores": {},
            "flags": {},
            "flag_counts": np.array([]),
            "n_indices": 0,
            "indices_used": [],
            "n_respondents": 0,
            "summary": {},
        }
        fig = plot_distributions(empty_result)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotFlaggedHeatmap(unittest.TestCase):
    """Tests for plot_flagged_heatmap."""

    def setUp(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        rng = np.random.default_rng(42)
        self.data = rng.integers(1, 6, size=(20, 8)).astype(float)
        self.result = screen(self.data, indices=["irv", "longstring"])

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        fig = plot_flagged_heatmap(self.result)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_flags(self) -> None:
        import matplotlib.pyplot as plt

        empty_result = {
            "scores": {},
            "flags": {},
            "flag_counts": np.array([]),
            "n_indices": 0,
            "indices_used": [],
            "n_respondents": 0,
            "summary": {},
        }
        fig = plot_flagged_heatmap(empty_result)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotFlagCounts(unittest.TestCase):
    """Tests for plot_flag_counts."""

    def setUp(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        rng = np.random.default_rng(42)
        self.data = rng.integers(1, 6, size=(20, 8)).astype(float)
        self.result = screen(self.data, indices=["irv", "longstring"])

    def test_returns_figure(self) -> None:
        import matplotlib.pyplot as plt

        fig = plot_flag_counts(self.result)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        import matplotlib.pyplot as plt

        fig = plot_flag_counts(self.result, figsize=(12, 6))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
