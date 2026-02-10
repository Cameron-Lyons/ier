"""Unit tests for acquiescence, screen, and visualization functions."""

import unittest

import numpy as np

from ier.acquiescence import acquiescence, acquiescence_flag
from ier.screen import screen
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
        self.assertIn("flag_counts", result)
        self.assertIn("n_indices", result)
        self.assertIn("indices_used", result)
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

    def test_straightliner_flagged(self) -> None:
        result = screen(self.data)
        self.assertGreater(result["flag_counts"][0], 0)

    def test_specific_indices(self) -> None:
        result = screen(self.data, indices=["irv", "longstring"])
        self.assertEqual(set(result["indices_used"]), {"irv", "longstring"})

    def test_invalid_index_raises(self) -> None:
        with self.assertRaises(ValueError):
            screen(self.data, indices=["nonexistent"])

    def test_evenodd_included_with_factors(self) -> None:
        result = screen(self.data, indices=["evenodd"], evenodd_factors=[5, 5])
        self.assertIn("evenodd", result["indices_used"])

    def test_mad_included_with_items(self) -> None:
        result = screen(
            self.data,
            indices=["mad"],
            mad_positive_items=[0, 1, 2],
            mad_negative_items=[3, 4, 5],
        )
        self.assertIn("mad", result["indices_used"])

    def test_summary_stats(self) -> None:
        result = screen(self.data, indices=["irv"])
        self.assertIn("irv", result["summary"])
        stats = result["summary"]["irv"]
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("n_flagged", stats)

    def test_all_score_lengths_match(self) -> None:
        result = screen(self.data)
        for name, scores in result["scores"].items():
            self.assertEqual(len(scores), 30, f"score length mismatch for {name}")

    def test_all_flag_lengths_match(self) -> None:
        result = screen(self.data)
        for name, flags in result["flags"].items():
            self.assertEqual(len(flags), 30, f"flag length mismatch for {name}")


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
