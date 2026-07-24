"""Unit tests for composite IER scorers."""

import unittest

import numpy as np

from ier import IndexOptions
from ier.composite import composite, composite_flag, composite_probability, composite_summary


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
