"""Unit tests for core IER indices and helpers."""

import unittest
from collections.abc import Iterator
from typing import cast
from unittest.mock import patch

import numpy as np
import numpy.typing as npt

from ier import MahadSummary, PsychsynSummary, mahad_summary, psychsyn_summary
from ier.evenodd import calculate_correlations, evenodd
from ier.irv import irv
from ier.longstring import (
    _avgstr_message,
    _longstr_message,
    _run_length_decode,
    _run_length_encode,
    longstring,
    longstring_scores,
)
from ier.mahad import _compute_mahalanobis_distance, mahad
from ier.psychsyn import (
    _complete_item_normalization,
    _compute_complete_person_scores,
    _compute_person_scores,
    _discover_item_pairs,
    _iter_item_correlation_tiles,
    _iter_pairwise_item_correlation_tiles,
    _normalized_complete_item_block,
    compute_person_correlations,
    get_highly_correlated_pairs,
    psychant,
    psychsyn,
    psychsyn_critval,
)


class TestLongstring(unittest.TestCase):
    """Test suite for longstring module functions.

    Tests run-length encoding/decoding, longest string detection,
    average string length calculation, and the main longstring function.
    """

    def test_run_length_encode(self) -> None:
        """Test run-length encoding produces correct character-count tuples."""
        self.assertEqual(
            _run_length_encode("AAABBBCCDAA"),
            [("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)],
        )
        self.assertEqual(_run_length_encode("A"), [("A", 1)])
        self.assertEqual(_run_length_encode(""), [])

    def test_run_length_encode_validation(self) -> None:
        """Test run-length encoding raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            _run_length_encode(123)
        with self.assertRaises(TypeError):
            _run_length_encode(None)

    def test_run_length_decode(self) -> None:
        """Test run-length decoding reconstructs original string correctly."""
        self.assertEqual(
            _run_length_decode([("A", 3), ("B", 3), ("C", 2), ("D", 1), ("A", 2)]),
            "AAABBBCCDAA",
        )
        self.assertEqual(_run_length_decode([("A", 1)]), "A")
        self.assertEqual(_run_length_decode([]), "")

    def test_run_length_decode_validation(self) -> None:
        """Test run-length decoding raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            _run_length_decode("not a list")
        with self.assertRaises(TypeError):
            _run_length_decode(None)

    def test_longstr_message(self) -> None:
        """Test longest string message returns correct character and length."""
        self.assertEqual(_longstr_message("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(_longstr_message("A"), ("A", 1))
        self.assertEqual(_longstr_message(""), None)

    def test_longstr_message_validation(self) -> None:
        """Test longest string message raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            _longstr_message(123)
        with self.assertRaises(TypeError):
            _longstr_message(None)

    def test_avgstr_message(self) -> None:
        """Test average string length calculation returns correct values."""
        self.assertAlmostEqual(_avgstr_message("AAABBBCCDAA"), 2.2)
        self.assertEqual(_avgstr_message("A"), 1.0)
        self.assertEqual(_avgstr_message(""), 0.0)

    def test_avgstr_message_validation(self) -> None:
        """Test average string message raises TypeError for invalid inputs."""
        with self.assertRaises(TypeError):
            _avgstr_message(123)
        with self.assertRaises(TypeError):
            _avgstr_message(None)

    def test_longstring(self) -> None:
        """Test main longstring function with single strings and lists."""
        self.assertEqual(longstring("AAAABBBCCDAA"), ("A", 4))
        self.assertEqual(longstring(["AAAABBBCCDAA", "A", ""]), [("A", 4), ("A", 1), None])

        self.assertAlmostEqual(longstring("AAABBBCCDAA", avg=True), 2.2)
        self.assertAlmostEqual(longstring(["AAABBBCCDAA", "A", ""], avg=True), [2.2, 1.0, 0.0])

    def test_longstring_numpy_array(self) -> None:
        """Test longstring function accepts numpy array input."""
        data: npt.NDArray[np.str_] = np.array(["AAAABBBCCDAA", "A", ""])
        result: list[tuple[str, int] | None] = longstring(data)
        expected: list[tuple[str, int] | None] = [("A", 4), ("A", 1), None]
        self.assertEqual(result, expected)

        object_data: npt.NDArray[np.object_] = np.array(["aaaa", "abc"], dtype=object)
        self.assertEqual(longstring(object_data), [("a", 4), ("a", 1)])

    def test_longstring_rejects_numeric_or_multidimensional_arrays(self) -> None:
        """Test arrays cannot be silently converted into meaningless text runs."""
        with self.assertRaisesRegex(TypeError, "must be strings"):
            longstring(np.array([1, 1, 2]))

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            longstring(np.array([["aaaa", "bbbb"], ["abc", "def"]]))

    def test_longstring_validation(self) -> None:
        """Test longstring function raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            longstring(None)
        with self.assertRaises(ValueError):
            longstring([])
        with self.assertRaises(TypeError):
            longstring(123)
        with self.assertRaises(TypeError):
            longstring(["abc", 123, "def"])

    def test_longstring_edge_cases(self) -> None:
        """Test longstring function handles edge cases correctly."""
        self.assertEqual(longstring("a"), ("a", 1))

        self.assertEqual(longstring("abcdef"), ("a", 1))

        self.assertEqual(longstring("aaaa"), ("a", 4))

    def test_missing_numeric_rows_use_bounded_compressed_batches(self) -> None:
        """Numeric longest runs retain missing-removal order in bounded groups."""
        from ier.longstring import _longstring_scores_complete

        rng = np.random.default_rng(20260803)
        data = rng.integers(1, 6, size=(257, 60)).astype(float)
        data[rng.random(data.shape) < 0.15] = np.nan
        data[0] = np.nan
        data[1, 1:] = np.nan
        original = data.copy()
        expected = np.zeros(len(data))
        for row_index, row in enumerate(data):
            retained = row[~np.isnan(row)]
            if retained.size:
                boundaries = np.flatnonzero(np.diff(retained) != 0) + 1
                expected[row_index] = np.max(
                    np.diff(np.concatenate(([0], boundaries, [retained.size])))
                )

        with (
            patch("ier.longstring._MISSING_COMPRESSION_BATCH_ELEMENTS", 300),
            patch(
                "ier.longstring._longstring_scores_complete",
                wraps=_longstring_scores_complete,
            ) as grouped,
        ):
            result = longstring_scores(data)

        self.assertGreater(grouped.call_count, 1)
        self.assertTrue(all(call.args[0].size <= 300 for call in grouped.call_args_list))
        self.assertTrue(all(not np.isnan(call.args[0]).any() for call in grouped.call_args_list))
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(data, original)


class TestIRV(unittest.TestCase):
    """Test suite for intra-individual response variability (IRV) function.

    Tests basic IRV calculation, handling of missing values, and split-half
    IRV computation with various configurations.
    """

    def test_basic_irv(self) -> None:
        """Test basic IRV calculation produces expected standard deviations."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x)
        expected: npt.NDArray[np.float64] = np.array([1.11803399, 2.23606798, 2.23606798])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_list_input(self) -> None:
        """Test IRV function accepts list input and converts appropriately."""
        x: list[list[int]] = [[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]]
        result: npt.NDArray[np.float64] = irv(x)
        expected: npt.NDArray[np.float64] = np.array([1.11803399, 2.23606798, 2.23606798])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_na(self) -> None:
        """Test IRV calculation correctly handles missing values when na_rm=True."""
        x: npt.NDArray[np.float64] = np.array([[1, np.nan, 3, 4], [2, 4, 6, 8], [np.nan, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x, na_rm=True)
        expected: npt.NDArray[np.float64] = np.array([1.2472191, 2.236068, 1.6329932])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_split(self) -> None:
        """Test split-half IRV calculation with automatic splitting."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x, split=True, num_split=2)
        expected: npt.NDArray[np.float64] = np.array([0.5, 1.0, 1.0])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_with_custom_split_points(self) -> None:
        """Test split IRV with custom split point indices."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [1, 3, 5, 7]])
        result: npt.NDArray[np.float64] = irv(x, split=True, split_points=[0, 2, 4])
        self.assertEqual(len(result), 3)

    def test_irv_custom_sections_match_independent_row_standard_deviations(self) -> None:
        """Test uneven section scores retain the established unweighted mean."""
        x = np.array(
            [
                [1.0, 2.0, 4.0, 4.0, 5.0, 9.0],
                [2.0, np.nan, 2.0, 3.0, 5.0, 8.0],
                [5.0, 4.0, 3.0, 2.0, 1.0, 1.0],
            ]
        )
        expected = np.mean(
            [
                np.nanstd(x[:, :2], axis=1),
                np.nanstd(x[:, 2:5], axis=1),
                np.nanstd(x[:, 5:], axis=1),
            ],
            axis=0,
        )

        result = irv(x, split=True, split_points=[0, 2, 5, 6])

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-15)

    def test_irv_with_split_and_na(self) -> None:
        """Test split IRV correctly handles missing values."""
        x: npt.NDArray[np.float64] = np.array([[1, 2, np.nan, 4], [2, 4, 6, 8], [1, 3, np.nan, 7]])
        result: npt.NDArray[np.float64] = irv(x, na_rm=True, split=True, num_split=2)
        expected: npt.NDArray[np.float64] = np.array([0.25, 1.0, 0.5])
        np.testing.assert_almost_equal(result, expected)

    def test_irv_validation(self) -> None:
        """Test IRV function raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            irv(None)
        with self.assertRaises(ValueError):
            irv([])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, num_split=0)
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[1, 3])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0, 5])
        with self.assertRaises(ValueError):
            irv([[1, 2, 3]], split=True, split_points=[0, 2, 1])

    def test_irv_edge_cases(self) -> None:
        """Test IRV handles edge cases like single-column and excessive splits."""
        x: npt.NDArray[np.float64] = np.array([[1], [2], [3]])
        result: npt.NDArray[np.float64] = irv(x)
        self.assertEqual(len(result), 3)

        x = np.array([[1, 2, 3], [4, 5, 6]])
        result = irv(x, split=True, num_split=5)
        self.assertEqual(len(result), 2)

    def test_irv_na_rm_false_propagates_nan(self) -> None:
        """Test IRV preserves NaN behavior when na_rm=False."""
        x: npt.NDArray[np.float64] = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
        result: npt.NDArray[np.float64] = irv(x, na_rm=False)
        self.assertTrue(np.isnan(result[0]))
        self.assertFalse(np.isnan(result[1]))


class TestMahadFunction(unittest.TestCase):
    """Test suite for Mahalanobis distance (MAHAD) function.

    Tests distance calculation, outlier flagging with different methods,
    handling of missing values, and summary statistics generation.
    """

    def setUp(self) -> None:
        """Initialize test data with normal cases and one outlier."""
        self.data: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
        )

    def test_basic_functionality(self) -> None:
        """Test basic Mahalanobis distance calculation."""
        distances: npt.NDArray[np.float64] = mahad(self.data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_list_input(self) -> None:
        """Test MAHAD function accepts list input."""
        data: list[list[int]] = [[1, 2, 3], [2, 3, 4], [3, 4, 5], [10, 10, 10]]
        distances: npt.NDArray[np.float64] = mahad(data)
        self.assertEqual(len(distances), 4)
        self.assertTrue((distances >= 0).all())

    def test_with_na_rm(self) -> None:
        """Test MAHAD handles missing values correctly when na_rm=True."""
        self.data_with_nan: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3], [2, np.nan, 4], [5, 6, 7], [10, 10, 10]], dtype=float
        )
        distances: npt.NDArray[np.float64] = mahad(self.data_with_nan, na_rm=True)
        self.assertTrue(np.all(np.isnan(distances) | (distances >= 0)))

    def test_without_na_rm_raises_error(self) -> None:
        """Test MAHAD raises ValueError when data contains NaN and na_rm=False."""
        self.data_with_nan: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3], [2, np.nan, 4], [5, 6, 7], [10, 10, 10]]
        )
        with self.assertRaises(ValueError):
            mahad(self.data_with_nan, na_rm=False)

    def test_flagging(self) -> None:
        """Test outlier flagging returns boolean array."""
        _: npt.NDArray[np.float64]
        flags: npt.NDArray[np.bool_]
        _, flags = mahad(self.data, flag=True)
        self.assertEqual(len(flags), 4)
        self.assertTrue(isinstance(flags[0], np.bool_))

    def test_flagging_with_different_methods(self) -> None:
        """Test outlier flagging works with IQR and z-score methods."""
        _: npt.NDArray[np.float64]
        flags_iqr: npt.NDArray[np.bool_]
        _, flags_iqr = mahad(self.data, flag=True, method="iqr")
        self.assertEqual(len(flags_iqr), 4)

        flags_zscore: npt.NDArray[np.bool_]
        _, flags_zscore = mahad(self.data, flag=True, method="zscore")
        self.assertEqual(len(flags_zscore), 4)

    def test_flagging_with_threshold(self) -> None:
        """Test outlier flagging respects confidence threshold."""
        distances: npt.NDArray[np.float64]
        flags: npt.NDArray[np.bool_]
        distances, flags = mahad(self.data, flag=True, confidence=0.99)
        threshold = 11.34486673014437
        flagged_distances: npt.NDArray[np.float64] = distances[flags]
        self.assertTrue((flagged_distances**2 > threshold).all())

    def test_no_negative_distances(self) -> None:
        """Test Mahalanobis distances are never negative."""
        distances: npt.NDArray[np.float64] = mahad(self.data)
        self.assertTrue((distances >= 0).all())

    def test_singular_covariance_matrix(self) -> None:
        """Test MAHAD handles singular covariance via pseudo-inverse path."""
        data: npt.NDArray[np.float64] = np.array(
            [
                [1.0, 1.0, 2.0],
                [2.0, 2.0, 4.0],
                [3.0, 3.0, 6.0],
                [4.0, 4.0, 8.0],
            ]
        )
        distances = mahad(data)
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all(distances >= 0))

    def test_constant_matrix_returns_zero_without_warnings(self) -> None:
        """Test an all-zero covariance uses an exact zero pseudo-inverse."""
        distances = mahad(np.full((20, 4), 3.0))

        np.testing.assert_array_equal(distances, np.zeros(20))

    def test_matrix_product_matches_quadratic_form(self) -> None:
        """Test bounded distance evaluation against the direct quadratic form."""
        rng = np.random.default_rng(42)
        data = rng.normal(size=(200, 12))
        original = data.copy()
        centered = data - np.mean(data, axis=0)
        covariance = np.cov(data, rowvar=False)
        inverse = np.linalg.pinv(covariance)
        expected = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse, centered))

        def small_row_slices(n_rows: int, n_columns: int) -> Iterator[tuple[int, int]]:
            del n_columns
            for start in range(0, n_rows, 17):
                yield start, min(start + 17, n_rows)

        with patch("ier.mahad.row_slices", side_effect=small_row_slices) as slices:
            result = _compute_mahalanobis_distance(data)

        self.assertEqual(slices.call_count, 2)
        np.testing.assert_array_equal(data, original)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_invalid_confidence(self) -> None:
        """Test MAHAD raises ValueError for invalid confidence values."""
        with self.assertRaises(ValueError):
            mahad(self.data, flag=True, confidence=1.1)
        with self.assertRaises(ValueError):
            mahad(self.data, flag=True, confidence=-0.1)

    def test_invalid_method(self) -> None:
        """Test MAHAD raises ValueError for invalid flagging method."""
        with self.assertRaises(ValueError):
            mahad(self.data, method="invalid")

    def test_validation(self) -> None:
        """Test MAHAD raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            mahad(None)
        with self.assertRaises(ValueError):
            mahad([])
        with self.assertRaises(ValueError):
            mahad([[1, 2]])

    def test_mahad_summary(self) -> None:
        """Test MAHAD summary returns expected statistics dictionary."""
        summary: MahadSummary = mahad_summary(self.data)
        self.assertEqual(
            set(summary),
            {
                "mean",
                "std",
                "min",
                "max",
                "median",
                "outliers",
                "total",
                "valid_count",
                "missing_count",
            },
        )
        self.assertIn("mean", summary)
        self.assertIn("std", summary)
        self.assertIn("outliers", summary)
        self.assertIn("total", summary)

    def test_mahad_summary_reports_partially_missing_cohort(self) -> None:
        """Available distances retain statistics when incomplete rows are removed."""
        data = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [np.nan, 2.0]])

        summary = mahad_summary(data, na_rm=True)

        self.assertEqual(summary["total"], 5)
        self.assertEqual(summary["valid_count"], 4)
        self.assertEqual(summary["missing_count"], 1)
        statistics = np.array(
            [summary["mean"], summary["std"], summary["min"], summary["max"], summary["median"]]
        )
        self.assertTrue(np.all(np.isfinite(statistics)))


class TestEvenOddFunction(unittest.TestCase):
    """Test suite for even-odd consistency scoring function.

    Tests basic functionality, handling of missing data, diagnostic output,
    and validation of factor specifications.
    """

    def test_basic_functionality(self) -> None:
        """Test basic even-odd consistency scoring."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_with_list_input(self) -> None:
        """Test even-odd function accepts list input."""
        data: list[list[int]] = [[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]]
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_with_missing_data(self) -> None:
        """Test even-odd handles missing values in input data."""
        data: npt.NDArray[np.float64] = np.array([[1, np.nan, 3, 4, np.nan, 6], [2, 3, 4, 5, 6, 7]])
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_diag_output(self) -> None:
        """Test even-odd returns diagnostic values when diag=True."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]])
        factors: list[int] = [3, 3]
        scores: npt.NDArray[np.float64]
        diag_vals: npt.NDArray[np.float64]
        scores, diag_vals = evenodd(data, factors, diag=True)
        self.assertEqual(len(scores), 2)
        self.assertEqual(len(diag_vals), 2)

    def test_varying_factors(self) -> None:
        """Test even-odd with different factor sizes."""
        data: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]]
        )
        factors: list[int] = [4, 4]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_factor_correlations_accumulate_without_stacking(self) -> None:
        """Streaming reduction matches a matrix reference with missing values."""
        rng = np.random.default_rng(42)
        data = rng.integers(1, 6, size=(100, 16)).astype(float)
        data[rng.random(data.shape) < 0.1] = np.nan
        factors = [4, 4, 4, 4]
        factor_correlations = []
        start = 0
        for factor_size in factors:
            stop = start + factor_size
            factor_correlations.append(
                calculate_correlations(data[:, start:stop:2], data[:, start + 1 : stop : 2])
            )
            start = stop
        matrix = np.column_stack(factor_correlations)
        valid = ~np.isnan(matrix)
        expected_counts = np.sum(valid, axis=1)
        expected_scores = np.divide(
            np.sum(matrix, axis=1, where=valid),
            expected_counts,
            out=np.zeros(len(data)),
            where=expected_counts > 0,
        )

        with patch(
            "ier.evenodd.np.column_stack",
            side_effect=AssertionError("factor correlation matrix was constructed"),
        ):
            scores, counts = evenodd(data, factors, diag=True)

        np.testing.assert_allclose(scores, expected_scores, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(counts, expected_counts)

    def test_single_item_factors(self) -> None:
        """Test even-odd with single-item factors."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        factors: list[int] = [1, 2, 2]
        scores: npt.NDArray[np.float64] = evenodd(data, factors)
        self.assertEqual(len(scores), 2)

    def test_validation(self) -> None:
        """Test even-odd raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            evenodd([], [])
        with self.assertRaises(ValueError):
            evenodd([[1, 2, 3]], [])
        with self.assertRaises(ValueError):
            evenodd([[1, 2, 3]], [2, 2])
        with self.assertRaises(ValueError):
            evenodd([], [2, 2])


class TestCalculateCorrelations(unittest.TestCase):
    """Test suite for the calculate_correlations helper function."""

    def test_basic_correlation(self) -> None:
        """Test basic correlation calculation between even and odd columns."""
        even_cols: npt.NDArray[np.float64] = np.array([[1, 3, 5], [2, 4, 6]])
        odd_cols: npt.NDArray[np.float64] = np.array([[2, 4, 6], [3, 5, 7]])
        result: npt.NDArray[np.float64] = calculate_correlations(even_cols, odd_cols)
        self.assertEqual(len(result), 2)
        np.testing.assert_almost_equal(result, [1.0, 1.0])

    def test_negative_correlation(self) -> None:
        """Test correlation calculation with negatively correlated data."""
        even_cols: npt.NDArray[np.float64] = np.array([[1, 2, 3], [4, 5, 6]])
        odd_cols: npt.NDArray[np.float64] = np.array([[3, 2, 1], [6, 5, 4]])
        result: npt.NDArray[np.float64] = calculate_correlations(even_cols, odd_cols)
        self.assertEqual(len(result), 2)
        np.testing.assert_almost_equal(result, [-1.0, -1.0])

    def test_correlations_use_bounded_row_blocks(self) -> None:
        """Wide pairwise-complete correlations preserve results in bounded workspaces."""
        rng = np.random.default_rng(20260803)
        left = rng.normal(size=(257, 30))
        right = rng.normal(size=(257, 30))
        missing = rng.random(left.shape) < 0.1
        left[missing] = np.nan
        right[rng.random(right.shape) < 0.1] = np.nan
        left[0] = 1.0
        left[1, 1:] = np.nan
        original_left = left.copy()
        original_right = right.copy()
        expected = np.empty(len(left))
        for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
            valid = ~(np.isnan(left_row) | np.isnan(right_row))
            if np.count_nonzero(valid) < 2:
                expected[row_index] = np.nan
                continue
            left_centered = left_row[valid] - np.mean(left_row[valid])
            right_centered = right_row[valid] - np.mean(right_row[valid])
            denominator = np.sqrt(
                np.sum(left_centered * left_centered) * np.sum(right_centered * right_centered)
            )
            expected[row_index] = (
                np.sum(left_centered * right_centered) / denominator if denominator > 0 else 0.0
            )
        np.clip(expected, -1.0, 1.0, out=expected)
        observed_shapes: list[tuple[int, ...]] = []
        original_isnan = np.isnan

        def small_row_slices(n_rows: int, n_columns: int) -> Iterator[tuple[int, int]]:
            del n_columns
            for start in range(0, n_rows, 17):
                yield start, min(start + 17, n_rows)

        def observed_isnan(values: np.ndarray) -> np.ndarray:
            observed_shapes.append(values.shape)
            return original_isnan(values)

        with (
            patch("ier._correlation.row_slices", side_effect=small_row_slices) as slices,
            patch("ier._correlation.np.isnan", side_effect=observed_isnan),
        ):
            result = calculate_correlations(left, right)

        self.assertGreaterEqual(slices.call_count, 2)
        self.assertTrue(observed_shapes)
        self.assertTrue(all(rows <= 17 and columns == 30 for rows, columns in observed_shapes))
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=2e-15, equal_nan=True)
        np.testing.assert_array_equal(left, original_left)
        np.testing.assert_array_equal(right, original_right)

    def test_two_point_correlations_use_exact_difference_signs(self) -> None:
        """Two-pair rows avoid centered matrices while preserving edge cases."""
        maximum = np.finfo(float).max
        even_cols = np.array(
            [
                [1.0, 3.0],
                [3.0, 1.0],
                [2.0, 2.0],
                [1.0, np.nan],
                [maximum, -maximum],
            ]
        )
        odd_cols = np.array(
            [
                [2.0, 4.0],
                [2.0, 4.0],
                [1.0, 4.0],
                [2.0, 3.0],
                [-maximum, maximum],
            ]
        )

        with patch(
            "ier._correlation.np.mean",
            side_effect=AssertionError("centered correlation path was used"),
        ):
            result = calculate_correlations(even_cols, odd_cols)

        np.testing.assert_array_equal(result, [1.0, -1.0, 0.0, np.nan, -1.0])

    def test_two_point_zero_variance_policy_is_preserved(self) -> None:
        """Callers can retain unavailable scores for constant two-pair rows."""
        from ier._correlation import row_correlations

        left = np.array([[1.0, 1.0], [1.0, 2.0]])
        right = np.array([[1.0, 2.0], [3.0, 3.0]])

        result = row_correlations(left, right, zero_variance=np.nan)

        self.assertTrue(np.isnan(result).all())

    def test_two_point_infinities_retain_centered_correlation_behavior(self) -> None:
        """Non-finite extremes remain on the established centered path."""
        even_cols = np.array(
            [
                [np.inf, 1.0],
                [np.inf, np.inf],
                [np.inf, np.nan],
                [-np.inf, 0.0],
            ]
        )
        odd_cols = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])

        result = calculate_correlations(even_cols, odd_cols)

        np.testing.assert_array_equal(result, [0.0, 0.0, np.nan, 0.0])

    def test_with_nan_values(self) -> None:
        """Test correlation calculation handles NaN values correctly."""
        even_cols: npt.NDArray[np.float64] = np.array([[1, np.nan, 5], [2, 4, 6]])
        odd_cols: npt.NDArray[np.float64] = np.array([[2, 4, 6], [3, 5, 7]])
        result: npt.NDArray[np.float64] = calculate_correlations(even_cols, odd_cols)
        self.assertEqual(len(result), 2)

    def test_mismatched_rows_raises_error(self) -> None:
        """Test that mismatched row counts raise ValueError."""
        even_cols: npt.NDArray[np.float64] = np.array([[1, 2, 3]])
        odd_cols: npt.NDArray[np.float64] = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            calculate_correlations(even_cols, odd_cols)

    def test_empty_columns(self) -> None:
        """Test correlation with empty column arrays."""
        even_cols: npt.NDArray[np.float64] = np.array([[1], [2]]).reshape(2, 1)[:, :0]
        odd_cols: npt.NDArray[np.float64] = np.array([[1], [2]]).reshape(2, 1)[:, :0]
        result: npt.NDArray[np.float64] = calculate_correlations(even_cols, odd_cols)
        self.assertEqual(len(result), 2)
        self.assertTrue(np.all(np.isnan(result)))

    def test_insufficient_valid_pairs(self) -> None:
        """Test correlation returns NaN when insufficient valid pairs exist."""
        even_cols: npt.NDArray[np.float64] = np.array([[1, np.nan, np.nan]])
        odd_cols: npt.NDArray[np.float64] = np.array([[np.nan, np.nan, 3]])
        result: npt.NDArray[np.float64] = calculate_correlations(even_cols, odd_cols)
        self.assertTrue(np.isnan(result[0]))


class TestPsychometricFunctions(unittest.TestCase):
    """Test suite for psychometric synonym/antonym detection functions.

    Tests psychsyn scoring, critical value calculation, psychant function,
    resampling for missing values, and summary statistics.
    """

    def setUp(self) -> None:
        """Initialize test data with simple progressive values."""
        self.data: npt.NDArray[np.float64] = np.array(
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
        )

    @staticmethod
    def _pairwise_item_correlation_reference(data: np.ndarray) -> np.ndarray:
        """Build a scalar pairwise-complete item correlation reference."""
        n_items = data.shape[1]
        correlations = np.full((n_items, n_items), np.nan)
        for row in range(n_items):
            for column in range(n_items):
                valid = np.isfinite(data[:, row]) & np.isfinite(data[:, column])
                if np.count_nonzero(valid) < 2:
                    continue
                left = data[valid, row]
                right = data[valid, column]
                if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
                    continue
                left = left - left[0]
                right = right - right[0]
                correlations[row, column] = np.corrcoef(left, right)[0, 1]
        return correlations

    def test_psychsyn_basic(self) -> None:
        """Test basic psychometric synonym scoring."""
        scores: npt.NDArray[np.float64] = psychsyn(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_with_list_input(self) -> None:
        """Test psychsyn function accepts list input."""
        data: list[list[int]] = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
        scores: npt.NDArray[np.float64] = psychsyn(data)
        self.assertEqual(len(scores), len(data))

    def test_psychsyn_diag(self) -> None:
        """Test psychsyn returns diagnostic values when diag=True."""
        scores: npt.NDArray[np.float64]
        diag_vals: npt.NDArray[np.float64]
        scores, diag_vals = psychsyn(self.data, diag=True)
        self.assertEqual(len(scores), self.data.shape[0])
        self.assertEqual(len(diag_vals), self.data.shape[0])

    def test_psychsyn_resample_na(self) -> None:
        """Test psychsyn resamples correlations for missing values."""
        self.data = self.data.astype(float)
        self.data[0, 0] = np.nan
        scores: npt.NDArray[np.float64] = psychsyn(self.data, resample_na=True)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_with_random_seed(self) -> None:
        """Test psychsyn produces reproducible results with random seed."""
        scores1: npt.NDArray[np.float64] = psychsyn(self.data, resample_na=True, random_seed=42)
        scores2: npt.NDArray[np.float64] = psychsyn(self.data, resample_na=True, random_seed=42)
        np.testing.assert_array_equal(scores1, scores2)

    def test_psychsyn_random_seed_does_not_mutate_global_rng(self) -> None:
        """Test psychsyn random_seed does not affect global numpy RNG state."""
        data = self.data.astype(float).copy()
        data[0, 0] = np.nan

        np.random.seed(123)
        expected_next = np.random.random()

        np.random.seed(123)
        psychsyn(data, resample_na=True, random_seed=7)
        actual_next = np.random.random()

        self.assertAlmostEqual(expected_next, actual_next)

    def test_psychsyn_critval(self) -> None:
        """Test critical value calculation returns tuples with indices and correlations."""
        results: list[tuple[int, int, float]] = psychsyn_critval(self.data)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 3 for t in results))

    def test_psychsyn_critval_with_min_correlation(self) -> None:
        """Test critical value filtering respects minimum correlation threshold."""
        results: list[tuple[int, int, float]] = psychsyn_critval(self.data, min_correlation=0.5)
        self.assertTrue(all(abs(t[2]) >= 0.5 for t in results))

    def test_bounded_pair_discovery_matches_correlation_matrix_reference(self) -> None:
        """Triangular blocks preserve pairwise-complete thresholds and order."""
        rng = np.random.default_rng(20260803)
        data = rng.normal(size=(53, 17))
        data[:, 2] = 1.0
        data[0, 5] = np.nan
        data[1, 8] = np.nan
        data[2, 11] = np.inf
        correlations = self._pairwise_item_correlation_reference(data)

        for critval, anto in ((0.2, False), (-0.2, True), (0.0, False), (0.0, True)):
            with self.subTest(critval=critval, anto=anto):
                expected = get_highly_correlated_pairs(correlations, critval, anto)
                actual = _discover_item_pairs(data, critval, anto)
                np.testing.assert_array_equal(actual, expected)

    def test_pair_discovery_excludes_undefined_zero_threshold_pairs(self) -> None:
        """Undefined constant or nonoverlapping pairs never match a zero cutoff."""
        data = np.array(
            [
                [1.0, np.nan, 2.0],
                [2.0, np.nan, 2.0],
                [np.nan, 1.0, 2.0],
                [np.nan, 2.0, 2.0],
            ]
        )

        self.assertEqual(len(_discover_item_pairs(data, 0.0, False)), 0)
        self.assertEqual(len(_discover_item_pairs(data, 0.0, True)), 0)

    def test_critical_value_catalog_matches_correlation_matrix_reference(self) -> None:
        """Blockwise correlation catalogs retain values and ranking direction."""
        rng = np.random.default_rng(37)
        data = rng.normal(size=(71, 13))
        data[:, 4] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.corrcoef(data, rowvar=False)
        rows, columns = np.triu_indices(data.shape[1], k=1)
        values = correlations[rows, columns]
        selected = ~np.isnan(values) & (np.abs(values) >= 0.1)

        for anto in (False, True):
            with self.subTest(anto=anto):
                order = np.argsort(values[selected]) if anto else np.argsort(-values[selected])
                expected = [
                    (
                        int(rows[selected][index]),
                        int(columns[selected][index]),
                        values[selected][index],
                    )
                    for index in order
                ]
                actual = psychsyn_critval(data, anto=anto, min_correlation=0.1)
                self.assertEqual(
                    [(left, right) for left, right, _ in actual],
                    [(left, right) for left, right, _ in expected],
                )
                np.testing.assert_allclose(
                    [value for _, _, value in actual],
                    [value for _, _, value in expected],
                    rtol=0.0,
                    atol=5e-16,
                )

    def test_missing_critical_value_catalog_uses_pairwise_complete_values(self) -> None:
        """The public cutoff catalog retains partially observed item correlations."""
        rng = np.random.default_rng(39)
        data = rng.normal(size=(43, 9))
        data[rng.random(data.shape) < 0.15] = np.nan
        correlations = self._pairwise_item_correlation_reference(data)
        rows, columns = np.triu_indices(data.shape[1], k=1)
        values = correlations[rows, columns]
        selected = np.isfinite(values) & (np.abs(values) >= 0.1)

        for anto in (False, True):
            with self.subTest(anto=anto):
                order = np.argsort(values[selected]) if anto else np.argsort(-values[selected])
                actual = psychsyn_critval(data, anto=anto, min_correlation=0.1)
                np.testing.assert_array_equal(
                    [(left, right) for left, right, _ in actual],
                    np.column_stack((rows[selected], columns[selected]))[order],
                )
                np.testing.assert_allclose(
                    [value for _, _, value in actual],
                    values[selected][order],
                    rtol=0.0,
                    atol=1e-15,
                )

    def test_item_correlation_tiles_obey_forced_workspace_bound(self) -> None:
        """Forced tiny blocks cover each lower-triangle pair exactly once."""
        rng = np.random.default_rng(41)
        data = rng.normal(size=(5, 17))

        with patch("ier.psychsyn._PSYCHSYN_CORRELATION_BLOCK_ELEMENTS", 25):
            item_offsets, item_norms, valid_columns = _complete_item_normalization(data)
            tiles = list(
                _iter_item_correlation_tiles(
                    data,
                    item_offsets,
                    item_norms,
                    valid_columns,
                )
            )

        expected_correlations = np.corrcoef(data, rowvar=False)
        self.assertTrue(all(len(values) <= 25 for _, _, values in tiles))
        for rows, columns, values in tiles:
            np.testing.assert_allclose(
                values,
                expected_correlations[rows, columns],
                rtol=0.0,
                atol=5e-16,
            )
        pairs = np.concatenate([np.column_stack((rows, columns)) for rows, columns, _ in tiles])
        expected = np.column_stack(np.tril_indices(data.shape[1], k=-1))
        np.testing.assert_array_equal(
            pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))],
            expected,
        )

    def test_complete_item_normalization_never_allocates_a_cohort_matrix(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(37, 19))
        allocated_shapes: list[tuple[int, ...]] = []
        real_array = np.array

        def tracked_array(*args: object, **kwargs: object) -> np.ndarray:
            result = real_array(*args, **kwargs)
            if result.ndim > 1:
                allocated_shapes.append(result.shape)
            return result

        with (
            patch("ier.psychsyn._PSYCHSYN_CORRELATION_BLOCK_ELEMENTS", 36),
            patch("ier.psychsyn.np.array", side_effect=tracked_array),
        ):
            item_offsets, item_norms, valid_columns = _complete_item_normalization(data)

        self.assertTrue(allocated_shapes)
        self.assertNotIn(data.shape, allocated_shapes)
        self.assertTrue(
            all(np.prod(shape) <= max(36, len(data)) for shape in allocated_shapes),
            allocated_shapes,
        )

        with (
            patch("ier.psychsyn._PSYCHSYN_CORRELATION_BLOCK_ELEMENTS", 36),
            patch(
                "ier.psychsyn._normalized_complete_item_block",
                wraps=_normalized_complete_item_block,
            ) as normalize_block,
        ):
            list(
                _iter_item_correlation_tiles(
                    data,
                    item_offsets,
                    item_norms,
                    valid_columns,
                )
            )

        for call in normalize_block.call_args_list:
            row_start, row_stop, column_start, column_stop = call.args[-4:]
            self.assertLessEqual(
                (row_stop - row_start) * (column_stop - column_start),
                36,
            )

    def test_pairwise_item_correlation_tiles_match_scalar_reference(self) -> None:
        """Missing-data tiles match scalar pairwise correlations under tiny bounds."""
        rng = np.random.default_rng(43)
        data = rng.normal(size=(31, 13))
        data[rng.random(data.shape) < 0.12] = np.nan
        data[:, 4] = 2.0
        expected = self._pairwise_item_correlation_reference(data)

        with patch("ier.psychsyn._PSYCHSYN_CORRELATION_BLOCK_ELEMENTS", 25):
            tiles = list(_iter_pairwise_item_correlation_tiles(data))

        for rows, columns, values in tiles:
            np.testing.assert_allclose(
                values,
                expected[rows, columns],
                rtol=0.0,
                atol=1e-15,
                equal_nan=True,
            )
        pairs = np.concatenate([np.column_stack((rows, columns)) for rows, columns, _ in tiles])
        expected_pairs = np.column_stack(np.tril_indices(data.shape[1], k=-1))
        np.testing.assert_array_equal(
            pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))],
            expected_pairs,
        )

    def test_pairwise_item_correlations_are_stable_for_large_offsets(self) -> None:
        """Column offsets prevent raw-moment cancellation with missing values."""
        rng = np.random.default_rng(45)
        data = 1e12 + rng.normal(size=(47, 9))
        data[rng.random(data.shape) < 0.1] = np.nan
        expected = self._pairwise_item_correlation_reference(data)

        with patch("ier.psychsyn._PSYCHSYN_CORRELATION_BLOCK_ELEMENTS", 36):
            tiles = list(_iter_pairwise_item_correlation_tiles(data))

        for rows, columns, values in tiles:
            np.testing.assert_allclose(
                values,
                expected[rows, columns],
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )

    def test_public_psychsyn_retains_pairwise_complete_items_and_counts(self) -> None:
        """Scattered omissions retain discoverable items and usable respondent pairs."""
        rng = np.random.default_rng(47)
        latent = rng.normal(size=(80, 1))
        data = latent + rng.normal(scale=0.05, size=(80, 8))
        for item in range(data.shape[1]):
            data[item, item] = np.nan
        original = data.copy()

        scores, pair_counts, item_pairs = psychsyn(
            data,
            critval=0.8,
            diag=True,
            resample_na=False,
            _return_item_info=True,
        )
        expected_pairs = np.column_stack(np.tril_indices(data.shape[1], k=-1))
        np.testing.assert_array_equal(item_pairs, expected_pairs)

        expected_scores = np.empty(len(data))
        expected_counts = np.empty(len(data), dtype=int)
        for index, row in enumerate(data):
            left = row[item_pairs[:, 0]]
            right = row[item_pairs[:, 1]]
            valid = np.isfinite(left) & np.isfinite(right)
            expected_counts[index] = np.count_nonzero(valid)
            expected_scores[index] = np.corrcoef(left[valid], right[valid])[0, 1]

        np.testing.assert_allclose(scores, expected_scores, rtol=0.0, atol=2e-15)
        np.testing.assert_array_equal(pair_counts, expected_counts)
        self.assertTrue(np.all(pair_counts[:8] < len(item_pairs)))
        np.testing.assert_array_equal(data, original)

    def test_public_pair_discovery_does_not_build_a_square_matrix(self) -> None:
        """Both public discovery paths stay behind the bounded block implementation."""
        with patch(
            "ier.psychsyn.np.corrcoef",
            side_effect=AssertionError("full item correlation matrix was constructed"),
        ):
            psychsyn(self.data)
            psychsyn_critval(self.data)

    def test_psychsyn_anto(self) -> None:
        """Test psychsyn with antonym detection mode enabled."""
        scores: npt.NDArray[np.float64] = psychsyn(self.data, anto=True, critval=-0.6)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant(self) -> None:
        """Test psychant convenience function for antonym detection."""
        scores: npt.NDArray[np.float64] = psychant(self.data)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychant_with_resample_na(self) -> None:
        """Test psychant handles missing value resampling."""
        scores: npt.NDArray[np.float64] = psychant(self.data, resample_na=True, random_seed=42)
        self.assertEqual(len(scores), self.data.shape[0])

    def test_psychsyn_summary(self) -> None:
        """Test psychsyn summary returns expected statistics dictionary."""
        summary: PsychsynSummary = psychsyn_summary(self.data)
        self.assertEqual(
            set(summary),
            {
                "mean_score",
                "std_score",
                "min_score",
                "max_score",
                "median_score",
                "item_pairs",
                "total_individuals",
                "valid_individuals",
                "missing_individuals",
            },
        )
        self.assertIn("mean_score", summary)
        self.assertIn("std_score", summary)
        self.assertIn("item_pairs", summary)
        self.assertIn("total_individuals", summary)

    def test_psychsyn_summary_handles_no_available_pairs(self) -> None:
        """No qualifying item pairs produce defined coverage without warnings."""
        data = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]])

        summary = psychsyn_summary(data, critval=0.99)

        self.assertEqual(summary["item_pairs"], 0)
        self.assertEqual(summary["total_individuals"], 3)
        self.assertEqual(summary["valid_individuals"], 0)
        self.assertEqual(summary["missing_individuals"], 3)
        statistics = np.array(
            [
                summary["mean_score"],
                summary["std_score"],
                summary["min_score"],
                summary["max_score"],
                summary["median_score"],
            ]
        )
        self.assertTrue(np.all(np.isnan(statistics)))

    def test_psychsyn_validation(self) -> None:
        """Test psychsyn raises appropriate errors for invalid inputs."""
        with self.assertRaises(ValueError):
            psychsyn(None)
        with self.assertRaises(ValueError):
            psychsyn([])
        with self.assertRaises(ValueError):
            psychsyn([[1]])
        with self.assertRaises(ValueError):
            psychsyn(self.data, critval=cast("float", "high"))
        with self.assertRaises(ValueError):
            psychsyn(self.data, critval=-0.5, anto=False)
        with self.assertRaises(ValueError):
            psychsyn([[1, 2], [3, 4]], critval=0.5, anto=True)

    def test_psychsyn_no_pairs_diagnostics_and_item_info(self) -> None:
        """Test no-pair branches return diagnostics and item-pair metadata."""
        data = [[1, 2, 3], [2, 4, 1], [3, 1, 4], [4, 3, 2]]
        scores, diag = psychsyn(data, critval=1.1, diag=True)
        self.assertTrue(np.all(np.isnan(scores)))
        self.assertTrue(np.all(diag == 0))

        info_scores, info_diag, pairs = psychsyn(data, critval=1.1, _return_item_info=True)
        self.assertTrue(np.all(np.isnan(info_scores)))
        self.assertTrue(np.all(info_diag == 0))
        self.assertEqual(pairs.shape[0], 0)

    def test_complete_psychsyn_batches_match_expanded_formula(self) -> None:
        """Finite response batches preserve the expanded pairwise formula."""
        rng = np.random.default_rng(20260802)
        latent = rng.normal(size=(257, 1))
        offsets = np.linspace(-1.0, 1.0, 20)
        data = latent + offsets + rng.normal(scale=0.05, size=(257, 20))
        correlations = np.corrcoef(data, rowvar=False)
        item_pairs = get_highly_correlated_pairs(correlations, critval=0.8, anto=False)

        response_i = data[:, item_pairs[:, 0]]
        response_j = data[:, item_pairs[:, 1]]
        person_corrs = compute_person_correlations(response_i, response_j)
        expected_scores = np.mean(person_corrs, axis=1)
        expected_diag = np.sum(~np.isnan(person_corrs), axis=1)

        with patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", 760):
            scores, diag = _compute_complete_person_scores(data, item_pairs)
            public_scores, public_diag = psychsyn(data, critval=0.8, diag=True)

        np.testing.assert_allclose(scores, expected_scores, rtol=0.0, atol=2e-15)
        np.testing.assert_array_equal(diag, expected_diag)
        np.testing.assert_allclose(public_scores, expected_scores, rtol=0.0, atol=2e-15)
        np.testing.assert_array_equal(public_diag, expected_diag)

    def test_psychsyn_requires_three_available_pairs(self) -> None:
        """Fewer than three item pairs remain diagnostically visible but unscored."""
        data = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        item_pairs = np.array([[0, 1], [0, 2]])

        scores, counts = _compute_complete_person_scores(data, item_pairs)

        self.assertTrue(np.isnan(scores).all())
        np.testing.assert_array_equal(counts, [2, 2])

    def test_missing_psychsyn_batches_match_expanded_formula(self) -> None:
        """Missing batches use every finite pair and preserve input values."""
        from ier._correlation import row_correlations

        rng = np.random.default_rng(20260802)
        latent = rng.normal(size=(71, 1))
        data = latent + rng.normal(scale=0.2, size=(71, 10))
        data[::7, 0] = np.nan
        data[::11, 3] = np.nan
        item_pairs = np.column_stack(np.tril_indices(data.shape[1], k=-1))

        response_i = data[:, item_pairs[:, 0]]
        response_j = data[:, item_pairs[:, 1]]
        valid = np.isfinite(response_i) & np.isfinite(response_j)
        expected_diag = np.sum(valid, axis=1)
        response_i[~valid] = np.nan
        response_j[~valid] = np.nan
        expected_scores = row_correlations(
            response_i,
            response_j,
            zero_variance=np.nan,
        )
        expected_scores[expected_diag < 3] = np.nan
        original = data.copy()

        with (
            patch("ier.psychsyn._PSYCHSYN_BATCH_ELEMENTS", len(item_pairs)),
            patch(
                "ier.psychsyn.compute_person_correlations",
                side_effect=AssertionError("expanded pair contributions were constructed"),
            ),
            patch("ier.psychsyn.row_correlations", wraps=row_correlations) as contracted,
        ):
            scores, diag = _compute_person_scores(
                data,
                item_pairs,
                resample_na=False,
                rng=np.random.default_rng(42),
            )
        self.assertGreater(contracted.call_count, 1)
        self.assertTrue(
            all(call.args[0].size <= len(item_pairs) for call in contracted.call_args_list)
        )
        np.testing.assert_allclose(scores, expected_scores, rtol=0.0, atol=2e-15, equal_nan=True)
        np.testing.assert_array_equal(diag, expected_diag)
        np.testing.assert_array_equal(data, original)

    def test_missing_psychsyn_resampling_retries_only_eligible_rows(self) -> None:
        """Seeded pair-direction retries preserve pair counts and insufficient rows."""
        data = np.array(
            [
                [1.0, 1.0, 2.0, 3.0],
                [1.0, np.nan, 2.0, 3.0],
            ]
        )
        item_pairs = np.array([[0, 1], [0, 2], [0, 3]])

        plain_scores, plain_counts = _compute_person_scores(
            data,
            item_pairs,
            resample_na=False,
            rng=np.random.default_rng(42),
        )
        first_scores, first_counts = _compute_person_scores(
            data,
            item_pairs,
            resample_na=True,
            rng=np.random.default_rng(42),
        )
        second_scores, second_counts = _compute_person_scores(
            data,
            item_pairs,
            resample_na=True,
            rng=np.random.default_rng(42),
        )

        self.assertTrue(np.isnan(plain_scores).all())
        self.assertTrue(np.isfinite(first_scores[0]))
        self.assertTrue(np.isnan(first_scores[1]))
        np.testing.assert_array_equal(first_scores, second_scores)
        np.testing.assert_array_equal(plain_counts, [3, 2])
        np.testing.assert_array_equal(first_counts, plain_counts)
        np.testing.assert_array_equal(second_counts, plain_counts)

    def test_psychsyn_edge_cases(self) -> None:
        """Test psychsyn handles edge cases like high thresholds and constant data."""
        data: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]])
        scores: npt.NDArray[np.float64] = psychsyn(data, critval=0.99)
        self.assertTrue(np.all(np.isnan(scores)) or np.all(scores == 0))

        data = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        scores = psychsyn(data, critval=0.5)
        self.assertEqual(len(scores), 3)


class TestGetHighlyCorrelatedPairs(unittest.TestCase):
    """Test suite for the get_highly_correlated_pairs helper function."""

    def test_finds_positive_correlations(self) -> None:
        """Test finding positively correlated item pairs."""
        corr_matrix: npt.NDArray[np.float64] = np.array(
            [[1.0, 0.8, 0.3], [0.8, 1.0, 0.2], [0.3, 0.2, 1.0]]
        )
        pairs: npt.NDArray[np.intp] = get_highly_correlated_pairs(
            corr_matrix, critval=0.7, anto=False
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue((pairs[0] == [1, 0]).all())

    def test_finds_negative_correlations(self) -> None:
        """Test finding negatively correlated item pairs (antonyms)."""
        corr_matrix: npt.NDArray[np.float64] = np.array(
            [[1.0, -0.8, 0.3], [-0.8, 1.0, 0.2], [0.3, 0.2, 1.0]]
        )
        pairs: npt.NDArray[np.intp] = get_highly_correlated_pairs(
            corr_matrix, critval=-0.7, anto=True
        )
        self.assertEqual(len(pairs), 1)
        self.assertTrue((pairs[0] == [1, 0]).all())

    def test_no_pairs_above_threshold(self) -> None:
        """Test returns empty array when no pairs meet threshold."""
        corr_matrix: npt.NDArray[np.float64] = np.array(
            [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]]
        )
        pairs: npt.NDArray[np.intp] = get_highly_correlated_pairs(
            corr_matrix, critval=0.9, anto=False
        )
        self.assertEqual(len(pairs), 0)

    def test_multiple_pairs(self) -> None:
        """Test finding multiple correlated pairs."""
        corr_matrix: npt.NDArray[np.float64] = np.array(
            [[1.0, 0.9, 0.85], [0.9, 1.0, 0.88], [0.85, 0.88, 1.0]]
        )
        pairs: npt.NDArray[np.intp] = get_highly_correlated_pairs(
            corr_matrix, critval=0.8, anto=False
        )
        self.assertEqual(len(pairs), 3)

    def test_excludes_diagonal(self) -> None:
        """Test that diagonal elements are not included as pairs."""
        corr_matrix: npt.NDArray[np.float64] = np.array([[1.0, 0.5], [0.5, 1.0]])
        pairs: npt.NDArray[np.intp] = get_highly_correlated_pairs(
            corr_matrix, critval=0.99, anto=False
        )
        self.assertEqual(len(pairs), 0)

    def test_zero_threshold_only_selects_strict_lower_triangle(self) -> None:
        """Zero thresholds do not admit diagonal or mirrored item pairs."""
        corr_matrix = np.array(
            [
                [1.0, 0.0, -0.2],
                [0.0, 1.0, 0.3],
                [-0.2, 0.3, 1.0],
            ]
        )

        synonyms = get_highly_correlated_pairs(corr_matrix, critval=0.0, anto=False)
        antonyms = get_highly_correlated_pairs(corr_matrix, critval=0.0, anto=True)

        np.testing.assert_array_equal(synonyms, [[1, 0], [2, 1]])
        np.testing.assert_array_equal(antonyms, [[1, 0], [2, 0]])


class TestComputePersonCorrelations(unittest.TestCase):
    """Test suite for the compute_person_correlations helper function."""

    def test_perfect_positive_correlation(self) -> None:
        """Test computation with perfectly correlated responses."""
        response_i: npt.NDArray[np.float64] = np.array([[1, 2, 3], [4, 5, 6]])
        response_j: npt.NDArray[np.float64] = np.array([[2, 4, 6], [8, 10, 12]])
        result: npt.NDArray[np.float64] = compute_person_correlations(response_i, response_j)
        self.assertEqual(result.shape, (2, 3))

    def test_empty_input(self) -> None:
        """Test with empty input arrays."""
        response_i: npt.NDArray[np.float64] = np.array([]).reshape(0, 3)
        response_j: npt.NDArray[np.float64] = np.array([]).reshape(0, 3)
        result: npt.NDArray[np.float64] = compute_person_correlations(response_i, response_j)
        self.assertEqual(len(result), 0)

    def test_single_person(self) -> None:
        """Test correlation computation for single person."""
        response_i: npt.NDArray[np.float64] = np.array([[1, 2, 3, 4]])
        response_j: npt.NDArray[np.float64] = np.array([[2, 3, 4, 5]])
        result: npt.NDArray[np.float64] = compute_person_correlations(response_i, response_j)
        self.assertEqual(result.shape[0], 1)

    def test_zero_std_handling(self) -> None:
        """Test handling of zero standard deviation (constant responses)."""
        response_i: npt.NDArray[np.float64] = np.array([[5, 5, 5, 5]])
        response_j: npt.NDArray[np.float64] = np.array([[3, 3, 3, 3]])
        result: npt.NDArray[np.float64] = compute_person_correlations(response_i, response_j)
        self.assertEqual(result.shape[0], 1)

    def test_varying_correlations(self) -> None:
        """Test with data producing varying correlations across persons."""
        response_i: npt.NDArray[np.float64] = np.array([[1, 2, 3], [3, 2, 1]])
        response_j: npt.NDArray[np.float64] = np.array([[1, 2, 3], [1, 2, 3]])
        result: npt.NDArray[np.float64] = compute_person_correlations(response_i, response_j)
        self.assertEqual(result.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
