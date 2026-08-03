"""Tests for missing-response diagnostics and orchestration."""

import unittest
from typing import Any, cast

import numpy as np

from ier import IndexOptions, composite, missing_rate, missing_rate_flag, screen


class TestMissingRate(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, np.nan, 3.0, 4.0],
                [np.nan, np.nan, 3.0, 4.0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )

    def test_rates_cover_complete_partial_and_fully_missing_rows(self) -> None:
        np.testing.assert_array_equal(missing_rate(self.data), [0.0, 0.25, 0.5, 1.0])

    def test_item_subset_uses_only_selected_columns(self) -> None:
        np.testing.assert_array_equal(
            missing_rate(self.data, item_indices=[0, 2]),
            [0.0, 0.0, 0.5, 1.0],
        )

    def test_applicability_mask_excludes_planned_omissions(self) -> None:
        applicable = np.array(
            [
                [True, False, True, False],
                [True, False, True, False],
                [True, True, False, False],
                [False, False, False, False],
            ]
        )

        np.testing.assert_allclose(
            missing_rate(self.data, applicable_mask=applicable),
            [0.0, 0.0, 1.0, np.nan],
            equal_nan=True,
        )

    def test_applicability_mask_combines_with_item_selection(self) -> None:
        applicable = np.array(
            [
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, True],
                [False, True, False, True],
            ]
        )

        np.testing.assert_allclose(
            missing_rate(self.data, item_indices=[0, 2], applicable_mask=applicable),
            [0.0, 0.0, 0.5, np.nan],
            equal_nan=True,
        )

    def test_item_selection_validation(self) -> None:
        cases: list[tuple[Any, str]] = [
            ([], "cannot be empty"),
            ([0, 0], "duplicates"),
            ([-1], "out of bounds"),
            ([4], "out of bounds"),
            ([True], "integer column indices"),
            ([1.5], "integer column indices"),
        ]
        for item_indices, message in cases:
            with (
                self.subTest(item_indices=item_indices),
                self.assertRaisesRegex(ValueError, message),
            ):
                missing_rate(self.data, item_indices=cast("Any", item_indices))

    def test_applicability_mask_validation(self) -> None:
        cases: list[tuple[Any, str]] = [
            ([[True, False]], "shape"),
            (np.ones(self.data.shape, dtype=int), "boolean values"),
            ([[True], [False, True]], "rectangular boolean matrix"),
        ]
        for applicable_mask, message in cases:
            with (
                self.subTest(applicable_mask=applicable_mask),
                self.assertRaisesRegex(ValueError, message),
            ):
                missing_rate(self.data, applicable_mask=cast("Any", applicable_mask))

    def test_fixed_threshold_is_inclusive(self) -> None:
        scores, flags = missing_rate_flag(self.data, threshold=0.5)
        np.testing.assert_array_equal(scores, [0.0, 0.25, 0.5, 1.0])
        np.testing.assert_array_equal(flags, [False, False, True, True])

    def test_percentile_threshold_flags_only_strict_extremes(self) -> None:
        tied_data = np.array([[1.0, 2.0], [np.nan, 2.0], [1.0, np.nan], [np.nan, np.nan]])
        _, flags = missing_rate_flag(tied_data, percentile=50.0)
        np.testing.assert_array_equal(flags, [False, False, False, True])

    def test_flagging_excludes_rows_without_applicable_items(self) -> None:
        applicable = np.array(
            [
                [True, True, True, True],
                [True, False, True, True],
                [True, False, True, True],
                [False, False, False, False],
            ]
        )
        scores, flags = missing_rate_flag(
            self.data,
            threshold=0.25,
            applicable_mask=applicable,
        )

        np.testing.assert_allclose(scores, [0.0, 0.0, 1 / 3, np.nan], equal_nan=True)
        np.testing.assert_array_equal(flags, [False, False, True, False])

    def test_flagging_validation(self) -> None:
        for threshold in [False, -0.1, 1.1, np.nan, np.inf]:
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                missing_rate_flag(self.data, threshold=cast("Any", threshold))
        for percentile in [False, -0.1, 100.1, np.nan, np.inf]:
            with (
                self.subTest(percentile=percentile),
                self.assertRaisesRegex(ValueError, "percentile"),
            ):
                missing_rate_flag(self.data, percentile=cast("Any", percentile))

    def test_screen_supports_fixed_missing_rate_thresholds(self) -> None:
        result = screen(
            self.data,
            indices=["missing_rate"],
            thresholds={"missing_rate": 0.5},
            min_flags=1,
        )
        np.testing.assert_array_equal(result["scores"]["missing_rate"], [0.0, 0.25, 0.5, 1.0])
        np.testing.assert_array_equal(result["flags"]["missing_rate"], [False, False, True, True])
        np.testing.assert_array_equal(result["consensus_flags"], [False, False, True, True])

    def test_registry_missing_options_apply_to_screen_and_composite(self) -> None:
        applicable = np.array(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, False, True, True],
                [False, False, False, False],
            ]
        )
        options = IndexOptions(
            missing_item_indices=[0, 2],
            missing_applicable_mask=applicable,
        )

        screened = screen(
            self.data,
            indices=["missing_rate"],
            options=options,
            thresholds={"missing_rate": 0.5},
            min_flags=1,
        )
        combined = composite(
            self.data,
            indices=["missing_rate"],
            options=options,
            standardize=False,
        )

        expected = [0.0, 0.0, 0.5, np.nan]
        np.testing.assert_allclose(screened["scores"]["missing_rate"], expected, equal_nan=True)
        np.testing.assert_array_equal(
            screened["flags"]["missing_rate"],
            [False, False, True, False],
        )
        np.testing.assert_allclose(combined, expected, equal_nan=True)

    def test_composite_accepts_missing_rate_without_standardization(self) -> None:
        result = composite(self.data, indices=["missing_rate"], standardize=False)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0.0, 0.25, 0.5, 1.0])
