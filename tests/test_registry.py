"""Tests for public index registry discovery."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from ier import index_catalog
from ier._registry import (
    INDEX_REGISTRY,
    IndexOptions,
    IndexSpec,
    score_registered_indices,
    validate_index_names,
)


class TestIndexCatalog(unittest.TestCase):
    def test_catalog_describes_all_registered_indices(self) -> None:
        catalog = index_catalog()

        self.assertEqual(len(catalog), 21)
        self.assertEqual(
            catalog["irv"],
            {
                "flag_direction": "low",
                "flag_mode": "percentile",
                "default_screen": True,
                "default_composite": True,
                "composite_enabled": True,
                "required_options": (),
            },
        )
        self.assertEqual(catalog["onset"]["flag_mode"], "present")
        self.assertFalse(catalog["onset"]["composite_enabled"])
        self.assertEqual(
            catalog["missing_rate"],
            {
                "flag_direction": "high",
                "flag_mode": "percentile",
                "default_screen": False,
                "default_composite": False,
                "composite_enabled": True,
                "required_options": (),
            },
        )
        self.assertEqual(catalog["evenodd"]["required_options"], ("evenodd_factors",))
        self.assertEqual(
            catalog["infrequency"]["required_options"],
            ("infrequency_item_indices", "infrequency_expected_responses"),
        )

    def test_catalog_returns_independent_metadata(self) -> None:
        catalog = index_catalog()
        catalog["irv"]["default_screen"] = False

        self.assertTrue(index_catalog()["irv"]["default_screen"])


class TestParallelIndexScoring(unittest.TestCase):
    def test_workers_run_concurrently_and_results_keep_selection_order(self) -> None:
        rendezvous = threading.Barrier(2)
        second_completed = threading.Event()

        def first_score(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del options
            rendezvous.wait(timeout=2)
            if not second_completed.wait(timeout=2):
                raise RuntimeError("second scorer did not run concurrently")
            return np.full(len(x), 1.0)

        def second_score(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del options
            rendezvous.wait(timeout=2)
            second_completed.set()
            return np.full(len(x), 2.0)

        additions = {
            "parallel_first": IndexSpec("parallel_first", first_score, "high"),
            "parallel_second": IndexSpec("parallel_second", second_score, "high"),
        }
        with patch.dict(INDEX_REGISTRY, additions):
            scores, errors = score_registered_indices(
                np.zeros((3, 2)),
                list(additions),
                IndexOptions(),
                workers=2,
            )

        self.assertEqual(list(scores), list(additions))
        np.testing.assert_array_equal(scores["parallel_first"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(scores["parallel_second"], [2.0, 2.0, 2.0])
        self.assertEqual(errors, {})

    def test_parallel_failures_retain_selection_order_and_strict_context(self) -> None:
        def broken_score(x: np.ndarray, options: IndexOptions) -> np.ndarray:
            del x, options
            raise RuntimeError("calculation failed")

        additions = {
            "parallel_missing": IndexSpec(
                "parallel_missing",
                lambda x, options: np.zeros(len(x)),
                "high",
                required_error=lambda options: "configuration missing",
            ),
            "parallel_broken": IndexSpec("parallel_broken", broken_score, "high"),
            "parallel_valid": IndexSpec(
                "parallel_valid",
                lambda x, options: np.arange(len(x), dtype=float),
                "high",
            ),
        }
        with patch.dict(INDEX_REGISTRY, additions):
            scores, errors = score_registered_indices(
                np.zeros((3, 2)),
                list(additions),
                IndexOptions(),
                workers=3,
            )
            with self.assertRaisesRegex(
                ValueError,
                "index 'parallel_missing' failed: configuration missing",
            ):
                score_registered_indices(
                    np.zeros((3, 2)),
                    list(additions),
                    IndexOptions(),
                    strict=True,
                    workers=3,
                )

        self.assertEqual(list(errors), ["parallel_missing", "parallel_broken"])
        self.assertEqual(list(scores), ["parallel_valid"])

    def test_workers_must_be_a_positive_integer(self) -> None:
        for workers in [0, -1, 1.5, True]:
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(
                    ValueError,
                    "workers must be a positive integer",
                ),
            ):
                score_registered_indices(
                    np.zeros((2, 2)),
                    ["irv"],
                    IndexOptions(),
                    workers=workers,  # type: ignore[arg-type]
                )

    def test_duplicate_indices_are_rejected_before_scoring(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate index 'irv'"):
            validate_index_names(["irv", "longstring", "irv"])

        with self.assertRaisesRegex(ValueError, "duplicate index 'irv'"):
            score_registered_indices(
                np.zeros((2, 2)),
                ["irv", "irv"],
                IndexOptions(),
                workers=2,
            )
