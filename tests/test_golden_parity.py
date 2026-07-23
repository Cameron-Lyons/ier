"""Golden / R-parity style fixtures for core pattern indices.

These expected values are hand-verified against the same definitions used by
common R careless-responding tooling (e.g. ``careless::irv`` /
``careless::longstring``):

- IRV: within-person standard deviation across items (population ``ddof=0``,
  matching NumPy ``np.std`` / ``np.nanstd`` defaults).
- longstring: length of the longest run of identical consecutive responses.

They are regression fixtures for scientific credibility, not a claim of
bit-identical output against every CRAN release.
"""

from __future__ import annotations

import unittest

import numpy as np

from ier import IndexOptions, composite, irv, longstring_pattern, screen
from ier.longstring import longstring_scores

# Respondent × item matrix chosen so each row has an obvious analytic answer.
GOLDEN_MATRIX = np.array(
    [
        [1, 1, 1, 1, 1],  # straightline: IRV=0, longstring=5
        [1, 2, 3, 4, 5],  # ascending: IRV=std(1..5), longstring=1
        [5, 5, 5, 1, 2],  # early straightline: longstring=3
        [1, 2, 2, 2, 2],  # late straightline: longstring=4
        [1, 2, 1, 2, 1],  # alternating: longstring=1
    ],
    dtype=float,
)

EXPECTED_IRV = np.array(
    [
        0.0,
        float(np.std([1, 2, 3, 4, 5])),
        float(np.std([5, 5, 5, 1, 2])),
        float(np.std([1, 2, 2, 2, 2])),
        float(np.std([1, 2, 1, 2, 1])),
    ]
)

EXPECTED_LONGSTRING = np.array([5.0, 1.0, 3.0, 4.0, 1.0])


class TestGoldenIrvLongstring(unittest.TestCase):
    """Hand-verified IRV / longstring values for R-parity workflows."""

    def test_irv_matches_hand_calculation(self) -> None:
        scores = irv(GOLDEN_MATRIX)
        np.testing.assert_allclose(scores, EXPECTED_IRV, rtol=0, atol=1e-12)

    def test_longstring_scores_match_hand_calculation(self) -> None:
        scores = longstring_scores(GOLDEN_MATRIX)
        np.testing.assert_array_equal(scores, EXPECTED_LONGSTRING)

    def test_screen_options_object_uses_same_scores(self) -> None:
        result = screen(
            GOLDEN_MATRIX,
            indices=["irv", "longstring"],
            options=IndexOptions(),
        )
        np.testing.assert_allclose(result["scores"]["irv"], EXPECTED_IRV, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(result["scores"]["longstring"], EXPECTED_LONGSTRING)

    def test_longstring_pattern_detects_alternating(self) -> None:
        # Alternating 1-2-1-2-1 should score higher than a pure straightline
        # (straightlines are excluded from pattern detection by design).
        pattern = longstring_pattern(GOLDEN_MATRIX, max_pattern_length=3)
        self.assertGreater(pattern[4], pattern[0])
        self.assertGreater(pattern[4], pattern[1])


class TestIndexOptionsApi(unittest.TestCase):
    """IndexOptions is the preferred shared config surface."""

    def test_options_overrides_legacy_kwargs(self) -> None:
        # When options is provided, legacy kwargs must be ignored.
        result = screen(
            GOLDEN_MATRIX,
            indices=["u3_poly", "midpoint"],
            options=IndexOptions(scale_min=1, scale_max=5),
            scale_min=99,
            scale_max=100,
        )
        self.assertEqual(result["errors"], {})
        self.assertEqual(set(result["indices_used"]), {"u3_poly", "midpoint"})

    def test_composite_accepts_full_index_options(self) -> None:
        scores = composite(
            GOLDEN_MATRIX,
            indices=["irv", "longstring", "longstring_pattern"],
            options=IndexOptions(longstring_max_pattern_length=3),
        )
        self.assertEqual(len(scores), len(GOLDEN_MATRIX))
