"""Golden / R-parity style fixtures for core pattern indices.

These expected values are hand-verified against the same definitions used by
common R careless-responding tooling (e.g. ``careless::irv`` /
``careless::longstring``):

- IRV: within-person standard deviation across items (population ``ddof=0``,
  matching NumPy ``np.std`` / ``np.nanstd`` defaults).
- longstring: length of the longest run of identical consecutive responses.
- longstring_pattern: longest consecutive repeating sub-pattern length.
- mahad (iqr path): Mahalanobis distances (NumPy-only; used by screen/composite).
- psychsyn: within-person synonym correlations for pairs above ``critval``.
- evenodd: mean even–odd consistency across factors.

They are regression fixtures for scientific credibility, not a claim of
bit-identical output against every CRAN release.
"""

from __future__ import annotations

import unittest

import numpy as np

from ier import (
    IndexOptions,
    composite,
    evenodd,
    irv,
    longstring_pattern,
    longstring_scores,
    mahad,
    screen,
)
from ier.psychsyn import psychsyn

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
EXPECTED_LONGSTRING_PATTERN = np.array([0.0, 0.0, 0.0, 0.0, 5.0])

# Broader matrix for multivariate / consistency indices.
PARITY_MATRIX = np.array(
    [
        [1, 2, 3, 4, 5, 4],
        [2, 3, 4, 5, 4, 3],
        [3, 3, 3, 3, 3, 3],
        [5, 1, 5, 1, 5, 1],
        [1, 1, 1, 1, 1, 1],
        [4, 4, 2, 2, 4, 4],
        [2, 2, 4, 4, 2, 2],
        [3, 4, 3, 4, 3, 4],
    ],
    dtype=float,
)

EVENODD_MATRIX = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [1, 5, 1, 5, 1, 5, 1, 5, 1, 5],
        [1, 1, 5, 5, 1, 1, 5, 5, 1, 1],
    ],
    dtype=float,
)
EXPECTED_EVENODD = np.array([1.0, 0.0, 0.0, 0.0])


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

    def test_longstring_pattern_locked(self) -> None:
        pattern = longstring_pattern(GOLDEN_MATRIX, max_pattern_length=3)
        np.testing.assert_array_equal(pattern, EXPECTED_LONGSTRING_PATTERN)
        self.assertGreater(pattern[4], pattern[0])
        self.assertGreater(pattern[4], pattern[1])


class TestGoldenMahadPsychsynEvenodd(unittest.TestCase):
    """Locked fixtures for mahad / psychsyn / evenodd."""

    def test_mahad_iqr_distances_locked(self) -> None:
        scores = mahad(PARITY_MATRIX, method="iqr")
        expected = np.array(
            [
                2.4748737341529177,
                2.474873734152922,
                0.7245688373094724,
                2.4272755646334576,
                2.427275564633457,
                2.006240264773889,
                2.0062402647738895,
                1.546501427954941,
            ]
        )
        np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-10)
        screen_result = screen(
            PARITY_MATRIX,
            indices=["mahad"],
            options=IndexOptions(),
        )
        np.testing.assert_allclose(screen_result["scores"]["mahad"], scores, rtol=0, atol=1e-12)

    def test_psychsyn_locked(self) -> None:
        scores = psychsyn(PARITY_MATRIX, critval=0.4, resample_na=True)
        expected = np.array(
            [
                0.1893885047696426,
                -0.4166547104932136,
                0.0,
                1.0,
                0.0,
                -0.5,
                -0.5,
                1.0,
            ]
        )
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)
        screen_result = screen(
            PARITY_MATRIX,
            indices=["psychsyn"],
            options=IndexOptions(psychsyn_critval=0.4),
        )
        np.testing.assert_allclose(
            screen_result["scores"]["psychsyn"], expected, rtol=0, atol=1e-12
        )

    def test_evenodd_locked(self) -> None:
        scores = evenodd(EVENODD_MATRIX, factors=[5, 5])
        np.testing.assert_allclose(scores, EXPECTED_EVENODD, rtol=0, atol=1e-12)
        screen_result = screen(
            EVENODD_MATRIX,
            indices=["evenodd"],
            options=IndexOptions(evenodd_factors=[5, 5]),
        )
        np.testing.assert_allclose(
            screen_result["scores"]["evenodd"], EXPECTED_EVENODD, rtol=0, atol=1e-12
        )


class TestIndexOptionsApi(unittest.TestCase):
    """IndexOptions is the sole shared config surface."""

    def test_screen_uses_index_options(self) -> None:
        result = screen(
            GOLDEN_MATRIX,
            indices=["u3_poly", "midpoint"],
            options=IndexOptions(scale_min=1, scale_max=5),
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
