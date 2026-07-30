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
- guttman / markov / person_total / midpoint / lz / onset: locked regression
  values (see also ``tests/fixtures/parity/*.json``).

They are regression fixtures for scientific credibility, not a claim of
bit-identical output against every CRAN release.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from ier import (
    IndexOptions,
    composite,
    evenodd,
    guttman,
    irv,
    longstring_pattern,
    longstring_scores,
    lz,
    mahad,
    markov,
    onset,
    person_total,
    screen,
)
from ier.psychsyn import psychsyn
from ier.u3_poly import midpoint_responding

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "parity"

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

ONSET_MATRIX = np.array(
    [
        [
            5.0,
            4.0,
            4.0,
            2.0,
            5.0,
            1.0,
            3.0,
            4.0,
            5.0,
            3.0,
            2.0,
            2.0,
            3.0,
            3.0,
            4.0,
            5.0,
            1.0,
            5.0,
            3.0,
            2.0,
            4.0,
            3.0,
            2.0,
            2.0,
            4.0,
            3.0,
            3.0,
            2.0,
            4.0,
            2.0,
            2.0,
            5.0,
            2.0,
            2.0,
            4.0,
            4.0,
            1.0,
            1.0,
            2.0,
            5.0,
        ],
        [
            3.0,
            4.0,
            2.0,
            2.0,
            4.0,
            5.0,
            1.0,
            1.0,
            4.0,
            2.0,
            3.0,
            1.0,
            5.0,
            3.0,
            5.0,
            4.0,
            4.0,
            2.0,
            4.0,
            1.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ],
        [3.0] * 40,
    ],
    dtype=float,
)
EXPECTED_ONSET = np.array([9.0, 15.0, np.nan])


def _expected_array(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


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


class TestGoldenExtendedIndices(unittest.TestCase):
    """Locked fixtures for guttman / markov / person_total / midpoint / lz / onset."""

    def test_guttman_locked(self) -> None:
        scores = guttman(PARITY_MATRIX)
        expected = np.array(
            [
                0.7333333333333333,
                0.6666666666666666,
                0.0,
                0.4666666666666667,
                0.0,
                0.13333333333333333,
                0.4,
                0.13333333333333333,
            ]
        )
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)

    def test_markov_locked(self) -> None:
        scores = markov(PARITY_MATRIX)
        expected = np.array(
            [
                -0.0,
                0.4,
                -0.0,
                -0.0,
                -0.0,
                0.9509775004326937,
                0.9509775004326937,
                -0.0,
            ]
        )
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)

    def test_person_total_locked(self) -> None:
        scores = person_total(PARITY_MATRIX)
        expected = np.array(
            [
                0.7635899174405921,
                0.6863485850246135,
                np.nan,
                0.4842001247062523,
                np.nan,
                -0.39129279043561477,
                0.39129279043561477,
                -0.4842001247062523,
            ]
        )
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12, equal_nan=True)

    def test_midpoint_locked(self) -> None:
        scores = midpoint_responding(PARITY_MATRIX, scale_min=1, scale_max=5)
        expected = np.array([1 / 6, 1 / 3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)

    def test_lz_locked(self) -> None:
        scores = lz(PARITY_MATRIX)
        expected = np.array(
            [
                1.1901012063680725,
                1.376379500972372,
                0.5659327404151528,
                -0.18770575500943293,
                0.5659327404151528,
                -0.46960760961907766,
                0.15231814658842585,
                -1.0950615670859407,
            ]
        )
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)

    def test_onset_locked(self) -> None:
        scores = onset(ONSET_MATRIX, window_size=5, min_items=20)
        np.testing.assert_allclose(scores, EXPECTED_ONSET, rtol=0, atol=1e-12, equal_nan=True)
        screen_result = screen(
            ONSET_MATRIX,
            indices=["onset"],
            options=IndexOptions(onset_window_size=5, onset_min_items=20),
        )
        np.testing.assert_allclose(
            screen_result["scores"]["onset"], EXPECTED_ONSET, rtol=0, atol=1e-12, equal_nan=True
        )


class TestParityJsonHarness(unittest.TestCase):
    """Load JSON fixtures so R-side regenerations can drop in replacement files."""

    def test_all_parity_fixtures(self) -> None:
        paths = sorted(FIXTURES.glob("*.json"))
        self.assertGreaterEqual(len(paths), 4)
        scorers = {
            "irv": lambda x, _opts: irv(x),
            "longstring": lambda x, _opts: longstring_scores(x),
            "mahad_iqr": lambda x, _opts: mahad(x, method="iqr"),
            "psychsyn": lambda x, opts: psychsyn(
                x, critval=float(opts.get("psychsyn_critval", 0.6)), resample_na=True
            ),
            "guttman": lambda x, _opts: guttman(x),
            "markov": lambda x, _opts: markov(x),
            "person_total": lambda x, _opts: person_total(x),
            "midpoint": lambda x, opts: midpoint_responding(
                x,
                scale_min=float(opts.get("scale_min", 1)),
                scale_max=float(opts.get("scale_max", 5)),
            ),
            "lz": lambda x, _opts: lz(x),
            "evenodd": lambda x, opts: evenodd(x, factors=list(opts["evenodd_factors"])),
            "onset": lambda x, opts: onset(
                x,
                window_size=int(opts.get("onset_window_size", 10)),
                min_items=int(opts.get("onset_min_items", 20)),
            ),
        }
        for path in paths:
            payload = json.loads(path.read_text())
            matrix = np.asarray(payload["matrix"], dtype=float)
            options = payload.get("options", {})
            for name, expected_list in payload["expected"].items():
                with self.subTest(fixture=path.name, index=name):
                    scores = scorers[name](matrix, options)
                    expected = _expected_array(expected_list)
                    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-10, equal_nan=True)


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
