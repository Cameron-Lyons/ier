"""Tests for bounded CLI JSON serialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np

from ier._cli_output import (
    _write_composite_json,
    _write_response_time_json,
    _write_screen_json,
)
from ier.cli import main

if TYPE_CHECKING:
    from ier.types import ScreenResult


def _screen_result() -> ScreenResult:
    scores = np.array([1.0, np.nan, np.inf, -np.inf, 5.0])
    flags = np.array([False, True, False, True, True])
    return {
        "scores": {"example": scores},
        "flags": {"example": flags},
        "thresholds": {"example": 2.0},
        "flag_counts": flags.astype(int),
        "valid_index_counts": np.array([1, 0, 1, 1, 1]),
        "consensus_eligible": np.array([True, False, True, True, True]),
        "consensus_flags": flags,
        "min_flags": 1,
        "min_valid_indices": None,
        "n_indices": 1,
        "indices_used": ["example"],
        "errors": {},
        "n_respondents": len(scores),
        "summary": {
            "example": {
                "mean": np.nan,
                "std": np.inf,
                "min": -np.inf,
                "max": 5.0,
                "n_flagged": 3,
            }
        },
    }


class TestCliJson(unittest.TestCase):
    def test_writers_preserve_schema_and_strict_values(self) -> None:
        result = _screen_result()
        identifiers = ["case-1", 'case-"2', "café", "case\\4", "case-5"]

        screen_output = StringIO()
        _write_screen_json(screen_output, result, identifiers)
        screen_payload = json.loads(screen_output.getvalue())
        self.assertEqual(screen_payload["n_respondents"], 5)
        self.assertEqual(screen_payload["flag_counts"], [0, 1, 0, 1, 1])
        self.assertEqual(screen_payload["valid_index_counts"], [1, 0, 1, 1, 1])
        self.assertEqual(screen_payload["consensus_eligible"], [True, False, True, True, True])
        self.assertIsNone(screen_payload["min_valid_indices"])
        self.assertEqual(screen_payload["consensus_flags"], result["flags"]["example"].tolist())
        self.assertEqual(screen_payload["scores"]["example"], [1.0, None, None, None, 5.0])
        self.assertEqual(screen_payload["flags"]["example"], [False, True, False, True, True])
        self.assertEqual(screen_payload["summary"]["example"]["mean"], None)
        self.assertEqual(screen_payload["respondent_ids"], identifiers)

        composite_output = StringIO()
        _write_composite_json(
            composite_output,
            result["scores"]["example"],
            "mean",
            identifiers,
        )
        composite_payload = json.loads(composite_output.getvalue())
        self.assertIs(composite_payload["standardized"], True)
        self.assertEqual(composite_payload["scores"], [1.0, None, None, None, 5.0])
        self.assertEqual(composite_payload["respondent_ids"], identifiers)
        self.assertEqual(composite_payload["errors"], {})
        self.assertNotIn("flags", composite_payload)
        self.assertNotIn("threshold", composite_payload)
        self.assertNotIn("probabilities", composite_payload)
        self.assertNotIn("probability_scale", composite_payload)
        self.assertNotIn("component_scores", composite_payload)
        self.assertNotIn("valid_index_counts", composite_payload)

        diagnostic_output = StringIO()
        _write_composite_json(
            diagnostic_output,
            result["scores"]["example"],
            "mean",
            errors={"mad": "missing item configuration"},
        )
        self.assertEqual(
            json.loads(diagnostic_output.getvalue())["errors"],
            {"mad": "missing item configuration"},
        )

        timing_output = StringIO()
        _write_response_time_json(
            timing_output,
            result["scores"]["example"],
            result["flags"]["example"],
            "median",
            "low",
            2.0,
            identifiers,
        )
        timing_payload = json.loads(timing_output.getvalue())
        self.assertEqual(timing_payload["scores"], [1.0, None, None, None, 5.0])
        self.assertEqual(timing_payload["flags"], [False, True, False, True, True])
        self.assertEqual(timing_payload["respondent_ids"], identifiers)

    def test_respondent_arrays_are_materialized_only_in_bounded_chunks(self) -> None:
        original_dumps = json.dumps
        chunk_lengths: list[int] = []

        def recording_dumps(
            value: object,
            *,
            allow_nan: bool = True,
            separators: tuple[str, str] | None = None,
        ) -> str:
            if isinstance(value, list):
                chunk_lengths.append(len(value))
            return original_dumps(
                value,
                allow_nan=allow_nan,
                separators=separators,
            )

        output = StringIO()
        component_output = StringIO()
        with (
            patch("ier._cli_output._JSON_ARRAY_CHUNK_SIZE", 2),
            patch("ier._cli_output.json.dumps", side_effect=recording_dumps),
        ):
            _write_screen_json(
                output,
                _screen_result(),
                ["one", "two", "three", "four", "five"],
            )
            _write_composite_json(
                component_output,
                np.arange(5, dtype=float),
                "mean",
                component_scores={
                    "first": np.arange(5, dtype=float),
                    "second": np.arange(5, dtype=float) * 2.0,
                },
                valid_index_counts=np.full(5, 2, dtype=np.int_),
                flags=np.array([False, False, False, False, True]),
                flag_threshold=3.0,
                flag_percentile=75.0,
                probabilities=np.linspace(0.1, 0.9, 5),
            )

        self.assertEqual(json.loads(output.getvalue())["n_respondents"], 5)
        component_payload = json.loads(component_output.getvalue())
        self.assertEqual(component_payload["component_scores"]["second"], [0.0, 2.0, 4.0, 6.0, 8.0])
        self.assertEqual(component_payload["valid_index_counts"], [2, 2, 2, 2, 2])
        self.assertEqual(component_payload["flags"], [False, False, False, False, True])
        self.assertEqual(component_payload["probability_scale"], "uncalibrated_logistic")
        np.testing.assert_allclose(component_payload["probabilities"], np.linspace(0.1, 0.9, 5))
        self.assertTrue(chunk_lengths)
        self.assertLessEqual(max(chunk_lengths), 2)
        self.assertIn(1, chunk_lengths)

    def test_command_json_path_does_not_build_a_complete_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "responses.csv"
            source.write_text("1,1,1\n1,2,3\n3,3,3\n", encoding="utf-8")
            destination = root / "screen.json"

            with patch(
                "ier._cli_output.StringIO",
                side_effect=AssertionError("buffered JSON path used"),
            ):
                code = main(
                    [
                        "screen",
                        str(source),
                        "--indices",
                        "irv",
                        "--format",
                        "json",
                        "--output",
                        str(destination),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(destination.read_text())["n_respondents"], 3)


if __name__ == "__main__":
    unittest.main()
