"""Tests for the ier CLI."""

from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ier.cli import (
    _emit_composite_csv,
    _emit_composite_json,
    _load_input,
    _load_matrix,
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    _parse_pair_list,
    _parse_thresholds,
    main,
)


class TestCli(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)
        self.csv_path = self.root / "data.csv"
        rows = [
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5],
            [5, 5, 5, 1, 2],
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["i1", "i2", "i3", "i4", "i5"])
            writer.writerows(rows)

    def test_screen_command(self) -> None:
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--scale-min",
                "1",
                "--scale-max",
                "5",
                "--top",
                "2",
            ]
        )
        self.assertEqual(code, 0)

    def test_screen_reports_soft_errors(self) -> None:
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "mad",
                "evenodd",
                "--top",
                "1",
            ]
        )
        self.assertEqual(code, 0)

    def test_strict_screen_returns_structured_index_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(
                [
                    "screen",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "mad",
                    "--strict",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("error: index 'mad' failed: mad_positive_items", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_composite_command(self) -> None:
        code = main(
            [
                "composite",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--method",
                "mean",
                "--top",
                "2",
            ]
        )
        self.assertEqual(code, 0)

    def test_response_time_json_preserves_ids_and_fixed_cutoff(self) -> None:
        timings = self.root / "timings.csv"
        timings.write_text(
            "participant,t1,t2,t3\nfast,0.4,0.5,0.6\nsteady,1,1,1\ntypical,2,3,4\n",
            encoding="utf-8",
        )
        out = self.root / "timings.json"

        code = main(
            [
                "response-time",
                str(timings),
                "--id-column",
                "participant",
                "--metric",
                "median",
                "--threshold",
                "1",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["metric"], "median")
        self.assertEqual(payload["flag_direction"], "low")
        self.assertEqual(payload["threshold"], 1.0)
        self.assertEqual(payload["respondent_ids"], ["fast", "steady", "typical"])
        self.assertEqual(payload["scores"], [0.5, 1.0, 3.0])
        self.assertEqual(payload["flags"], [True, True, False])

    def test_response_time_consistency_csv_selects_timing_columns(self) -> None:
        timings = self.root / "mixed-timings.csv"
        timings.write_text(
            "participant,cohort,t1,t2,t3\nfast,A,0.4,0.5,0.6\nsteady,B,1,1,1\ntypical,A,2,3,4\n",
            encoding="utf-8",
        )
        out = self.root / "timings.csv.out"

        code = main(
            [
                "response-time",
                str(timings),
                "--id-column",
                "participant",
                "--item-columns",
                "t1,t2,t3",
                "--metric",
                "consistency",
                "--threshold",
                "0",
                "--format",
                "csv",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        rows = list(csv.DictReader(StringIO(out.read_text(encoding="utf-8"))))
        self.assertEqual([row["respondent"] for row in rows], ["fast", "steady", "typical"])
        self.assertEqual([row["response_time_flag"] for row in rows], ["0", "1", "0"])

    def test_response_time_text_uses_default_low_tail(self) -> None:
        timings = self.root / "text-timings.csv"
        timings.write_text(
            "participant,t1,t2,t3\nfast,0.4,0.5,0.6\nsteady,1,1,1\ntypical,2,3,4\n",
            encoding="utf-8",
        )
        stdout = StringIO()

        with patch("sys.stdout", stdout):
            code = main(
                [
                    "response-time",
                    str(timings),
                    "--id-column",
                    "participant",
                    "--metric",
                    "mean",
                    "--top",
                    "2",
                ]
            )

        self.assertEqual(code, 0)
        text = stdout.getvalue()
        self.assertIn("flag direction: low", text)
        self.assertIn("threshold: 0.55", text)
        self.assertIn("flagged: 1", text)
        self.assertLess(text.index("fast\t"), text.index("steady\t"))

    def test_response_time_mixture_flags_fast_component(self) -> None:
        timings = self.root / "mixture-timings.csv"
        timings.write_text(
            "t1,t2,t3\n"
            "0.4,0.5,0.6\n0.5,0.6,0.7\n0.3,0.4,0.5\n0.6,0.7,0.8\n"
            "4,5,6\n5,6,7\n6,7,8\n4.5,5,5.5\n",
            encoding="utf-8",
        )
        out = self.root / "mixture.json"

        code = main(
            [
                "response-time",
                str(timings),
                "--metric",
                "mixture",
                "--threshold",
                "0.5",
                "--random-seed",
                "42",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["flag_direction"], "high")
        self.assertEqual(payload["flags"], [True, True, True, True, False, False, False, False])
        self.assertTrue(all(0.0 <= score <= 1.0 for score in payload["scores"]))

    def test_response_time_invalid_components_returns_structured_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(
                [
                    "response-time",
                    str(self.csv_path),
                    "--metric",
                    "mixture",
                    "--components",
                    "1",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("error: n_components must be at least 2", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_standard_stream_pipeline_is_forward_only(self) -> None:
        class ForwardOnlyInput(StringIO):
            def seek(self, *args: object, **kwargs: object) -> int:
                raise AssertionError("standard input must not be rewound")

        stdin = ForwardOnlyInput("participant,i1,i2,i3\nfast,1,1,1\ntypical,1,2,3\n")
        stdout = StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            code = main(
                [
                    "screen",
                    "-",
                    "--id-column",
                    "participant",
                    "--indices",
                    "irv",
                    "--format",
                    "json",
                    "--output",
                    "-",
                ]
            )

        self.assertEqual(code, 0)
        self.assertFalse(stdin.closed)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["respondent_ids"], ["fast", "typical"])
        self.assertEqual(payload["scores"]["irv"], [0.0, np.std([1.0, 2.0, 3.0])])

    def test_gzip_input_and_output(self) -> None:
        timings = self.root / "timings.csv.gz"
        with gzip.open(timings, mode="wt", newline="", encoding="utf-8") as handle:
            handle.write("participant,t1,t2,t3\nfast,0.4,0.5,0.6\nsteady,1,1,1\ntypical,2,3,4\n")
        out = self.root / "timing-scores.json.gz"

        code = main(
            [
                "response-time",
                str(timings),
                "--id-column",
                "participant",
                "--threshold",
                "1",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        with gzip.open(out, mode="rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["respondent_ids"], ["fast", "steady", "typical"])
        self.assertEqual(payload["scores"], [0.5, 1.0, 3.0])
        self.assertEqual(payload["flags"], [True, True, False])

    def test_indices_command_text_output(self) -> None:
        stdout = StringIO()
        with patch("sys.stdout", stdout):
            code = main(["indices"])

        self.assertEqual(code, 0)
        self.assertIn("index\tdirection\tflag_mode", stdout.getvalue())
        self.assertIn("irv\tlow\tpercentile", stdout.getvalue())

    def test_indices_command_json_output(self) -> None:
        out = self.root / "indices.json"
        code = main(["indices", "--format", "json", "--output", str(out)])

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["n_indices"], 21)
        self.assertEqual(payload["indices"]["onset"]["flag_mode"], "present")
        self.assertEqual(payload["indices"]["missing_rate"]["flag_direction"], "high")
        self.assertEqual(payload["indices"]["evenodd"]["required_options"], ["evenodd_factors"])

    def test_indices_command_csv_output(self) -> None:
        out = self.root / "indices.csv"
        code = main(["indices", "--format", "csv", "--output", str(out)])

        self.assertEqual(code, 0)
        rows = list(csv.DictReader(StringIO(out.read_text(encoding="utf-8"))))
        self.assertEqual(len(rows), 21)
        onset = next(row for row in rows if row["index"] == "onset")
        self.assertEqual(onset["flag_mode"], "present")

    def test_screen_missing_rate_index(self) -> None:
        missing = self.root / "missing-rate.csv"
        missing.write_text("i1,i2,i3\n1,2,3\n1,,3\n,,3\n", encoding="utf-8")
        out = self.root / "missing-rate.json"

        code = main(
            [
                "screen",
                str(missing),
                "--indices",
                "missing_rate",
                "--threshold",
                "missing_rate=0.5",
                "--min-flags",
                "1",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        np.testing.assert_allclose(payload["scores"]["missing_rate"], [0.0, 1 / 3, 2 / 3])
        self.assertEqual(payload["consensus_flags"], [False, False, True])

    def test_screen_json_output(self) -> None:
        out = self.root / "screen.json"
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(out.exists())
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("flag_counts", payload)
        self.assertIn("consensus_flags", payload)
        self.assertEqual(payload["min_flags"], 2)
        self.assertEqual(set(payload["thresholds"]), {"irv", "longstring"})

    def test_named_respondent_ids_are_preserved_across_outputs(self) -> None:
        identified = self.root / "identified.csv"
        identified.write_text(
            "participant,i1,i2,i3,i4,i5\ncase-01,1,1,1,1,1\ncase-02,1,2,3,4,5\ncase-03,5,5,5,1,2\n",
            encoding="utf-8",
        )
        matrix, identifiers = _load_input(identified, None, "participant")
        self.assertEqual(identifiers, ["case-01", "case-02", "case-03"])
        self.assertEqual(matrix.shape, (3, 5))

        screen_out = self.root / "identified-screen.csv"
        self.assertEqual(
            main(
                [
                    "screen",
                    str(identified),
                    "--id-column",
                    "participant",
                    "--indices",
                    "irv",
                    "longstring",
                    "--format",
                    "csv",
                    "--output",
                    str(screen_out),
                ]
            ),
            0,
        )
        rows = list(csv.DictReader(StringIO(screen_out.read_text(encoding="utf-8"))))
        self.assertEqual([row["respondent"] for row in rows], identifiers)

        composite_out = self.root / "identified-composite.json"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(identified),
                    "--id-column",
                    "participant",
                    "--indices",
                    "irv",
                    "longstring",
                    "--format",
                    "json",
                    "--output",
                    str(composite_out),
                ]
            ),
            0,
        )
        payload = json.loads(composite_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["respondent_ids"], identifiers)

    def test_named_item_columns_exclude_metadata_and_preserve_order(self) -> None:
        mixed = self.root / "mixed-columns.csv"
        mixed.write_text(
            "participant,cohort,i1,age,i2,i3,i4\n"
            "case-01,A,1,34,2,3,4\n"
            "case-02,B,4,29,3,2,1\n"
            "case-03,A,2,41,2,4,4\n",
            encoding="utf-8",
        )

        matrix, identifiers = _load_input(
            mixed,
            None,
            "participant",
            ["i3", "i1", "i4", "i2"],
        )

        self.assertEqual(identifiers, ["case-01", "case-02", "case-03"])
        np.testing.assert_array_equal(
            matrix,
            [[3.0, 1.0, 4.0, 2.0], [2.0, 4.0, 1.0, 3.0], [4.0, 2.0, 4.0, 2.0]],
        )

    def test_numeric_looking_item_headers_can_be_selected(self) -> None:
        numeric_headers = self.root / "numeric-headers.csv"
        numeric_headers.write_text(
            "1,2,metadata\n3,4,A\n5,6,B\n",
            encoding="utf-8",
        )

        matrix, identifiers = _load_input(numeric_headers, None, item_columns=["2", "1"])

        self.assertIsNone(identifiers)
        np.testing.assert_array_equal(matrix, [[4.0, 3.0], [6.0, 5.0]])

    def test_item_column_selection_works_across_commands_and_outputs(self) -> None:
        mixed = self.root / "mixed-command-columns.csv"
        mixed.write_text(
            "participant,group,i1,i2,i3,i4\n"
            "case-01,A,1,1,1,1\n"
            "case-02,B,1,2,3,4\n"
            "case-03,A,4,4,2,1\n",
            encoding="utf-8",
        )
        screen_out = self.root / "selected-screen.json"
        composite_out = self.root / "selected-composite.csv"

        self.assertEqual(
            main(
                [
                    "screen",
                    str(mixed),
                    "--id-column",
                    "participant",
                    "--item-columns",
                    "i1,i2",
                    "--item-columns",
                    "i3,i4",
                    "--indices",
                    "irv",
                    "longstring",
                    "--format",
                    "json",
                    "--output",
                    str(screen_out),
                ]
            ),
            0,
        )
        screen_payload = json.loads(screen_out.read_text(encoding="utf-8"))
        self.assertEqual(screen_payload["respondent_ids"], ["case-01", "case-02", "case-03"])

        self.assertEqual(
            main(
                [
                    "composite",
                    str(mixed),
                    "--id-column",
                    "participant",
                    "--item-columns",
                    "i1,i2,i3,i4",
                    "--indices",
                    "irv",
                    "longstring",
                    "--format",
                    "csv",
                    "--output",
                    str(composite_out),
                ]
            ),
            0,
        )
        composite_rows = list(csv.DictReader(StringIO(composite_out.read_text(encoding="utf-8"))))
        self.assertEqual(
            [row["respondent"] for row in composite_rows],
            ["case-01", "case-02", "case-03"],
        )

    def test_invalid_item_column_selections_return_structured_errors(self) -> None:
        cases = {
            "missing": (
                "participant,i1,i2\na,1,2\n",
                ["--item-columns", "unknown"],
                "item column 'unknown' was not found",
            ),
            "duplicate-request": (
                "participant,i1,i2\na,1,2\n",
                ["--item-columns", "i1,i1"],
                "item columns cannot contain duplicate names",
            ),
            "duplicate-header": (
                "participant,i1,i1\na,1,2\n",
                ["--item-columns", "i1"],
                "item column 'i1' appears more than once",
            ),
            "id-overlap": (
                "participant,i1,i2\na,1,2\n",
                ["--id-column", "participant", "--item-columns", "participant,i1"],
                "cannot also be selected as an item column",
            ),
            "headerless": (
                "1,2,3\n4,5,6\n",
                ["--item-columns", "i1,i2"],
                "item column 'i1' was not found in the header",
            ),
        }
        for name, (contents, args, message) in cases.items():
            with self.subTest(name=name):
                path = self.root / f"{name}-items.csv"
                path.write_text(contents, encoding="utf-8")
                stderr = StringIO()
                with patch("sys.stderr", stderr):
                    code = main(["screen", str(path), *args, "--indices", "irv"])
                self.assertEqual(code, 1)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_invalid_respondent_id_columns_return_structured_errors(self) -> None:
        cases = {
            "missing": "participant,i1,i2\na,1,2\n",
            "duplicate": "participant,i1,i2\na,1,2\na,2,3\n",
            "blank": "participant,i1,i2\n,1,2\nb,2,3\n",
        }
        expected = {
            "missing": "ID column 'unknown' was not found",
            "duplicate": "contains duplicate values",
            "blank": "contains blank values",
        }
        for name, contents in cases.items():
            with self.subTest(name=name):
                path = self.root / f"{name}-ids.csv"
                path.write_text(contents, encoding="utf-8")
                stderr = StringIO()
                column = "unknown" if name == "missing" else "participant"
                with patch("sys.stderr", stderr):
                    code = main(["screen", str(path), "--id-column", column])
                self.assertEqual(code, 1)
                self.assertIn(expected[name], stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_screen_fixed_threshold(self) -> None:
        out = self.root / "screen-threshold.json"
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "longstring",
                "--threshold",
                "longstring=1",
                "--min-flags",
                "1",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["thresholds"], {"longstring": 1.0})
        self.assertEqual(payload["flags"]["longstring"], [True, True, True])
        self.assertEqual(payload["consensus_flags"], [True, True, True])

    def test_screen_custom_consensus_threshold(self) -> None:
        out = self.root / "screen-consensus.json"
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--min-flags",
                "1",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["min_flags"], 1)
        self.assertEqual(
            payload["consensus_flags"],
            [count >= 1 for count in payload["flag_counts"]],
        )

    def test_screen_invalid_consensus_threshold_returns_structured_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(["screen", str(self.csv_path), "--min-flags", "0"])

        self.assertEqual(code, 1)
        self.assertIn("error: min_flags must be a positive integer", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_screen_json_uses_null_for_non_finite_scores(self) -> None:
        constant = self.root / "constant.csv"
        constant.write_text("1,1,1,1\n1,1,1,1\n1,1,1,1\n", encoding="utf-8")
        out = self.root / "screen-strict.json"

        code = main(
            [
                "screen",
                str(constant),
                "--indices",
                "psychsyn",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["scores"]["psychsyn"], [None, None, None])
        self.assertIsNone(payload["summary"]["psychsyn"]["mean"])

    def test_composite_json_uses_null_for_all_non_finite_values(self) -> None:
        text = _emit_composite_json(
            np.array([1.0, np.nan, np.inf, -np.inf], dtype=float),
            "mean",
        )

        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text)["scores"], [1.0, None, None, None])

    def test_composite_csv_uses_empty_cells_for_all_non_finite_values(self) -> None:
        text = _emit_composite_csv(np.array([1.0, np.nan, np.inf, -np.inf], dtype=float))

        self.assertEqual(
            list(csv.reader(StringIO(text))),
            [
                ["respondent", "composite_score"],
                ["0", "1.0"],
                ["1", ""],
                ["2", ""],
                ["3", ""],
            ],
        )

    def test_screen_csv_uses_empty_cells_for_non_finite_scores(self) -> None:
        constant = self.root / "constant.csv"
        constant.write_text("1,1,1,1\n1,1,1,1\n1,1,1,1\n", encoding="utf-8")
        out = self.root / "screen.csv"

        code = main(
            [
                "screen",
                str(constant),
                "--indices",
                "psychsyn",
                "--format",
                "csv",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        rows = list(csv.DictReader(StringIO(out.read_text(encoding="utf-8"))))
        self.assertEqual([row["psychsyn_score"] for row in rows], ["", "", ""])
        self.assertTrue(all(row["consensus_flag"] in {"0", "1"} for row in rows))

    def test_composite_csv_output(self) -> None:
        out = self.root / "scores.csv"
        code = main(
            [
                "composite",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--format",
                "csv",
                "--output",
                str(out),
            ]
        )
        self.assertEqual(code, 0)
        text = out.read_text(encoding="utf-8")
        self.assertIn("composite_score", text)

    def test_jagged_csv_errors(self) -> None:
        jagged = self.root / "jagged.csv"
        jagged.write_text("1,2,3\n4,5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "jagged"):
            _load_matrix(jagged, ",")

    def test_blank_csv_cells_load_as_nan(self) -> None:
        missing = self.root / "missing-values.csv"
        missing.write_text("i1,i2,i3\n1,,3\n,5,6\n", encoding="utf-8")

        matrix = _load_matrix(missing, None)

        np.testing.assert_equal(
            matrix,
            np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]]),
        )

    def test_blank_first_cell_does_not_discard_first_data_row(self) -> None:
        missing = self.root / "missing-first.csv"
        missing.write_text(",2,3\n4,5,6\n", encoding="utf-8")

        matrix = _load_matrix(missing, ",")

        np.testing.assert_equal(
            matrix,
            np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )

    def test_blank_first_header_cell_still_drops_header_row(self) -> None:
        missing = self.root / "blank-header.csv"
        missing.write_text(",item2,item3\n1,2,3\n4,5,6\n", encoding="utf-8")

        matrix = _load_matrix(missing, ",")

        np.testing.assert_array_equal(
            matrix,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )

    def test_screen_csv_and_composite_json(self) -> None:
        screen_out = self.root / "screen.csv"
        composite_out = self.root / "composite.json"
        self.assertEqual(
            main(
                [
                    "screen",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--format",
                    "csv",
                    "--output",
                    str(screen_out),
                ]
            ),
            0,
        )
        self.assertIn("irv_score", screen_out.read_text(encoding="utf-8"))
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--format",
                    "json",
                    "--output",
                    str(composite_out),
                ]
            ),
            0,
        )
        self.assertIn("scores", composite_out.read_text(encoding="utf-8"))

    def test_index_options_cli_flags(self) -> None:
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "evenodd",
                "semantic_syn",
                "--evenodd-factors",
                "2,3",
                "--semantic-item-pairs",
                "0,1;2,3",
                "--top",
                "1",
            ]
        )
        self.assertEqual(code, 0)

    def test_parse_helpers(self) -> None:
        self.assertIsNone(_parse_int_list(None))
        self.assertIsNone(_parse_int_list(""))
        self.assertEqual(_parse_int_list("1, 2"), [1, 2])
        self.assertIsNone(_parse_float_list(None))
        self.assertIsNone(_parse_float_list(" , "))
        self.assertEqual(_parse_float_list("1.5,2"), [1.5, 2.0])
        self.assertIsNone(_parse_name_list(None))
        self.assertEqual(_parse_name_list(["i1, i2", "i3"]), ["i1", "i2", "i3"])
        self.assertIsNone(_parse_pair_list(None))
        self.assertEqual(_parse_pair_list("0,1;2,3"), [(0, 1), (2, 3)])
        self.assertIsNone(_parse_pair_list(";;"))
        self.assertIsNone(_parse_thresholds(None))
        self.assertEqual(
            _parse_thresholds(["irv=0.5", "longstring = 4"]), {"irv": 0.5, "longstring": 4.0}
        )
        with self.assertRaises(ValueError):
            _parse_pair_list("0-1")
        with self.assertRaises(ValueError):
            _parse_thresholds(["irv"])
        with self.assertRaises(ValueError):
            _parse_thresholds(["irv=1", "irv=2"])
        with self.assertRaises(ValueError):
            _parse_name_list([" , "])

    def test_invalid_fixed_threshold_returns_structured_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(["screen", str(self.csv_path), "--threshold", "irv=nan"])

        self.assertEqual(code, 1)
        self.assertIn("error: threshold for irv must be a finite number", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_invalid_semantic_pairs_cli(self) -> None:
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "irv",
                "--semantic-item-pairs",
                "bad",
            ]
        )
        self.assertEqual(code, 1)

    def test_invalid_index_returns_structured_error(self) -> None:
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            code = main(["screen", str(self.csv_path), "--indices", "nonexistent"])

        self.assertEqual(code, 1)
        self.assertIn("error: invalid index 'nonexistent'", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_invalid_percentile_returns_structured_error(self) -> None:
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            code = main(["screen", str(self.csv_path), "--percentile", "101"])

        self.assertEqual(code, 1)
        self.assertIn("error:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_explicit_delimiter(self) -> None:
        tsv = self.root / "data.tsv"
        tsv.write_text("1\t2\t3\n4\t5\t6\n", encoding="utf-8")
        matrix = _load_matrix(tsv, "\t")
        self.assertEqual(matrix.shape, (2, 3))
        code = main(["screen", str(tsv), "--delimiter", "\t", "--indices", "irv", "--top", "1"])
        self.assertEqual(code, 0)

    def test_invalid_delimiter_returns_structured_error(self) -> None:
        for delimiter in ["", "||", "\n"]:
            with self.subTest(delimiter=delimiter):
                stderr = StringIO()

                with patch("sys.stderr", stderr):
                    code = main(["screen", str(self.csv_path), "--delimiter", delimiter])

                self.assertEqual(code, 1)
                self.assertIn(
                    "error: delimiter must be exactly one non-newline character",
                    stderr.getvalue(),
                )
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_whitespace_delimited_input(self) -> None:
        whitespace = self.root / "data.txt"
        whitespace.write_text("i1 i2 i3\n1  2 3\n4 5   6\n", encoding="utf-8")

        matrix = _load_matrix(whitespace, None)

        np.testing.assert_array_equal(matrix, np.array([[1, 2, 3], [4, 5, 6]], dtype=float))
        code = main(["screen", str(whitespace), "--indices", "irv", "--top", "1"])
        self.assertEqual(code, 0)

    def test_sniffer_fallback(self) -> None:
        weird = self.root / "weird.csv"
        weird.write_text("1  2\t3\n4\t5  6\n", encoding="utf-8")

        with patch("ier.cli.csv.Sniffer.sniff", side_effect=csv.Error("nope")):
            matrix = _load_matrix(weird, None)

        np.testing.assert_array_equal(matrix, np.array([[1, 2, 3], [4, 5, 6]], dtype=float))

    def test_empty_file(self) -> None:
        empty = self.root / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_matrix(empty, ",")

    def test_header_only(self) -> None:
        header_only = self.root / "header.csv"
        header_only.write_text("a,b,c\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_matrix(header_only, ",")

    def test_non_numeric(self) -> None:
        bad = self.root / "bad.csv"
        bad.write_text("1,2,x\n4,5,6\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_matrix(bad, ",")

    def test_missing_file(self) -> None:
        code = main(["screen", str(self.root / "missing.csv")])
        self.assertEqual(code, 1)
