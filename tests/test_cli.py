"""Tests for the ier CLI."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ier.cli import (
    _load_matrix,
    _parse_float_list,
    _parse_int_list,
    _parse_pair_list,
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
        self.assertIn("flag_counts", out.read_text(encoding="utf-8"))

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
        self.assertIsNone(_parse_pair_list(None))
        self.assertEqual(_parse_pair_list("0,1;2,3"), [(0, 1), (2, 3)])
        self.assertIsNone(_parse_pair_list(";;"))
        with self.assertRaises(ValueError):
            _parse_pair_list("0-1")

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

    def test_explicit_delimiter(self) -> None:
        tsv = self.root / "data.tsv"
        tsv.write_text("1\t2\t3\n4\t5\t6\n", encoding="utf-8")
        matrix = _load_matrix(tsv, "\t")
        self.assertEqual(matrix.shape, (2, 3))
        code = main(["screen", str(tsv), "--delimiter", "\t", "--indices", "irv", "--top", "1"])
        self.assertEqual(code, 0)

    def test_sniffer_fallback(self) -> None:
        weird = self.root / "weird.csv"
        weird.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
        with (
            patch("ier.cli.csv.Sniffer.sniff", side_effect=csv.Error("nope")),
            self.assertRaises(ValueError),
        ):
            # Space-separated with comma fallback fails float parsing.
            _load_matrix(weird, None)

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
