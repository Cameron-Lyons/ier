"""Tests for the ier CLI."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ier.cli import _load_matrix, main


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
