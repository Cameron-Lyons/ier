"""Tests for versioned NumPy CLI result archives."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ier._cli_npz import _write_npz_archive
from ier.cli import main


class TestCliNpz(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(self._td.name)

    def test_outputs_are_typed_pickle_free_and_preserve_ids(self) -> None:
        identified = self.root / "identified.csv"
        identified.write_text(
            "participant,i1,i2,i3,i4,i5\ncase-01,1,1,1,1,1\ncase-02,1,2,3,4,5\ncase-03,5,5,5,1,2\n",
            encoding="utf-8",
        )
        screen_out = self.root / "screen.npz"
        composite_out = self.root / "composite.npz"
        timing_out = self.root / "timing.npz"

        common = [
            str(identified),
            "--id-column",
            "participant",
            "--format",
            "npz",
        ]
        self.assertEqual(
            main(
                [
                    "screen",
                    *common,
                    "--indices",
                    "irv",
                    "longstring",
                    "--output",
                    str(screen_out),
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "composite",
                    *common,
                    "--indices",
                    "irv",
                    "longstring",
                    "--weight",
                    "irv=2",
                    "--weight",
                    "longstring=0.5",
                    "--min-valid-indices",
                    "2",
                    "--include-components",
                    "--output",
                    str(composite_out),
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "response-time",
                    *common,
                    "--threshold",
                    "2",
                    "--output",
                    str(timing_out),
                ]
            ),
            0,
        )

        def read_all(path: Path) -> dict[str, np.ndarray]:
            with np.load(path, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}

        screen_result = read_all(screen_out)
        self.assertEqual(screen_result["schema_version"].item(), 1)
        self.assertEqual(screen_result["result_type"].item(), "screen")
        self.assertEqual(screen_result["n_respondents"].item(), 3)
        self.assertEqual(screen_result["index_names"].tolist(), ["irv", "longstring"])
        self.assertEqual(
            screen_result["respondent_ids"].tolist(), ["case-01", "case-02", "case-03"]
        )
        self.assertEqual(screen_result["summary_columns"].tolist(), ["mean", "std", "min", "max"])
        self.assertEqual(screen_result["summary_statistics"].shape, (2, 4))
        self.assertEqual(screen_result["score__irv"].shape, (3,))
        self.assertEqual(screen_result["flag__irv"].dtype, np.bool_)
        self.assertEqual(screen_result["consensus_flags"].dtype, np.bool_)

        composite_result = read_all(composite_out)
        self.assertEqual(composite_result["result_type"].item(), "composite")
        self.assertEqual(composite_result["method"].item(), "mean")
        self.assertTrue(composite_result["standardized"].item())
        self.assertEqual(composite_result["weight_names"].tolist(), ["irv", "longstring"])
        self.assertEqual(composite_result["weights"].tolist(), [2.0, 0.5])
        self.assertEqual(composite_result["min_valid_indices"].item(), 2)
        self.assertEqual(composite_result["error_names"].tolist(), [])
        self.assertEqual(composite_result["error_messages"].tolist(), [])
        self.assertEqual(composite_result["index_names"].tolist(), ["irv", "longstring"])
        self.assertEqual(composite_result["valid_index_counts"].tolist(), [2, 2, 2])
        self.assertEqual(composite_result["score__irv"].shape, (3,))
        self.assertEqual(composite_result["score__longstring"].shape, (3,))
        self.assertEqual(composite_result["scores"].shape, (3,))
        self.assertEqual(
            composite_result["respondent_ids"].tolist(), ["case-01", "case-02", "case-03"]
        )

        timing_result = read_all(timing_out)
        self.assertEqual(timing_result["result_type"].item(), "response_time")
        self.assertEqual(timing_result["metric"].item(), "median")
        self.assertEqual(timing_result["flag_direction"].item(), "low")
        self.assertEqual(timing_result["threshold"].item(), 2.0)
        self.assertEqual(timing_result["flags"].tolist(), [True, False, False])
        self.assertEqual(timing_result["flags"].dtype, np.bool_)

    def test_composite_archive_records_disabled_standardization(self) -> None:
        source = self.root / "responses.csv"
        source.write_text("1,1,1,1\n1,2,3,4\n4,4,1,2\n", encoding="utf-8")
        out = self.root / "raw-composite.npz"

        self.assertEqual(
            main(
                [
                    "composite",
                    str(source),
                    "--indices",
                    "irv",
                    "longstring",
                    "--no-standardize",
                    "--format",
                    "npz",
                    "--output",
                    str(out),
                ]
            ),
            0,
        )

        with np.load(out, allow_pickle=False) as result:
            self.assertFalse(result["standardized"].item())
            self.assertEqual(result["standardized"].dtype, np.bool_)

    def test_output_preserves_non_finite_values(self) -> None:
        constant = self.root / "constant.csv"
        constant.write_text("1,1,1,1\n1,1,1,1\n1,1,1,1\n", encoding="utf-8")
        out = self.root / "screen.npz"

        code = main(
            [
                "screen",
                str(constant),
                "--indices",
                "psychsyn",
                "--format",
                "npz",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        with np.load(out, allow_pickle=False) as archive:
            self.assertTrue(np.isnan(archive["score__psychsyn"]).all())
            self.assertTrue(np.isnan(archive["summary_statistics"]).all())

    def test_screen_output_preserves_presence_thresholds_and_soft_failures(self) -> None:
        matrix = self.root / "responses.csv"
        rows = [",".join([str(value)] * 20) for value in [1, 2, 3]]
        matrix.write_text("\n".join(rows) + "\n", encoding="utf-8")
        out = self.root / "screen.npz"

        code = main(
            [
                "screen",
                str(matrix),
                "--indices",
                "irv",
                "mad",
                "onset",
                "--format",
                "npz",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        with np.load(out, allow_pickle=False) as archive:
            self.assertEqual(archive["index_names"].tolist(), ["irv", "onset"])
            self.assertTrue(np.isnan(archive["thresholds"][1]))
            self.assertEqual(archive["error_names"].tolist(), ["mad"])
            self.assertIn("mad_positive_items", archive["error_messages"][0])

    def test_composite_output_preserves_soft_failures(self) -> None:
        matrix = self.root / "partial-composite.csv"
        matrix.write_text("1,2,3,4\n4,3,2,1\n", encoding="utf-8")
        out = self.root / "partial-composite.npz"
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            code = main(
                [
                    "composite",
                    str(matrix),
                    "--indices",
                    "irv",
                    "mad",
                    "--format",
                    "npz",
                    "--output",
                    str(out),
                ]
            )

        self.assertEqual(code, 0)
        self.assertIn("warning: index 'mad' was skipped", stderr.getvalue())
        with np.load(out, allow_pickle=False) as archive:
            self.assertEqual(archive["error_names"].tolist(), ["mad"])
            self.assertIn("mad_positive_items", archive["error_messages"][0])
            self.assertNotIn("index_names", archive.files)
            self.assertNotIn("valid_index_counts", archive.files)

    def test_output_path_validation_precedes_input_loading(self) -> None:
        missing = self.root / "missing.csv"
        cases = [
            ([], "requires --output"),
            (["--output", "-"], "requires --output"),
            (["--output", str(self.root / "result.bin")], "ending in .npz"),
        ]

        for extra, message in cases:
            with self.subTest(extra=extra):
                stderr = StringIO()
                with patch("sys.stderr", stderr):
                    code = main(["screen", str(missing), "--format", "npz", *extra])

                self.assertEqual(code, 1)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn("No such file", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_writer_rejects_object_arrays(self) -> None:
        out = self.root / "unsafe.npz"

        with self.assertRaisesRegex(ValueError, "cannot contain object arrays"):
            _write_npz_archive(out, {"unsafe": np.asarray([object()], dtype=object)})

        self.assertFalse(out.exists())
