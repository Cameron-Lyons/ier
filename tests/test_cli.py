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

from ier import (
    composite,
    composite_flag,
    composite_probability,
    composite_scores,
    composite_summary,
    irv,
    lz,
    psychant,
    psychsyn,
    save_response_time_archive,
    save_score_archive,
)
from ier._cli_input import _load_input, _load_matrix, _load_numeric_vector
from ier._cli_output import _emit_composite_json, _emit_composite_text, _write_composite_csv
from ier.cli import (
    _parse_float_list,
    _parse_int_list,
    _parse_name_list,
    _parse_pair_list,
    _parse_percentiles,
    _parse_thresholds,
    _parse_weights,
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

    def test_irv_section_options_match_direct_scores(self) -> None:
        data = np.loadtxt(self.csv_path, delimiter=",", skiprows=1)

        for option, expected in [
            (["--irv-num-split", "2"], irv(data, split=True, num_split=2)),
            (
                ["--irv-split-points", "0,2,5"],
                irv(data, split=True, split_points=[0, 2, 5]),
            ),
        ]:
            with self.subTest(option=option):
                output = self.root / f"irv-{option[0][2:]}.json"
                code = main(
                    [
                        "screen",
                        str(self.csv_path),
                        "--indices",
                        "irv",
                        *option,
                        "--format",
                        "json",
                        "--output",
                        str(output),
                    ]
                )

                self.assertEqual(code, 0)
                payload = json.loads(output.read_text(encoding="utf-8"))
                np.testing.assert_allclose(payload["scores"]["irv"], expected, rtol=0.0, atol=1e-15)

    def test_psychometric_random_seed_options_match_direct_scores(self) -> None:
        data = np.array(
            [
                [1.0, 1.0, 2.0, 3.0],
                [1.0, np.nan, 2.0, 3.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        path = self.root / "psychometric.csv"
        np.savetxt(path, data, delimiter=",")
        item_pairs = np.array([[0, 1], [0, 2], [0, 3]])
        cases = [
            ("psychsyn", "--psychsyn-random-seed", psychsyn, 17),
            ("psychant", "--psychant-random-seed", psychant, 29),
        ]

        for index, option, scorer, seed in cases:
            with self.subTest(index=index):
                output = self.root / f"{index}-seeded.json"
                with patch("ier.psychsyn._discover_item_pairs", return_value=item_pairs):
                    expected = scorer(data, resample_na=True, random_seed=seed)
                    code = main(
                        [
                            "screen",
                            str(path),
                            "--indices",
                            index,
                            option,
                            str(seed),
                            "--format",
                            "json",
                            "--output",
                            str(output),
                        ]
                    )

                self.assertEqual(code, 0)
                payload = json.loads(output.read_text(encoding="utf-8"))
                np.testing.assert_array_equal(payload["scores"][index], expected)

    def test_archive_info_auto_detects_score_metadata_in_text_and_json(self) -> None:
        archive = self.root / "scores.npz"
        save_score_archive(
            archive,
            {
                "irv": [0.9, 0.5, 0.1],
                "longstring": [2.0, 4.0, 8.0],
            },
            respondent_ids=["case-a", "case-b", "case-c"],
            errors={"mad": "missing item configuration"},
        )
        output = self.root / "archive.json"
        stdout = StringIO()

        with (
            patch("ier.cli._load_input", side_effect=AssertionError("matrix input loaded")),
            patch("sys.stdout", stdout),
        ):
            text_code = main(["archive-info", str(archive)])
            json_code = main(
                [
                    "archive-info",
                    str(archive),
                    "--format",
                    "json",
                    "--output",
                    str(output),
                ]
            )

        self.assertEqual(text_code, 0)
        self.assertEqual(json_code, 0)
        self.assertIn("result type: screen", stdout.getvalue())
        self.assertIn("indices (2): irv, longstring", stdout.getvalue())
        self.assertIn("respondent identifiers: yes", stdout.getvalue())
        self.assertIn("mad: missing item configuration", stdout.getvalue())
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")),
            {
                "schema_version": 1,
                "result_type": "screen",
                "n_respondents": 3,
                "has_respondent_ids": True,
                "n_indices": 2,
                "indices": ["irv", "longstring"],
                "errors": {"mad": "missing item configuration"},
            },
        )

    def test_archive_info_reports_response_time_cutoff_and_flags(self) -> None:
        archive = self.root / "timing.npz"
        output = self.root / "timing-info.json"
        save_response_time_archive(
            archive,
            [0.5, 1.0, 1.5],
            [True, True, False],
            threshold=1.0,
            threshold_source="fixed",
            respondent_ids=["fast", "tie", "slow"],
        )

        code = main(
            [
                "archive-info",
                str(archive),
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )

        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")),
            {
                "schema_version": 2,
                "result_type": "response_time",
                "n_respondents": 3,
                "has_respondent_ids": True,
                "metric": "median",
                "flag_direction": "low",
                "threshold": 1.0,
                "threshold_source": "fixed",
                "percentile": None,
                "n_flagged": 2,
                "flag_rate": 2 / 3,
            },
        )

    def test_screen_reflag_reuses_selected_scores_across_text_json_and_csv(self) -> None:
        archive = self.root / "retained-scores.npz"
        save_score_archive(
            archive,
            {
                "irv": [0.9, 0.5, 0.1],
                "longstring": [2.0, 4.0, 8.0],
            },
            respondent_ids=["case-a", "case-b", "case-c"],
            errors={"mad": "missing item configuration"},
        )
        json_out = self.root / "rescreened.json"
        csv_out = self.root / "rescreened.csv"
        stdout = StringIO()
        stderr = StringIO()
        decision_options = [
            "--indices",
            "longstring",
            "irv",
            "--threshold",
            "longstring=4",
            "--index-percentile",
            "irv=50",
            "--min-flags",
            "1",
            "--min-valid-indices",
            "2",
        ]

        with (
            patch("ier.cli._load_input", side_effect=AssertionError("rescored")),
            patch("sys.stderr", stderr),
        ):
            json_code = main(
                [
                    "screen-reflag",
                    str(archive),
                    *decision_options,
                    "--format",
                    "json",
                    "--output",
                    str(json_out),
                ]
            )
            csv_code = main(
                [
                    "screen-reflag",
                    str(archive),
                    *decision_options,
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            )
            with patch("sys.stdout", stdout):
                text_code = main(
                    [
                        "screen-reflag",
                        str(archive),
                        *decision_options,
                        "--top",
                        "2",
                    ]
                )

        self.assertEqual(json_code, 0)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["indices_used"], ["longstring", "irv"])
        self.assertEqual(payload["respondent_ids"], ["case-a", "case-b", "case-c"])
        self.assertEqual(payload["errors"], {"mad": "missing item configuration"})
        self.assertEqual(payload["thresholds"], {"longstring": 4.0, "irv": 0.5})
        self.assertEqual(
            payload["threshold_sources"],
            {"longstring": "fixed", "irv": "percentile"},
        )
        self.assertEqual(payload["percentiles"], {"longstring": None, "irv": 50.0})
        self.assertEqual(payload["flag_counts"], [0, 1, 2])
        self.assertEqual(payload["valid_index_counts"], [2, 2, 2])
        self.assertEqual(payload["consensus_flags"], [False, True, True])

        self.assertEqual(csv_code, 0)
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        self.assertEqual([row["respondent"] for row in rows], ["case-a", "case-b", "case-c"])
        self.assertEqual([row["consensus_flag"] for row in rows], ["0", "1", "1"])
        self.assertEqual(
            list(rows[0])[-4:],
            [
                "longstring_score",
                "longstring_flag",
                "irv_score",
                "irv_flag",
            ],
        )

        self.assertEqual(text_code, 0)
        self.assertIn("indices: longstring, irv", stdout.getvalue())
        self.assertIn("longstring=4 (fixed)", stdout.getvalue())
        self.assertIn("irv=0.5 (tail percentile=50)", stdout.getvalue())
        self.assertIn("warning: index 'mad' was skipped", stderr.getvalue())

    def test_screen_reflag_accepts_composite_score_archive(self) -> None:
        archive = self.root / "components.npz"
        save_score_archive(
            archive,
            {"irv": [0.9, 0.5, 0.1], "longstring": [2.0, 4.0, 8.0]},
            result_type="composite",
        )
        out = self.root / "rescreened-components.json"

        code = main(
            [
                "screen-reflag",
                str(archive),
                "--threshold",
                "irv=0.5",
                "--threshold",
                "longstring=4",
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
        self.assertEqual(payload["indices_used"], ["irv", "longstring"])
        self.assertEqual(payload["consensus_flags"], [False, True, True])

    def test_screen_reflag_reports_invalid_selection_without_traceback(self) -> None:
        archive = self.root / "retained-scores.npz"
        save_score_archive(archive, {"irv": [0.1, 0.5, 0.9]})

        for indices, message in [
            (["longstring"], "does not contain selected index: longstring"),
            (["irv", "irv"], "must not contain duplicates"),
        ]:
            with self.subTest(indices=indices):
                stderr = StringIO()
                with patch("sys.stderr", stderr):
                    code = main(
                        [
                            "screen-reflag",
                            str(archive),
                            "--indices",
                            *indices,
                        ]
                    )

                self.assertEqual(code, 1)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_composite_recombine_reuses_selected_scores_across_formats(self) -> None:
        archive = self.root / "retained-components.npz"
        component_scores = {
            "irv": np.asarray([0.9, 0.5, 0.1, np.nan]),
            "longstring": np.asarray([2.0, 4.0, 8.0, 4.0]),
            "person_total": np.asarray([0.8, 0.4, 0.2, 0.1]),
        }
        save_score_archive(
            archive,
            component_scores,
            result_type="composite",
            respondent_ids=["case-a", "case-b", "case-c", "case-d"],
            errors={"mad": "missing item configuration"},
        )
        json_out = self.root / "recombined.json"
        csv_out = self.root / "recombined.csv"
        stdout = StringIO()
        stderr = StringIO()
        combination_options = [
            "--indices",
            "longstring",
            "irv",
            "--method",
            "sum",
            "--no-standardize",
            "--weight",
            "longstring=2",
            "--weight",
            "irv=0.5",
            "--min-valid-indices",
            "2",
            "--threshold",
            "7.75",
            "--include-components",
            "--include-probability",
        ]

        with (
            patch("ier.cli._load_input", side_effect=AssertionError("rescored")),
            patch("sys.stderr", stderr),
        ):
            json_code = main(
                [
                    "composite-recombine",
                    str(archive),
                    *combination_options,
                    "--format",
                    "json",
                    "--output",
                    str(json_out),
                ]
            )
            csv_code = main(
                [
                    "composite-recombine",
                    str(archive),
                    *combination_options,
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            )
            with patch("sys.stdout", stdout):
                text_code = main(
                    [
                        "composite-recombine",
                        str(archive),
                        *combination_options,
                        "--top",
                        "2",
                    ]
                )

        expected = composite_scores(
            {"longstring": component_scores["longstring"], "irv": component_scores["irv"]},
            method="sum",
            standardize=False,
            weights={"longstring": 2.0, "irv": 0.5},
            min_valid_indices=2,
        )
        self.assertEqual(json_code, 0)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["method"], "sum")
        self.assertFalse(payload["standardized"])
        self.assertEqual(payload["indices_used"], ["longstring", "irv"])
        self.assertEqual(payload["respondent_ids"], ["case-a", "case-b", "case-c", "case-d"])
        self.assertEqual(payload["errors"], {"mad": "missing item configuration"})
        self.assertEqual(payload["weights"], {"longstring": 2.0, "irv": 0.5})
        np.testing.assert_allclose(payload["scores"][:3], expected[:3])
        self.assertIsNone(payload["scores"][3])
        self.assertEqual(payload["valid_index_counts"], [2, 2, 2, 1])
        self.assertEqual(list(payload["component_scores"]), ["longstring", "irv"])
        self.assertEqual(payload["threshold"], 7.75)
        self.assertEqual(payload["threshold_source"], "fixed")
        self.assertEqual(payload["flags"], [False, True, True, False])
        self.assertEqual(payload["probability_scale"], "uncalibrated_logistic")
        self.assertIsNone(payload["probabilities"][3])

        self.assertEqual(csv_code, 0)
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        self.assertEqual(
            [row["respondent"] for row in rows], ["case-a", "case-b", "case-c", "case-d"]
        )
        self.assertEqual([row["composite_flag"] for row in rows], ["0", "1", "1", "0"])
        self.assertEqual(rows[3]["composite_score"], "")
        self.assertEqual(rows[3]["valid_index_count"], "1")
        self.assertEqual(
            list(rows[0]),
            [
                "respondent",
                "composite_score",
                "composite_probability",
                "composite_flag",
                "valid_index_count",
                "longstring_score",
                "irv_score",
            ],
        )

        self.assertEqual(text_code, 0)
        self.assertIn("method: sum", stdout.getvalue())
        self.assertIn("standardized: false", stdout.getvalue())
        self.assertIn("indices: longstring, irv", stdout.getvalue())
        self.assertIn("threshold: 7.75 (fixed)", stdout.getvalue())
        self.assertIn("warning: index 'mad' was skipped", stderr.getvalue())

    def test_composite_recombine_accepts_screen_archive_and_reports_invalid_selection(
        self,
    ) -> None:
        archive = self.root / "screen-components.npz"
        save_score_archive(
            archive,
            {"irv": [0.9, 0.5, 0.1], "longstring": [2.0, 4.0, 8.0]},
            respondent_ids=["a", "b", "c"],
        )
        out = self.root / "screen-components.json"
        self.assertEqual(
            main(
                [
                    "composite-recombine",
                    str(archive),
                    "--no-standardize",
                    "--method",
                    "max",
                    "--format",
                    "json",
                    "--output",
                    str(out),
                ]
            ),
            0,
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["respondent_ids"], ["a", "b", "c"])
        np.testing.assert_allclose(payload["scores"], [2.0, 4.0, 8.0])

        invalid_cases = [
            (["missing"], "does not contain selected index: missing"),
            (["irv", "irv"], "must not contain duplicates"),
        ]
        for indices, message in invalid_cases:
            with self.subTest(indices=indices):
                stderr = StringIO()
                with patch("sys.stderr", stderr):
                    code = main(
                        [
                            "composite-recombine",
                            str(archive),
                            "--indices",
                            *indices,
                        ]
                    )
                self.assertEqual(code, 1)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_parallel_workers_across_screen_and_composite_commands(self) -> None:
        screen_out = self.root / "parallel-screen.json"
        composite_out = self.root / "parallel-composite.csv"

        screen_code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--workers",
                "2",
                "--format",
                "json",
                "--output",
                str(screen_out),
            ]
        )
        composite_code = main(
            [
                "composite",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--workers",
                "2",
                "--format",
                "csv",
                "--output",
                str(composite_out),
            ]
        )

        self.assertEqual(screen_code, 0)
        self.assertEqual(composite_code, 0)
        self.assertEqual(json.loads(screen_out.read_text(encoding="utf-8"))["n_indices"], 2)
        self.assertEqual(len(composite_out.read_text(encoding="utf-8").splitlines()), 4)

    def test_invalid_worker_count_returns_structured_error(self) -> None:
        missing = self.root / "missing.csv"
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(["screen", str(missing), "--workers", "0"])

        self.assertEqual(code, 1)
        self.assertIn("error: workers must be a positive integer", stderr.getvalue())
        self.assertNotIn("No such file", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_screen_reports_soft_errors(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
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
        self.assertIn("warning: index 'mad' was skipped", stderr.getvalue())
        self.assertIn("warning: index 'evenodd' was skipped", stderr.getvalue())

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

    def test_composite_standardization_can_be_disabled_across_formats(self) -> None:
        matrix, _ = _load_input(self.csv_path, None, None)
        expected_details = composite_summary(
            matrix,
            indices=["irv", "longstring"],
            standardize=False,
        )
        expected = composite(
            matrix,
            indices=["irv", "longstring"],
            standardize=False,
        )
        self.assertIsInstance(expected, np.ndarray)
        np.testing.assert_allclose(expected_details["composite"], expected)

        json_out = self.root / "raw-composite.json"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--no-standardize",
                    "--include-components",
                    "--format",
                    "json",
                    "--output",
                    str(json_out),
                ]
            ),
            0,
        )
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertIs(payload["standardized"], False)
        np.testing.assert_allclose(payload["scores"], expected)
        for name in expected_details["indices_used"]:
            np.testing.assert_allclose(
                payload["component_scores"][name],
                expected_details["indices"][name],
            )

        csv_out = self.root / "raw-composite.csv"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--no-standardize",
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            ),
            0,
        )
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        np.testing.assert_allclose(
            [float(row["composite_score"]) for row in rows],
            expected,
        )

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(
                main(
                    [
                        "composite",
                        str(self.csv_path),
                        "--indices",
                        "irv",
                        "longstring",
                        "--no-standardize",
                    ]
                ),
                0,
            )
        self.assertIn("standardized: false", stdout.getvalue())

    def test_composite_probability_is_opt_in_across_formats_without_rescoring(self) -> None:
        matrix, _ = _load_input(self.csv_path, None, None)
        indices = ["irv", "longstring"]
        expected_scores = composite(matrix, indices=indices)
        expected_probabilities = composite_probability(matrix, indices=indices)
        self.assertIsInstance(expected_scores, np.ndarray)

        json_out = self.root / "probability-composite.json"
        with patch("ier.cli.composite", wraps=composite) as score_mock:
            self.assertEqual(
                main(
                    [
                        "composite",
                        str(self.csv_path),
                        "--indices",
                        *indices,
                        "--include-probability",
                        "--threshold",
                        "0",
                        "--format",
                        "json",
                        "--output",
                        str(json_out),
                    ]
                ),
                0,
            )
        self.assertEqual(score_mock.call_count, 1)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["probability_scale"], "uncalibrated_logistic")
        np.testing.assert_allclose(payload["scores"], expected_scores)
        np.testing.assert_allclose(payload["probabilities"], expected_probabilities)
        self.assertEqual(payload["flags"], (expected_scores >= 0.0).tolist())

        csv_out = self.root / "probability-composite.csv"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    *indices,
                    "--include-probability",
                    "--include-components",
                    "--threshold",
                    "0",
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            ),
            0,
        )
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        self.assertEqual(
            list(rows[0]),
            [
                "respondent",
                "composite_score",
                "composite_probability",
                "composite_flag",
                "valid_index_count",
                "irv_score",
                "longstring_score",
            ],
        )
        np.testing.assert_allclose(
            [float(row["composite_probability"]) for row in rows],
            expected_probabilities,
        )

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(
                main(
                    [
                        "composite",
                        str(self.csv_path),
                        "--indices",
                        *indices,
                        "--include-probability",
                        "--top",
                        "3",
                    ]
                ),
                0,
            )
        text = stdout.getvalue()
        self.assertIn("probability: logistic (uncalibrated)", text)
        self.assertIn("index, score, probability", text)

    def test_composite_fixed_threshold_flags_across_formats(self) -> None:
        matrix, _ = _load_input(self.csv_path, None, None)
        expected_scores, expected_flags = composite_flag(
            matrix,
            indices=["irv", "longstring"],
            threshold=0.0,
        )

        json_out = self.root / "flagged-composite.json"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--threshold",
                    "0",
                    "--include-components",
                    "--format",
                    "json",
                    "--output",
                    str(json_out),
                ]
            ),
            0,
        )
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        np.testing.assert_allclose(payload["scores"], expected_scores)
        self.assertEqual(payload["flags"], expected_flags.tolist())
        self.assertEqual(payload["threshold"], 0.0)
        self.assertEqual(payload["threshold_source"], "fixed")
        self.assertNotIn("percentile", payload)

        csv_out = self.root / "flagged-composite.csv"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--threshold",
                    "0",
                    "--include-components",
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            ),
            0,
        )
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        self.assertEqual(
            list(rows[0]),
            [
                "respondent",
                "composite_score",
                "composite_flag",
                "valid_index_count",
                "irv_score",
                "longstring_score",
            ],
        )
        self.assertEqual(
            [row["composite_flag"] for row in rows],
            [str(int(value)) for value in expected_flags],
        )

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(
                main(
                    [
                        "composite",
                        str(self.csv_path),
                        "--indices",
                        "irv",
                        "longstring",
                        "--threshold",
                        "0",
                        "--top",
                        "3",
                    ]
                ),
                0,
            )
        text = stdout.getvalue()
        self.assertIn("threshold: 0 (fixed)", text)
        self.assertIn(f"flagged: {int(np.sum(expected_flags))}", text)
        self.assertIn("index, score, flag", text)

    def test_composite_percentile_flags_record_strict_tail_rule(self) -> None:
        matrix, _ = _load_input(self.csv_path, None, None)
        expected_scores, expected_flags = composite_flag(
            matrix,
            indices=["irv", "longstring"],
            percentile=50.0,
        )
        out = self.root / "percentile-composite.json"

        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--percentile",
                    "50",
                    "--format",
                    "json",
                    "--output",
                    str(out),
                ]
            ),
            0,
        )

        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["threshold_source"], "percentile")
        self.assertEqual(payload["percentile"], 50.0)
        self.assertAlmostEqual(payload["threshold"], float(np.percentile(expected_scores, 50.0)))
        self.assertEqual(payload["flags"], expected_flags.tolist())

    def test_composite_soft_errors_are_visible_in_text_json_and_csv(self) -> None:
        json_out = self.root / "partial-composite.json"
        json_stderr = StringIO()
        with patch("sys.stderr", json_stderr):
            json_code = main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "mad",
                    "--include-components",
                    "--format",
                    "json",
                    "--output",
                    str(json_out),
                ]
            )

        self.assertEqual(json_code, 0)
        json_payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(list(json_payload["errors"]), ["mad"])
        self.assertIn("mad_positive_items", json_payload["errors"]["mad"])
        self.assertEqual(json_payload["indices_used"], ["irv"])
        self.assertEqual(list(json_payload["component_scores"]), ["irv"])
        self.assertEqual(json_payload["valid_index_counts"], [1, 1, 1])
        self.assertIn("warning: index 'mad' was skipped", json_stderr.getvalue())

        text_stdout = StringIO()
        text_stderr = StringIO()
        with patch("sys.stdout", text_stdout), patch("sys.stderr", text_stderr):
            text_code = main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "mad",
                    "--top",
                    "1",
                ]
            )

        self.assertEqual(text_code, 0)
        self.assertIn("errors:\n  mad: mad_positive_items", text_stdout.getvalue())
        self.assertIn("warning: index 'mad' was skipped", text_stderr.getvalue())

        csv_out = self.root / "partial-composite.csv"
        csv_stderr = StringIO()
        with patch("sys.stderr", csv_stderr):
            csv_code = main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "mad",
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            )

        self.assertEqual(csv_code, 0)
        self.assertTrue(
            csv_out.read_text(encoding="utf-8").startswith("respondent,composite_score")
        )
        self.assertIn("warning: index 'mad' was skipped", csv_stderr.getvalue())

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
        self.assertEqual(payload["threshold_source"], "fixed")
        self.assertNotIn("percentile", payload)
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

    def test_response_time_json_records_requested_percentile(self) -> None:
        timings = self.root / "percentile-timings.csv"
        timings.write_text("1,2,3\n2,3,4\n3,4,5\n", encoding="utf-8")
        out = self.root / "percentile-timings.json"

        code = main(
            [
                "response-time",
                str(timings),
                "--percentile",
                "50",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["threshold_source"], "percentile")
        self.assertEqual(payload["percentile"], 50.0)
        self.assertEqual(payload["threshold"], 3.0)
        self.assertEqual(payload["flags"], [True, False, False])

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
        self.assertIn("threshold: 0.55 (percentile)", text)
        self.assertIn("percentile: 5", text)
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

    def test_response_time_reflag_reuses_legacy_archive_across_text_json_and_csv(
        self,
    ) -> None:
        archive = self.root / "retained-timing.npz"
        scores = np.asarray([0.5, 1.0, 1.0, 3.0])
        save_response_time_archive(
            archive,
            scores,
            scores <= 1.0,
            threshold=1.0,
            respondent_ids=["fast", "tie-a", "tie-b", "slow"],
        )
        json_out = self.root / "reflagged.json"
        csv_out = self.root / "reflagged.csv"
        stdout = StringIO()

        with patch("ier.cli._load_input", side_effect=AssertionError("rescored")):
            json_code = main(
                [
                    "response-time-reflag",
                    str(archive),
                    "--percentile",
                    "50",
                    "--format",
                    "json",
                    "--output",
                    str(json_out),
                ]
            )
            csv_code = main(
                [
                    "response-time-reflag",
                    str(archive),
                    "--threshold",
                    "1",
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            )
            with patch("sys.stdout", stdout):
                text_code = main(
                    [
                        "response-time-reflag",
                        str(archive),
                        "--percentile",
                        "50",
                        "--top",
                        "2",
                    ]
                )

        self.assertEqual(json_code, 0)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["metric"], "median")
        self.assertEqual(payload["flag_direction"], "low")
        self.assertEqual(payload["threshold"], 1.0)
        self.assertEqual(payload["threshold_source"], "percentile")
        self.assertEqual(payload["percentile"], 50.0)
        self.assertEqual(payload["scores"], scores.tolist())
        self.assertEqual(payload["flags"], [True, False, False, False])
        self.assertEqual(payload["respondent_ids"], ["fast", "tie-a", "tie-b", "slow"])

        self.assertEqual(csv_code, 0)
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        self.assertEqual([row["respondent"] for row in rows], ["fast", "tie-a", "tie-b", "slow"])
        self.assertEqual([row["response_time_flag"] for row in rows], ["1", "1", "1", "0"])

        self.assertEqual(text_code, 0)
        self.assertIn("threshold: 1 (percentile)", stdout.getvalue())
        self.assertIn("percentile: 50", stdout.getvalue())

    def test_response_time_reflag_preserves_mixture_high_tail(self) -> None:
        archive = self.root / "mixture-timing.npz"
        scores = np.asarray([0.1, 0.7, 0.7, 0.9])
        save_response_time_archive(
            archive,
            scores,
            scores >= 0.7,
            threshold=0.7,
            metric="mixture",
            flag_direction="high",
        )
        out = self.root / "mixture-reflagged.json"

        code = main(
            [
                "response-time-reflag",
                str(archive),
                "--percentile",
                "50",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["metric"], "mixture")
        self.assertEqual(payload["flag_direction"], "high")
        self.assertEqual(payload["threshold"], 0.7)
        self.assertEqual(payload["flags"], [False, False, False, True])

    def test_response_time_reflag_reports_invalid_archive_without_traceback(self) -> None:
        invalid = self.root / "invalid.npz"
        np.savez(invalid, values=np.asarray([1.0, 2.0]))
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            code = main(["response-time-reflag", str(invalid), "--threshold", "1"])

        self.assertEqual(code, 1)
        self.assertIn("missing required member: schema_version", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

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

    def test_csv_commands_stream_to_plain_gzip_and_standard_output(self) -> None:
        screen_out = self.root / "screen.csv"
        composite_out = self.root / "composite.csv.gz"
        indices_out = self.root / "indices.csv"
        stdout = StringIO()

        with patch("sys.stdout", stdout):
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
            self.assertEqual(
                main(
                    [
                        "composite",
                        str(self.csv_path),
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
            self.assertEqual(
                main(
                    [
                        "response-time",
                        str(self.csv_path),
                        "--format",
                        "csv",
                        "--output",
                        "-",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(["indices", "--format", "csv", "--output", str(indices_out)]),
                0,
            )

        self.assertFalse(stdout.closed)
        self.assertEqual(len(stdout.getvalue().splitlines()), 4)
        self.assertTrue(stdout.getvalue().startswith("respondent,response_time_score"))
        self.assertTrue(screen_out.read_text(encoding="utf-8").startswith("respondent,flag_count"))
        self.assertTrue(indices_out.read_text(encoding="utf-8").startswith("index,flag_direction"))
        with gzip.open(composite_out, mode="rt", encoding="utf-8") as handle:
            self.assertTrue(handle.read().startswith("respondent,composite_score"))

    def test_screen_csv_writer_emits_rows_individually(self) -> None:
        out = self.root / "screen.csv"
        with patch(
            "ier._cli_output.csv.DictWriter.writerows",
            side_effect=AssertionError("buffered"),
        ):
            code = main(
                [
                    "screen",
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
        self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 4)

    def test_matrix_loader_converts_rows_incrementally(self) -> None:
        converted_cells = 0

        def iter_rows(path: Path, delimiter: str | None) -> object:
            del path, delimiter
            yield ["1", "2"]
            self.assertEqual(converted_cells, 2)
            yield ["3", "4"]

        def parse_cell(cell: str) -> float:
            nonlocal converted_cells
            converted_cells += 1
            return float(cell)

        with (
            patch("ier._cli_input._iter_rows", side_effect=iter_rows),
            patch("ier._cli_input._parse_numeric_cell", side_effect=parse_cell),
        ):
            matrix = _load_matrix(Path("unused.csv"), None)

        np.testing.assert_array_equal(matrix, [[1.0, 2.0], [3.0, 4.0]])
        matrix[0, 0] = 9.0
        self.assertEqual(matrix[0, 0], 9.0)

    def test_npy_input_is_memory_mapped_and_scores(self) -> None:
        path = self.root / "responses.npy"
        expected = np.array(
            [[1, 1, 1, 1, 1], [1, 2, 3, 4, 5], [5, 5, 5, 1, 2]],
            dtype=np.float64,
        )
        np.save(path, expected)

        matrix = _load_matrix(path, None)

        self.assertIsInstance(matrix, np.memmap)
        self.assertFalse(matrix.flags.writeable)
        np.testing.assert_array_equal(matrix, expected)

        out = self.root / "screen.json"
        code = main(
            [
                "screen",
                str(path),
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
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(payload["scores"]), {"irv", "longstring"})
        self.assertEqual(len(payload["consensus_flags"]), expected.shape[0])

    def test_npy_input_rejects_header_options_and_delimiter(self) -> None:
        path = self.root / "responses.npy"
        np.save(path, np.ones((2, 3), dtype=np.float64))

        for option in [
            ["--delimiter", ","],
            ["--id-column", "participant"],
            ["--item-columns", "i1,i2"],
        ]:
            with self.subTest(option=option):
                stderr = StringIO()
                with patch("sys.stderr", stderr):
                    code = main(["screen", str(path), "--indices", "irv", *option])

                self.assertEqual(code, 1)
                self.assertIn(".npy input", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_npy_input_rejects_invalid_arrays(self) -> None:
        cases = [
            ("vector", np.array([1.0, 2.0]), "non-empty 2D"),
            ("empty", np.empty((0, 2)), "non-empty 2D"),
            ("complex", np.array([[1 + 2j]]), "real numeric"),
            ("text", np.array([["one", "two"]]), "real numeric"),
        ]

        for name, values, message in cases:
            with self.subTest(name=name):
                path = self.root / f"{name}.npy"
                np.save(path, values)

                with self.assertRaisesRegex(ValueError, message):
                    _load_matrix(path, None)

    def test_npy_input_rejects_malformed_files_and_archives(self) -> None:
        malformed = self.root / "malformed.npy"
        malformed.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "failed to load NumPy matrix"):
            _load_matrix(malformed, None)

        archive = self.root / "archive.npy"
        with archive.open("wb") as handle:
            np.savez(handle, responses=np.ones((2, 3), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "not an archive"):
            _load_matrix(archive, None)

    def test_numeric_vector_loader_accepts_safe_binary_and_text_shapes(self) -> None:
        expected = np.linspace(-2.0, 2.0, 5)
        npy_path = self.root / "values.npy"
        row_path = self.root / "values.csv"
        column_path = self.root / "values.txt.gz"
        np.save(npy_path, expected)
        np.savetxt(row_path, expected[np.newaxis], delimiter=",")
        with gzip.open(column_path, mode="wt", encoding="utf-8") as handle:
            np.savetxt(handle, expected)

        npy_values = _load_numeric_vector(npy_path, "test vector")
        row_values = _load_numeric_vector(row_path, "test vector")
        column_values = _load_numeric_vector(column_path, "test vector")

        self.assertIsInstance(npy_values, np.memmap)
        np.testing.assert_array_equal(npy_values, expected)
        np.testing.assert_array_equal(row_values, expected)
        np.testing.assert_array_equal(column_values, expected)

    def test_numeric_vector_loader_rejects_matrices_pickle_and_standard_input(self) -> None:
        matrix = self.root / "matrix.npy"
        objects = self.root / "objects.npy"
        complex_values = self.root / "complex.npy"
        compressed = self.root / "compressed.npy.gz"
        np.save(matrix, np.ones((2, 2)))
        np.save(objects, np.array([{"unsafe": True}], dtype=object))
        np.save(complex_values, np.array([1.0 + 2.0j]))
        with gzip.open(compressed, mode="wb") as handle:
            np.save(handle, np.ones(2))

        with self.assertRaisesRegex(ValueError, "non-empty one-dimensional test vector"):
            _load_numeric_vector(matrix, "test vector")
        with self.assertRaisesRegex(ValueError, "failed to load test vector"):
            _load_numeric_vector(objects, "test vector")
        with self.assertRaisesRegex(ValueError, "real numeric test vector"):
            _load_numeric_vector(complex_values, "test vector")
        with self.assertRaisesRegex(ValueError, "compressed .npy test vector"):
            _load_numeric_vector(compressed, "test vector")
        with self.assertRaisesRegex(ValueError, "cannot use standard input"):
            _load_numeric_vector(Path("-"), "test vector")

    def test_calibrated_lz_parameter_files_match_direct_screen_and_composite(self) -> None:
        matrix, _ = _load_input(self.csv_path, None)
        difficulty = np.linspace(-1.5, 1.5, matrix.shape[1])
        discrimination = np.linspace(0.6, 1.8, matrix.shape[1])
        theta = np.linspace(-1.0, 1.0, matrix.shape[0])
        difficulty_path = self.root / "difficulty.csv"
        discrimination_path = self.root / "discrimination.npy"
        theta_path = self.root / "theta.txt.gz"
        np.savetxt(difficulty_path, difficulty[np.newaxis], delimiter=",")
        np.save(discrimination_path, discrimination)
        with gzip.open(theta_path, mode="wt", encoding="utf-8") as handle:
            np.savetxt(handle, theta)
        expected = lz(
            matrix,
            difficulty=difficulty,
            discrimination=discrimination,
            theta=theta,
        )
        shared_arguments = [
            "--indices",
            "lz",
            "--lz-difficulty",
            str(difficulty_path),
            "--lz-discrimination",
            str(discrimination_path),
            "--lz-theta",
            str(theta_path),
            "--lz-model",
            "2pl",
            "--format",
            "json",
        ]

        screen_out = self.root / "calibrated-screen.json"
        composite_out = self.root / "calibrated-composite.json"
        one_pl_out = self.root / "calibrated-one-pl.json"
        screen_code = main(
            ["screen", str(self.csv_path), *shared_arguments, "--output", str(screen_out)]
        )
        composite_code = main(
            [
                "composite",
                str(self.csv_path),
                *shared_arguments,
                "--no-standardize",
                "--output",
                str(composite_out),
            ]
        )
        one_pl_code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "lz",
                "--lz-difficulty",
                str(difficulty_path),
                "--lz-model",
                "1pl",
                "--format",
                "json",
                "--output",
                str(one_pl_out),
            ]
        )

        self.assertEqual(screen_code, 0)
        self.assertEqual(composite_code, 0)
        self.assertEqual(one_pl_code, 0)
        screen_payload = json.loads(screen_out.read_text(encoding="utf-8"))
        composite_payload = json.loads(composite_out.read_text(encoding="utf-8"))
        one_pl_payload = json.loads(one_pl_out.read_text(encoding="utf-8"))
        np.testing.assert_array_equal(screen_payload["scores"]["lz"], expected)
        np.testing.assert_array_equal(composite_payload["scores"], -expected)
        np.testing.assert_array_equal(
            one_pl_payload["scores"]["lz"],
            lz(matrix, difficulty=difficulty, model="1pl"),
        )

    def test_invalid_lz_parameter_file_returns_structured_error(self) -> None:
        difficulty = self.root / "difficulty.npy"
        np.save(difficulty, np.ones((2, 2)))
        stderr = StringIO()

        with (
            patch("ier.cli._load_input", side_effect=AssertionError("matrix input loaded")),
            patch("sys.stderr", stderr),
        ):
            code = main(
                [
                    "screen",
                    "unused.csv",
                    "--indices",
                    "lz",
                    "--lz-difficulty",
                    str(difficulty),
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("non-empty one-dimensional LZ difficulty vector", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_compressed_npy_input_returns_structured_error(self) -> None:
        path = self.root / "responses.npy.gz"
        with gzip.open(path, mode="wb") as handle:
            np.save(handle, np.ones((2, 3), dtype=np.float64))
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            code = main(["screen", str(path), "--indices", "irv"])

        self.assertEqual(code, 1)
        self.assertIn("compressed .npy input is not supported", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

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

    def test_screen_missing_rate_required_item_subset(self) -> None:
        missing = self.root / "missing-rate-subset.csv"
        missing.write_text("i1,i2,i3\n1,2,3\n1,,3\n,,3\n", encoding="utf-8")
        out = self.root / "missing-rate-subset.json"

        code = main(
            [
                "screen",
                str(missing),
                "--indices",
                "missing_rate",
                "--missing-item-indices",
                "0,2",
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
        np.testing.assert_allclose(payload["scores"]["missing_rate"], [0.0, 0.0, 0.5])
        self.assertEqual(payload["consensus_flags"], [False, False, True])

    def test_screen_infrequency_available_case_policy(self) -> None:
        attention = self.root / "attention-checks.csv"
        attention.write_text(
            "i1,i2,metadata\n5,1,9\n,1,9\n1,,9\n,,9\n",
            encoding="utf-8",
        )
        out = self.root / "attention-checks.json"

        code = main(
            [
                "screen",
                str(attention),
                "--indices",
                "infrequency",
                "--infrequency-item-indices",
                "0,1",
                "--infrequency-expected-responses",
                "5,1",
                "--infrequency-proportion",
                "--infrequency-missing",
                "omit",
                "--threshold",
                "infrequency=0.5",
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
        self.assertEqual(payload["scores"]["infrequency"], [0.0, 0.0, 1.0, None])
        self.assertEqual(payload["consensus_flags"], [False, False, True, False])

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

    def test_composite_weights_are_applied_and_reported(self) -> None:
        out = self.root / "weighted-composite.json"
        code = main(
            [
                "composite",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--weight",
                "irv=3",
                "--weight",
                "longstring=0.5",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["weights"], {"irv": 3.0, "longstring": 0.5})
        matrix, _ = _load_input(self.csv_path, None, None)
        expected = composite(
            matrix,
            indices=["irv", "longstring"],
            weights={"irv": 3.0, "longstring": 0.5},
        )
        np.testing.assert_allclose(payload["scores"], expected)

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            text_code = main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--weight",
                    "irv=3",
                ]
            )
        self.assertEqual(text_code, 0)
        self.assertIn("weights: irv=3", stdout.getvalue())

    def test_composite_minimum_valid_indices_are_applied_and_reported(self) -> None:
        incomplete = self.root / "incomplete-composite.csv"
        incomplete.write_text(
            "i1,i2,i3,i4\n1,2,3,4\n1,,,\n",
            encoding="utf-8",
        )
        out = self.root / "complete-enough.json"

        code = main(
            [
                "composite",
                str(incomplete),
                "--indices",
                "longstring",
                "markov",
                "--min-valid-indices",
                "2",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["min_valid_indices"], 2)
        self.assertIsInstance(payload["scores"][0], float)
        self.assertIsNone(payload["scores"][1])

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            text_code = main(
                [
                    "composite",
                    str(incomplete),
                    "--indices",
                    "longstring",
                    "markov",
                    "--min-valid-indices",
                    "2",
                ]
            )
        self.assertEqual(text_code, 0)
        self.assertIn("minimum valid indices: 2", stdout.getvalue())

    def test_composite_components_are_available_in_json_csv_and_text(self) -> None:
        json_out = self.root / "component-composite.json"
        code = main(
            [
                "composite",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--include-components",
                "--format",
                "json",
                "--output",
                str(json_out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertIs(payload["standardized"], True)
        self.assertEqual(payload["indices_used"], ["irv", "longstring"])
        self.assertEqual(list(payload["component_scores"]), ["irv", "longstring"])
        self.assertEqual(payload["valid_index_counts"], [2, 2, 2])

        matrix, _ = _load_input(self.csv_path, None, None)
        expected = composite_summary(matrix, indices=["irv", "longstring"])
        np.testing.assert_allclose(payload["scores"], expected["composite"])
        for name in expected["indices_used"]:
            np.testing.assert_allclose(payload["component_scores"][name], expected["indices"][name])

        csv_out = self.root / "component-composite.csv"
        self.assertEqual(
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--include-components",
                    "--format",
                    "csv",
                    "--output",
                    str(csv_out),
                ]
            ),
            0,
        )
        rows = list(csv.DictReader(StringIO(csv_out.read_text(encoding="utf-8"))))
        self.assertEqual(
            list(rows[0]),
            ["respondent", "composite_score", "valid_index_count", "irv_score", "longstring_score"],
        )
        self.assertEqual([row["valid_index_count"] for row in rows], ["2", "2", "2"])

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            text_code = main(
                [
                    "composite",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--include-components",
                    "--top",
                    "1",
                ]
            )
        self.assertEqual(text_code, 0)
        self.assertIn("indices: irv, longstring", stdout.getvalue())
        self.assertIn("index, score, valid_indices, irv, longstring", stdout.getvalue())

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
        self.assertEqual(payload["threshold_sources"], {"longstring": "fixed"})
        self.assertEqual(payload["percentiles"], {"longstring": None})
        self.assertEqual(payload["flags"]["longstring"], [True, True, True])
        self.assertEqual(payload["consensus_flags"], [True, True, True])

    def test_screen_per_index_percentiles_and_provenance(self) -> None:
        out = self.root / "screen-percentiles.json"
        code = main(
            [
                "screen",
                str(self.csv_path),
                "--indices",
                "irv",
                "longstring",
                "--index-percentile",
                "irv=80",
                "--index-percentile",
                "longstring=99",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["threshold_sources"], {"irv": "percentile", "longstring": "percentile"}
        )
        self.assertEqual(payload["percentiles"], {"irv": 80.0, "longstring": 99.0})
        self.assertAlmostEqual(
            payload["thresholds"]["irv"],
            float(np.percentile(payload["scores"]["irv"], 20)),
        )
        self.assertAlmostEqual(
            payload["thresholds"]["longstring"],
            float(np.percentile(payload["scores"]["longstring"], 99)),
        )

    def test_screen_rejects_conflicting_cutoff_overrides(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(
                [
                    "screen",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "--threshold",
                    "irv=0.5",
                    "--index-percentile",
                    "irv=90",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("both a threshold and percentile", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_screen_text_reports_cutoff_provenance(self) -> None:
        stdout = StringIO()
        with patch("sys.stdout", stdout):
            code = main(
                [
                    "screen",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--index-percentile",
                    "irv=80",
                    "--threshold",
                    "longstring=2",
                    "--top",
                    "0",
                ]
            )

        self.assertEqual(code, 0)
        self.assertIn("(tail percentile=80)", stdout.getvalue())
        self.assertIn("longstring=2 (fixed)", stdout.getvalue())
        self.assertIn("index coverage:", stdout.getvalue())
        self.assertIn("irv: valid=3/3, unavailable=0", stdout.getvalue())

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

    def test_screen_minimum_valid_indices_controls_consensus(self) -> None:
        source = self.root / "coverage-screen.csv"
        source.write_text(
            "i1,i2,i3,i4\n5,1,1,1\n,,1,2\n,,3,3\n",
            encoding="utf-8",
        )
        out = self.root / "coverage-screen.json"

        code = main(
            [
                "screen",
                str(source),
                "--indices",
                "infrequency",
                "longstring",
                "--infrequency-item-indices",
                "0,1",
                "--infrequency-expected-responses",
                "5,1",
                "--infrequency-missing",
                "omit",
                "--threshold",
                "infrequency=1",
                "--threshold",
                "longstring=2",
                "--min-flags",
                "1",
                "--min-valid-indices",
                "2",
                "--format",
                "json",
                "--output",
                str(out),
            ]
        )

        self.assertEqual(code, 0)
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["min_valid_indices"], 2)
        self.assertEqual(payload["valid_index_counts"], [2, 1, 1])
        self.assertEqual(payload["consensus_eligible"], [True, False, False])
        self.assertEqual(payload["flag_counts"], [1, 0, 1])
        self.assertEqual(payload["consensus_flags"], [True, False, False])

    def test_screen_invalid_consensus_threshold_returns_structured_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(["screen", str(self.csv_path), "--min-flags", "0"])

        self.assertEqual(code, 1)
        self.assertIn("error: min_flags must be a positive integer", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_screen_invalid_minimum_valid_indices_returns_structured_error(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            code = main(
                [
                    "screen",
                    str(self.csv_path),
                    "--indices",
                    "irv",
                    "longstring",
                    "--min-valid-indices",
                    "3",
                ]
            )

        self.assertEqual(code, 1)
        self.assertIn("error: min_valid_indices cannot exceed", stderr.getvalue())
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
        self.assertEqual(payload["summary"]["psychsyn"]["n_valid"], 0)
        self.assertEqual(payload["summary"]["psychsyn"]["n_unavailable"], 3)
        self.assertIsNone(payload["summary"]["psychsyn"]["flag_rate"])

    def test_composite_json_uses_null_for_all_non_finite_values(self) -> None:
        text = _emit_composite_json(
            np.array([1.0, np.nan, np.inf, -np.inf], dtype=float),
            "mean",
        )

        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(json.loads(text)["scores"], [1.0, None, None, None])

    def test_composite_text_ranking_excludes_non_finite_scores(self) -> None:
        text = _emit_composite_text(
            np.array([1.0, np.nan, np.inf, -np.inf], dtype=float),
            "mean",
            4,
        )

        self.assertIn("\n  0\t1.000000", text)
        self.assertNotIn("\n  1\t", text)
        self.assertNotIn("\n  2\t", text)
        self.assertNotIn("\n  3\t", text)

    def test_composite_component_arrays_must_be_respondent_aligned(self) -> None:
        scores = np.array([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            _emit_composite_json(scores, "mean", component_scores={"irv": scores})
        with self.assertRaisesRegex(ValueError, "valid index count length"):
            _write_composite_csv(
                StringIO(),
                scores,
                component_scores={"irv": scores},
                valid_index_counts=np.array([1]),
            )
        with self.assertRaisesRegex(ValueError, "component score length for irv"):
            _emit_composite_text(
                scores,
                "mean",
                2,
                component_scores={"irv": np.array([1.0])},
                valid_index_counts=np.array([1, 1]),
            )
        with self.assertRaisesRegex(ValueError, "flags and threshold"):
            _emit_composite_json(scores, "mean", flags=np.array([True, False]))
        with self.assertRaisesRegex(ValueError, "percentile requires"):
            _emit_composite_json(scores, "mean", flag_percentile=95.0)
        with self.assertRaisesRegex(ValueError, "flag length"):
            _emit_composite_json(
                scores,
                "mean",
                flags=np.array([True]),
                flag_threshold=1.0,
            )
        with self.assertRaisesRegex(ValueError, "threshold must be finite"):
            _emit_composite_json(
                scores,
                "mean",
                flags=np.array([True, False]),
                flag_threshold=np.nan,
            )
        with self.assertRaisesRegex(ValueError, "percentile must be between"):
            _emit_composite_json(
                scores,
                "mean",
                flags=np.array([True, False]),
                flag_threshold=1.0,
                flag_percentile=101.0,
            )
        with self.assertRaisesRegex(ValueError, "flag length"):
            _write_composite_csv(
                StringIO(),
                scores,
                flags=np.array([True]),
            )
        with self.assertRaisesRegex(ValueError, "probability length"):
            _emit_composite_json(
                scores,
                "mean",
                probabilities=np.array([0.5]),
            )

    def test_composite_csv_uses_empty_cells_for_all_non_finite_values(self) -> None:
        output = StringIO()
        _write_composite_csv(
            output,
            np.array([1.0, np.nan, np.inf, -np.inf], dtype=float),
        )

        self.assertEqual(
            list(csv.reader(StringIO(output.getvalue()))),
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
        self.assertEqual([row["valid_index_count"] for row in rows], ["0", "0", "0"])
        self.assertEqual([row["consensus_eligible"] for row in rows], ["1", "1", "1"])
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
                "mad",
                "--evenodd-factors",
                "2,3",
                "--semantic-item-pairs",
                "0,1;2,3",
                "--mad-positive-items",
                "0,2",
                "--mad-negative-items",
                "1,3",
                "--mad-scale-min",
                "0.5",
                "--mad-scale-max",
                "5.5",
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
        self.assertIsNone(_parse_percentiles(None))
        self.assertEqual(
            _parse_thresholds(["irv=0.5", "longstring = 4"]), {"irv": 0.5, "longstring": 4.0}
        )
        self.assertEqual(
            _parse_percentiles(["irv=90", "longstring = 99"]),
            {"irv": 90.0, "longstring": 99.0},
        )
        with self.assertRaises(ValueError):
            _parse_pair_list("0-1")
        with self.assertRaises(ValueError):
            _parse_thresholds(["irv"])
        with self.assertRaises(ValueError):
            _parse_thresholds(["irv=1", "irv=2"])
        with self.assertRaises(ValueError):
            _parse_percentiles(["irv=90", "irv=95"])
        with self.assertRaises(ValueError):
            _parse_name_list([" , "])
        self.assertIsNone(_parse_weights(None))
        self.assertEqual(
            _parse_weights(["irv=2", "longstring = 0.5"]),
            {"irv": 2.0, "longstring": 0.5},
        )
        with self.assertRaises(ValueError):
            _parse_weights(["irv=1", "irv=2"])

    def test_invalid_composite_weights_return_structured_errors(self) -> None:
        cases = [
            (["--weight", "irv=0"], "positive finite"),
            (["--weight", "irv=nan"], "positive finite"),
            (["--weight", "irv=1", "--weight", "irv=2"], "duplicate weight"),
            (["--weight", "mahad=2"], "not selected"),
            (["--min-valid-indices", "0"], "positive integer"),
            (["--min-valid-indices", "3"], "cannot exceed"),
        ]
        for extra, message in cases:
            stderr = StringIO()
            with self.subTest(extra=extra), patch("sys.stderr", stderr):
                code = main(
                    [
                        "composite",
                        str(self.csv_path),
                        "--indices",
                        "irv",
                        "longstring",
                        *extra,
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn(message, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

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

    def test_invalid_composite_flagging_values_return_structured_errors(self) -> None:
        for option, value, message in [
            ("--percentile", "101", "percentile"),
            ("--threshold", "nan", "threshold"),
        ]:
            stderr = StringIO()
            with self.subTest(option=option), patch("sys.stderr", stderr):
                code = main(["composite", str(self.csv_path), option, value])
            self.assertEqual(code, 1)
            self.assertIn(message, stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_composite_fixed_and_percentile_cutoffs_are_mutually_exclusive(self) -> None:
        stderr = StringIO()
        with patch("sys.stderr", stderr), self.assertRaises(SystemExit) as raised:
            main(
                [
                    "composite",
                    str(self.csv_path),
                    "--threshold",
                    "0",
                    "--percentile",
                    "95",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())

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

        with patch("ier._cli_input.csv.Sniffer.sniff", side_effect=csv.Error("nope")):
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
