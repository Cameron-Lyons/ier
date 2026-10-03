"""Exercise an installed distribution's API, CLI, and reusable-score archives."""

from __future__ import annotations

import csv
import gzip
import json
import shutil
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np

import ier


def _run_cli(executable: str, *arguments: str) -> str:
    completed = subprocess.run(
        [executable, *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout


def _check_scoring(executable: str) -> None:
    responses = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, np.nan, 4.0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    )
    indices = ["irv", "longstring", "missing_rate"]
    thresholds = {"irv": 0.25, "longstring": 3.0, "missing_rate": 0.25}
    expected_scores = {
        "irv": [0.0, np.sqrt(1.25), 1.0, np.nan],
        "longstring": [4.0, 1.0, 1.0, 0.0],
        "missing_rate": [0.0, 0.0, 0.5, 1.0],
    }
    result = ier.screen(
        responses,
        indices=indices,
        thresholds=thresholds,
        min_flags=2,
        min_valid_indices=3,
        strict=True,
    )
    for name, expected in expected_scores.items():
        np.testing.assert_allclose(result["scores"][name], expected, equal_nan=True)
    np.testing.assert_array_equal(result["flag_counts"], [2, 0, 1, 1])
    np.testing.assert_array_equal(result["valid_index_counts"], [3, 3, 3, 2])
    np.testing.assert_array_equal(result["consensus_eligible"], [True, True, True, False])
    np.testing.assert_array_equal(result["consensus_flags"], [True, False, False, False])
    if result["errors"]:
        raise RuntimeError(f"installed scoring failed: {result['errors']}")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "responses.csv.gz"
        identifiers = ["constant", "varied", "partial", "missing"]
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            handle.write(
                "respondent,q1,q2,q3,q4\n"
                "constant,1,1,1,1\n"
                "varied,1,2,3,4\n"
                "partial,2,,4,\n"
                "missing,,,,\n"
            )
        arguments = [
            "screen",
            str(source),
            "--header",
            "present",
            "--id-column",
            "respondent",
            "--indices",
            *indices,
            "--min-valid-indices",
            "3",
            "--strict",
        ]
        for name, threshold in thresholds.items():
            arguments.extend(["--threshold", f"{name}={threshold}"])

        payload = json.loads(_run_cli(executable, *arguments, "--format", "json"))
        if payload["respondent_ids"] != identifiers or payload["errors"]:
            raise RuntimeError("installed CLI lost respondent identifiers or failed scoring")
        if payload["consensus_flags"] != result["consensus_flags"].tolist():
            raise RuntimeError("installed CLI disagrees with API screening decisions")
        for name, expected in expected_scores.items():
            cli_scores = np.asarray(payload["scores"][name], dtype=float)
            np.testing.assert_allclose(cli_scores, expected, equal_nan=True)

        archive_path = root / "scores.npz"
        _run_cli(executable, *arguments, "--format", "npz", "--output", str(archive_path))
        archive = ier.load_score_archive(archive_path)
        if archive["respondent_ids"] != identifiers:
            raise RuntimeError("installed CLI archive lost respondent identifiers")
        for name, expected in expected_scores.items():
            np.testing.assert_allclose(archive["scores"][name], expected, equal_nan=True)
        restored = ier.screen_scores(
            archive["scores"], thresholds=thresholds, min_flags=2, min_valid_indices=3
        )
        np.testing.assert_array_equal(restored["consensus_flags"], result["consensus_flags"])
        reuse_arguments = [
            "screen-scores",
            str(archive_path),
            "--min-valid-indices",
            "3",
            "--strict",
            "--format",
            "json",
        ]
        for name, threshold in thresholds.items():
            reuse_arguments.extend(["--threshold", f"{name}={threshold}"])
        reused_payload = json.loads(_run_cli(executable, *reuse_arguments))
        if reused_payload != payload:
            raise RuntimeError("installed CLI archive replay disagrees with original screening")


def _check_response_times(executable: str) -> None:
    """Check installed timing archives, inclusive cutoffs, and percentile ties."""
    identifiers = ["fast", "boundary", "slow", "partial", "missing"]
    expected_scores = [1.0, 2.0, 6.0, 2.0, np.nan]
    initial_flags = [True, True, False, True, False]
    percentile_flags = [True, False, False, False, False]
    # The observed medians sort to [1, 2, 2, 6]. Their 50th percentile is 2,
    # which excludes tied scores, unlike an inclusive fixed threshold of 2.
    expected_payload = {
        "n_respondents": 5,
        "metric": "median",
        "flag_direction": "low",
        "threshold": 2.0,
        "scores": [1.0, 2.0, 6.0, 2.0, None],
        "flags": initial_flags,
        "respondent_ids": identifiers,
    }

    def check_archive(path: Path, expected_flags: list[bool]) -> None:
        saved = ier.load_response_time_archive(path)
        np.testing.assert_allclose(saved["scores"], expected_scores, equal_nan=True)
        np.testing.assert_array_equal(saved["flags"], expected_flags)
        if (
            saved["respondent_ids"] != identifiers
            or saved["metric"] != "median"
            or saved["flag_direction"] != "low"
            or saved["threshold"] != 2.0
        ):
            raise RuntimeError("installed timing archive lost aligned IDs or decision metadata")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "times.csv"
        source.write_text(
            "respondent,t1,t2,t3\n"
            "fast,1,1,1\n"
            "boundary,2,2,2\n"
            "slow,4,6,8\n"
            "partial,1,,3\n"
            "missing,,,\n",
            encoding="utf-8",
        )
        original = root / "timing.npz"
        _run_cli(
            executable,
            "response-time",
            str(source),
            "--header",
            "present",
            "--id-column",
            "respondent",
            "--threshold",
            "2",
            "--format",
            "npz",
            "--output",
            str(original),
        )
        check_archive(original, initial_flags)
        source.unlink()

        replay = json.loads(
            _run_cli(executable, "response-time-scores", str(original), "--format", "json")
        )
        if replay != expected_payload:
            raise RuntimeError("installed timing replay changed fixed-cutoff decisions")

        exported = root / "timing-export.npz"
        _run_cli(
            executable,
            "response-time-scores",
            str(original),
            "--format",
            "npz",
            "--output",
            str(exported),
        )
        check_archive(exported, initial_flags)
        _run_cli(
            executable,
            "response-time-scores",
            str(exported),
            "--percentile",
            "50",
            "--format",
            "npz",
            "--output",
            str(exported),
        )
        check_archive(exported, percentile_flags)
        expected_payload["flags"] = percentile_flags
        reflagged = json.loads(
            _run_cli(executable, "response-time-scores", str(exported), "--format", "json")
        )
        if reflagged != expected_payload:
            raise RuntimeError("installed timing replay changed saved percentile tie decisions")


def _check_planned_omissions(executable: str) -> None:
    """Check exact survey names and skip masks through installed scoring and replay."""
    identifiers = ["planned, branch", "omitted", "complete", "inapplicable"]
    expected_scores = [0.0, 1.0, 0.0, np.nan]
    expected_flags = [False, True, False, False]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "branching.csv"
        with source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "q1", "Question, skipped", "q2", "notes"])
            writer.writerows(
                [
                    [identifiers[0], 1, "", "", "unselected metadata"],
                    [identifiers[1], "", "", "", "unselected metadata"],
                    [identifiers[2], 3, 3, 3, "unselected metadata"],
                    [identifiers[3], "", "", "", "unselected metadata"],
                ]
            )
        # Masks match selected q2, q1, Question axes, rather than file column order.
        # The first row has one observed applicable response, the second has two
        # omissions, and the last row has no applicable responses at all.
        mask = np.array(
            [[False, True, False], [True, False, True], [True, True, False], [False] * 3]
        )
        binary_mask = root / "applicable.npy"
        np.save(binary_mask, mask)
        text_mask = root / "applicable.csv.gz"
        with gzip.open(text_mask, "wt", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(mask.astype(np.uint8))

        archived: list[tuple[str, Path, list[str], dict[str, object]]] = []
        for command, mask_path in (("screen", binary_mask), ("composite", text_mask)):
            decisions = (
                ["--threshold", "missing_rate=0.6", "--min-flags", "1"]
                if command == "screen"
                else ["--threshold", "0.6", "--no-standardize", "--include-components"]
            )
            arguments = [
                command,
                str(source),
                "--id-column",
                "id",
                "--item-columns",
                "q2,q1",
                "--item-column",
                "Question, skipped",
                "--missing-applicable-mask",
                str(mask_path),
                "--indices",
                "missing_rate",
                "--strict",
                *decisions,
            ]
            payload = json.loads(_run_cli(executable, *arguments, "--format", "json"))
            if payload["respondent_ids"] != identifiers or payload["errors"]:
                raise RuntimeError("installed skip-mask scoring lost IDs or failed scoring")
            score_values = (
                payload["scores"]["missing_rate"] if command == "screen" else payload["scores"]
            )
            np.testing.assert_allclose(
                np.asarray(score_values, dtype=float), expected_scores, equal_nan=True
            )
            actual_flags = payload["consensus_flags"] if command == "screen" else payload["flags"]
            if actual_flags != expected_flags:
                raise RuntimeError("installed skip-mask scoring flagged planned omissions")
            archive_path = root / f"{command}-branching.npz"
            _run_cli(executable, *arguments, "--format", "npz", "--output", str(archive_path))
            saved = ier.load_score_archive(archive_path)
            np.testing.assert_allclose(
                saved["scores"]["missing_rate"], expected_scores, equal_nan=True
            )
            if saved["respondent_ids"] != identifiers:
                raise RuntimeError("installed skip-mask archive lost respondent identifiers")
            archived.append((command, archive_path, decisions, payload))

        source.unlink()
        binary_mask.unlink()
        text_mask.unlink()
        for command, archive_path, decisions, payload in archived:
            replayed = json.loads(
                _run_cli(
                    executable,
                    f"{command}-scores",
                    str(archive_path),
                    *decisions,
                    "--format",
                    "json",
                )
            )
            if replayed != payload:
                raise RuntimeError("installed skip-mask archive replay changed scores or decisions")


def main() -> int:
    """Verify metadata and real workflows using only runtime dependencies."""
    distribution_version = version("insufficient-effort")
    if ier.__version__ != distribution_version:
        raise RuntimeError(
            f"ier.__version__={ier.__version__!r} does not match installed "
            f"distribution version {distribution_version!r}"
        )

    executable = shutil.which("ier")
    if executable is None:
        raise RuntimeError("installed distribution is missing the ier console script")

    expected = f"ier {distribution_version}"
    actual = _run_cli(executable, "--version").strip()
    if actual != expected:
        raise RuntimeError(f"ier --version returned {actual!r}; expected {expected!r}")

    _check_scoring(executable)
    _check_response_times(executable)
    _check_planned_omissions(executable)
    print(f"verified installed insufficient-effort {distribution_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
