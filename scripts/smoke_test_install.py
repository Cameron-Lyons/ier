"""Exercise an installed distribution's API, CLI, and reusable-score archives."""

from __future__ import annotations

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
    print(f"verified installed insufficient-effort {distribution_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
