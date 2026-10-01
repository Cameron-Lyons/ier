"""Run every performance benchmark at a small scale through its documented CLI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MATRIX = ["--respondents", "64", "--items", "24", "--repeats", "1", "--warmup", "0"]
_CASES = [
    (
        "archive",
        ["--respondents", "64", "--indices", "3", "--repeats", "1", "--write-repeats", "1"],
    ),
    ("cli_input", ["--respondents", "32", "--items", "6", "--repeats", "1"]),
    ("cli_output", ["--respondents", "32", "--indices", "3", "--repeats", "1"]),
    ("composite", [*_MATRIX, "--sensitivity-scenarios", "2"]),
    ("evenodd", ["--respondents", "64", "--factors", "2", "--repeats", "1", "--warmup", "0"]),
    (
        "evenodd",
        [
            "--respondents",
            "16",
            "--factors",
            "1",
            "--missing-rate",
            "1",
            "--order",
            "F",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
    ),
    (
        "evenodd",
        [
            "--respondents",
            "16",
            "--factors",
            "1",
            "--factor-items",
            "7",
            "--missing-rate",
            "0.5",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
    ),
    ("flagging", ["--respondents", "64", "--repeats", "1", "--warmup", "0"]),
    ("guttman", _MATRIX),
    ("guttman", [*_MATRIX, "--structure", "continuous", "--items", "769", "--order", "F"]),
    ("guttman", [*_MATRIX, "--missing-rate", "1"]),
    ("lz", _MATRIX),
    ("mahad", _MATRIX),
    ("mahad", [*_MATRIX, "--operation", "qq", "--items", "1"]),
    ("mahad", [*_MATRIX, "--operation", "qq", "--items", "2", "--missing-row-rate", "0.1"]),
    ("markov", _MATRIX),
    ("onset", _MATRIX),
    ("onset", [*_MATRIX, "--missing-rate", "1", "--order", "F"]),
    ("onset", [*_MATRIX, "--items", "4", "--window-size", "3", "--min-items", "3"]),
    ("onset", [*_MATRIX, "--items", "61", "--window-size", "32", "--min-items", "32"]),
    (
        "onset",
        [
            *_MATRIX,
            "--items",
            "61",
            "--window-size",
            "32",
            "--min-items",
            "32",
            "--structure",
            "continuous",
            "--missing-rate",
            "0.1",
        ],
    ),
    ("onset", [*_MATRIX, "--dtype", "int64", "--integer-offset", str(-(2**63))]),
    (
        "onset",
        [*_MATRIX, "--dtype", "uint64", "--integer-offset", str(2**64 - 6), "--order", "F"],
    ),
    ("orchestration", ["--respondents", "64", "--indices", "3", "--repeats", "1"]),
    ("pair_differences", _MATRIX),
    ("pair_differences", [*_MATRIX, "--dtype", "int64", "--missing-rate", "0"]),
    (
        "pair_differences",
        [
            *_MATRIX,
            "--dtype",
            "uint64",
            "--missing-rate",
            "0",
            "--integer-offset",
            str(2**64 - 6),
            "--order",
            "F",
        ],
    ),
    ("pair_differences", [*_MATRIX, "--missing-rate", "1", "--order", "F"]),
    ("pair_differences", [*_MATRIX, "--scale", "2.5e307", "--offset", "2.5e307"]),
    ("pair_differences", [*_MATRIX, "--scale", "1e-15", "--offset", "1.1", "--order", "F"]),
    ("person_total", _MATRIX),
    ("person_total", [*_MATRIX, "--order", "F"]),
    ("person_total", [*_MATRIX, "--missing-rate", "0.1"]),
    ("person_total", [*_MATRIX, "--missing-rate", "1", "--order", "F"]),
    ("person_total", [*_MATRIX, "--strict"]),
    ("person_total", [*_MATRIX, "--missing-rate", "0.1", "--strict"]),
    ("psychsyn", _MATRIX),
    (
        "psychsyn",
        [*_MATRIX, "--dtype", "int64", "--missing-rate", "0", "--integer-offset", str(2**60)],
    ),
    (
        "psychsyn",
        [
            *_MATRIX,
            "--operation",
            "correlations",
            "--dtype",
            "uint64",
            "--missing-rate",
            "0",
            "--integer-offset",
            str(2**64 - 256),
            "--order",
            "F",
        ],
    ),
    ("psychsyn", [*_MATRIX, "--items", "3"]),
    (
        "psychsyn",
        [*_MATRIX, "--missing-mode", "scattered", "--missing-rate", "1", "--critval", "0"],
    ),
    ("reliability", [*_MATRIX, "--splits", "2"]),
    ("reliability", [*_MATRIX, "--splits", "2", "--structure", "constant", "--items", "14"]),
    ("reliability", [*_MATRIX, "--splits", "2", "--structure", "near-constant", "--order", "F"]),
    (
        "reliability",
        [
            "--respondents",
            "1",
            "--items",
            "4",
            "--splits",
            "1",
            "--seed",
            "0",
            "--split-seed",
            "0",
            "--missing-rate",
            "0",
            "--order",
            "F",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
    ),
    ("response_checks", _MATRIX),
    ("response_time", _MATRIX),
    ("response_time", [*_MATRIX, "--scale", "1e300", "--no-log-transform"]),
    (
        "response_time",
        [*_MATRIX, "--structure", "constant", "--scale", "1e300", "--no-log-transform"],
    ),
    (
        "response_time",
        [*_MATRIX, "--structure", "near-constant", "--scale", "1e300", "--no-log-transform"],
    ),
    ("response_time", [*_MATRIX, "--items", "1", "--missing-rate", "0.5", "--order", "F"]),
    (
        "response_time",
        [*_MATRIX, "--operation", "median", "--respondents", "1", "--missing-rate", "1"],
    ),
    (
        "response_time",
        [*_MATRIX, "--operation", "median", "--items", "601", "--order", "F"],
    ),
    ("row_reductions", _MATRIX),
    (
        "row_reductions",
        [*_MATRIX, "--dtype", "int64", "--missing-rate", "0", "--integer-offset", str(2**60)],
    ),
    (
        "row_reductions",
        [
            *_MATRIX,
            "--dtype",
            "uint64",
            "--missing-rate",
            "0",
            "--integer-offset",
            str(2**64 - 6),
            "--order",
            "F",
        ],
    ),
    ("row_reductions", [*_MATRIX, "--structure", "constant", "--items", "14", "--irv-splits", "2"]),
    (
        "row_reductions",
        [*_MATRIX, "--structure", "near-constant", "--dtype", "float32", "--order", "F"],
    ),
    ("row_reductions", [*_MATRIX, "--irv-splits", "7", "--order", "F"]),
    ("row_reductions", [*_MATRIX, "--irv-splits", "100", "--missing-rate", "1"]),
    (
        "row_reductions",
        [*_MATRIX, "--dtype", "float32", "--strict", "--order", "F", "--irv-splits", "2"],
    ),
    ("screen", [*_MATRIX, "--sensitivity-scenarios", "2"]),
    ("sequence_scoring", _MATRIX),
    ("lz", [*_MATRIX, "--operation", "discrimination", "--categories", "5"]),
    ("psychsyn", [*_MATRIX, "--operation", "psychsyn_critval"]),
    ("psychsyn", [*_MATRIX, "--operation", "correlations", "--order", "F"]),
    ("psychsyn", [*_MATRIX, "--operation", "correlations", "--items", "1", "--missing-rate", "0"]),
    (
        "psychsyn",
        [
            *_MATRIX,
            "--operation",
            "correlations",
            "--missing-mode",
            "scattered",
            "--missing-rate",
            "1",
        ],
    ),
    (
        "cli_output",
        [
            "--respondents",
            "32",
            "--indices",
            "3",
            "--repeats",
            "1",
            "--workflow",
            "composite",
            "--flagged",
            "--probability",
            "--format",
            "json",
            "--compression",
            "gzip",
        ],
    ),
]


@pytest.mark.parametrize(("name", "arguments"), _CASES, ids=[name for name, _ in _CASES])
def test_performance_benchmark_cli(name: str, arguments: list[str]) -> None:
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment.pop("PYTHONTRACEMALLOC", None)
    result = subprocess.run(
        [sys.executable, "-W", "error", str(_ROOT / "benchmarks" / f"bench_{name}.py"), *arguments],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "median=" in result.stdout or "full=" in result.stdout
    assert "peak=" in result.stdout
