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
    ("flagging", ["--respondents", "64", "--repeats", "1", "--warmup", "0"]),
    ("guttman", _MATRIX),
    ("lz", _MATRIX),
    ("mahad", _MATRIX),
    ("markov", _MATRIX),
    ("onset", _MATRIX),
    ("orchestration", ["--respondents", "64", "--indices", "3", "--repeats", "1"]),
    ("pair_differences", _MATRIX),
    ("person_total", _MATRIX),
    ("psychsyn", _MATRIX),
    ("reliability", [*_MATRIX, "--splits", "2"]),
    ("response_checks", _MATRIX),
    ("response_time", _MATRIX),
    ("row_reductions", _MATRIX),
    ("screen", [*_MATRIX, "--sensitivity-scenarios", "2"]),
    ("sequence_scoring", _MATRIX),
    ("lz", [*_MATRIX, "--operation", "discrimination", "--categories", "5"]),
    ("psychsyn", [*_MATRIX, "--operation", "psychsyn_critval"]),
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
