"""Extreme-unit benchmarks validate exact scores through their public CLI."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return import_module("bench_person_total")


@pytest.mark.parametrize(
    "options",
    [
        ["--structure", "near-constant"],
        ["--structure", "near-constant", "--order", "F", "--missing-rate", "0.2"],
        ["--structure", "near-constant", "--scale", "1e300"],
        ["--structure", "categorical", "--scale", "5e-324"],
        ["--structure", "categorical", "--scale", "5e-324", "--offset", "2.2250738585072014e-308"],
        ["--structure", "constant"],
        ["--structure", "constant", "--missing-rate", "1"],
        ["--structure", "near-constant", "--strict", "--missing-rate", "0.2"],
    ],
)
def test_extreme_benchmark_cli_checks_its_measured_scores(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_person_total.py",
            "--respondents",
            "33",
            "--items",
            "7",
            "--repeats",
            "1",
            "--warmup",
            "0",
            *options,
        ],
    )
    benchmark.main()
    output = capsys.readouterr().out
    assert "person_total: median=" in output
    assert "peak=" in output
    assert "n_valid=" in output


def test_benchmark_exact_oracle_rejects_the_original_subnormal_scores(
    benchmark: ModuleType,
) -> None:
    data = np.array([[1.0, 2, 3], [2, 1, 4], [1, 2, 4]]) * np.nextafter(0.0, 1.0)
    incorrect = np.array([np.sqrt(27 / 28), 11 / 14, 1.0])
    with pytest.raises(AssertionError):
        benchmark._check_scores(data, incorrect, na_rm=True, exact=True)


@pytest.mark.parametrize(
    "options",
    [["--scale", "-1"], ["--scale", "nan"], ["--offset", "inf"], ["--scale", "1e308"]],
)
def test_benchmark_rejects_invalid_response_units(
    benchmark: ModuleType, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    monkeypatch.setattr(
        "sys.argv", ["bench_person_total.py", "--respondents", "33", "--items", "7", *options]
    )
    with pytest.raises(SystemExit) as raised:
        benchmark.main()
    assert raised.value.code == 2
