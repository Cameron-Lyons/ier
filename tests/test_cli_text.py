"""Ranked text previews preserve decisions, finite scores, and input-order ties."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier._cli_output import _ranked_rows
from ier.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _reference_rows(scores: np.ndarray, top: int, direction: str) -> list[int]:
    def key(row: int) -> tuple[float | int, int]:
        value = scores[row].item()
        return (-value if direction == "high" else value), row

    return sorted((row for row, value in enumerate(scores) if np.isfinite(value)), key=key)[
        : max(top, 0)
    ]


@pytest.mark.parametrize("direction", ["high", "low"])
@pytest.mark.parametrize("top", [-5, 0, 1, 2, 10, 32, 97, 257, 1000])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
@pytest.mark.parametrize(
    "kind", ["continuous", "ties", "constant", "missing", "extreme", "int64", "uint64"]
)
def test_ranked_rows_match_scalar_order(kind: str, layout: str, top: int, direction: str) -> None:
    rng = np.random.default_rng(568)
    if kind == "continuous":
        scores = rng.normal(size=257)
        scores[::5] = np.nan
        scores[1::17] = np.inf
        scores[2::17] = -np.inf
    elif kind == "ties":
        scores = rng.integers(-3, 4, size=257).astype(float)
        scores[::11] = np.nan
    elif kind == "constant":
        scores = np.full(257, 1.1)
    elif kind == "missing":
        scores = np.resize([np.nan, np.inf, -np.inf], 257)
    elif kind == "extreme":
        scores = np.resize(
            [np.finfo(float).max, -np.finfo(float).max, 1e-300, -1e-300, 0.0, -0.0], 257
        )
    else:
        dtype = np.int64 if kind == "int64" else np.uint64
        scores = rng.bit_generator.random_raw(257).view(dtype)
        scores[::11] = np.iinfo(dtype).max
        scores[1::11] = np.iinfo(dtype).min
    if layout == "strided":
        scores = scores[::-2]
    original = scores.copy()
    scores.flags.writeable = False
    expected = _reference_rows(scores, top, direction)
    with patch("ier._cli_output._TEXT_RANK_BATCH_SIZE", 17):
        actual = _ranked_rows(scores, top, direction)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(scores, original)


@pytest.mark.parametrize("direction", ["high", "low"])
def test_ranked_rows_support_empty_input(direction: str) -> None:
    assert _ranked_rows(np.empty(0), 10, direction).shape == (0,)


@pytest.mark.parametrize("direction", ["high", "low"])
def test_ranked_rows_keep_fewer_available_scores_than_requested(direction: str) -> None:
    scores = np.full(1025, np.nan)
    scores[[0, 1, 333, 555, 1024]] = [-np.inf, np.inf, 3, 1, 3]
    with patch("ier._cli_output._TEXT_RANK_BATCH_SIZE", 17):
        actual = _ranked_rows(scores, 10, direction)
    np.testing.assert_array_equal(actual, _reference_rows(scores, 10, direction))


@pytest.mark.parametrize("top", [0, -3])
def test_summary_only_preview_does_not_scan_scores(top: int) -> None:
    with patch("ier._cli_output.np.isfinite", side_effect=AssertionError("unexpected scan")):
        assert _ranked_rows(np.ones(100), top).size == 0


@pytest.mark.parametrize(
    "command",
    [
        ["screen", "--indices", "longstring"],
        ["composite", "--indices", "longstring", "--include-components", "--include-probability"],
        ["response-time", "--metric", "mean"],
        ["response-time", "--metric", "mixture", "--random-seed", "42"],
    ],
)
def test_scoring_commands_rank_ties_in_input_order(tmp_path: Path, command: list[str]) -> None:
    path = tmp_path / "responses.csv"
    path.write_text("id,q1,q2,q3\nzeta,1,1,1\nalpha,1,1,1\nmiddle,1,1,1\n", encoding="utf-8")
    output = StringIO()
    with patch("sys.stdout", output):
        status = main([command[0], str(path), *command[1:], "--id-column", "id", "--top", "2"])
    assert status == 0
    rows = [line.strip().split("\t")[0] for line in output.getvalue().splitlines() if "\t" in line]
    assert rows == ["zeta", "alpha"]
