"""Compressed NPZ output preserves every scoring and replay workflow."""

from __future__ import annotations

import json
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest

from ier import load_response_time_archive, load_score_archive
from ier.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _arguments(command: str) -> list[str]:
    if command.startswith("screen"):
        return [
            "--indices",
            "irv",
            "longstring",
            "--threshold",
            "irv=0",
            "--threshold",
            "longstring=3",
            "--min-flags",
            "1",
        ]
    if command.startswith("composite"):
        return [
            "--indices",
            "irv",
            "longstring",
            "--no-standardize",
            "--include-components",
            "--include-probability",
            "--threshold",
            "2",
        ]
    return ["--threshold", "1"]


@pytest.mark.parametrize(
    "command",
    [
        "screen",
        "composite",
        "response-time",
        "screen-scores",
        "composite-scores",
        "response-time-scores",
    ],
)
def test_compression_preserves_members_decisions_and_in_place_replay(
    tmp_path: Path, command: str
) -> None:
    source = tmp_path / "responses.csv"
    source.write_text(
        'id,q1,q2,q3,q4\n"α,first",1,1,1,1\nsecond,1,2,3,4\nthird,1,1,1,4\nmissing,,,,\n',
        encoding="utf-8",
    )
    options = _arguments(command)
    if command.endswith("-scores"):
        saved = tmp_path / "original.npz"
        fresh = "response-time" if command == "response-time-scores" else "screen"
        assert (
            main(
                [
                    fresh,
                    str(source),
                    "--id-column",
                    "id",
                    *_arguments(fresh),
                    "--format",
                    "npz",
                    "--output",
                    str(saved),
                ]
            )
            == 0
        )
        source.unlink()
        source = saved
    else:
        options += ["--id-column", "id"]

    stored = tmp_path / "stored.npz"
    compressed = tmp_path / "compressed.npz"
    for destination, compress_args in ((stored, []), (compressed, ["--compress"])):
        assert (
            main(
                [
                    command,
                    str(source),
                    *options,
                    "--format",
                    "npz",
                    "--output",
                    str(destination),
                    *compress_args,
                ]
            )
            == 0
        )
        with ZipFile(destination) as archive:
            expected = ZIP_DEFLATED if compress_args else ZIP_STORED
            assert all(member.compress_type == expected for member in archive.infolist())

    with (
        np.load(stored, allow_pickle=False) as plain,
        np.load(compressed, allow_pickle=False) as packed,
    ):
        assert plain.files == packed.files
        for name in plain.files:
            np.testing.assert_array_equal(plain[name], packed[name], err_msg=name)

    loader = (
        load_response_time_archive if command.startswith("response-time") else load_score_archive
    )
    assert loader(compressed)["respondent_ids"] == ["α,first", "second", "third", "missing"]
    replay = (
        "response-time-scores"
        if command.startswith("response-time")
        else "composite-scores"
        if command.startswith("composite")
        else "screen-scores"
    )
    # Replacing the compressed source in place must close the reader first,
    # preserve decisions, and remain reusable without the original CSV.
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        assert main([replay, str(compressed), "--format", "json", *_arguments(replay)]) == 0
    before = json.loads(stdout.getvalue())
    assert (
        main(
            [
                replay,
                str(compressed),
                "--format",
                "npz",
                "--output",
                str(compressed),
                "--compress",
                *_arguments(replay),
            ]
        )
        == 0
    )
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        assert main([replay, str(compressed), "--format", "json", *_arguments(replay)]) == 0
    assert json.loads(stdout.getvalue()) == before
    flag_field = "consensus_flags" if replay == "screen-scores" else "flags"
    expected_flags = (
        [True, False, False, False] if replay == "composite-scores" else [True, False, True, False]
    )
    assert before[flag_field] == expected_flags


@pytest.mark.parametrize("output_format", ["text", "json", "csv"])
@pytest.mark.parametrize(
    "command",
    [
        "screen",
        "composite",
        "response-time",
        "screen-scores",
        "composite-scores",
        "response-time-scores",
    ],
)
def test_compression_rejects_text_formats_before_reading_or_replacing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: str, output_format: str
) -> None:
    destination = tmp_path / "previous.txt"
    destination.write_text("previous result", encoding="utf-8")
    assert (
        main(
            [
                command,
                str(tmp_path / "absent-input"),
                "--format",
                output_format,
                "--compress",
                "--output",
                str(destination),
            ]
        )
        == 1
    )
    assert "--compress requires --format npz" in capsys.readouterr().err
    assert destination.read_text(encoding="utf-8") == "previous result"
