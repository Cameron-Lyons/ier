"""Result files remain complete across streamed writes and write failures."""

from __future__ import annotations

import os
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from stat import S_IMODE, S_ISFIFO
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from ier._cli_output import (
    _output_stream,
    _write_json_value,
    _write_output,
    _write_stream_output,
)
from ier._cli_streams import _open_text_path
from ier.cli import main

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Literal, TextIO


@pytest.mark.parametrize("suffix", [".txt", ".csv.gz", ".json.bz2", ".txt.xz", ".csv.GZ"])
@pytest.mark.parametrize("existing", [False, True])
def test_destination_changes_only_after_stream_closes(
    tmp_path: Path, suffix: str, existing: bool
) -> None:
    destination = tmp_path / f"result{suffix}"
    if existing:
        destination.write_bytes(b"previous complete result")
    with _output_stream(destination) as stream:
        stream.write("respondent,score\n")
        stream.flush()
        assert destination.exists() == existing
        if existing:
            assert destination.read_bytes() == b"previous complete result"
        stream.write("café,1.5\n")
    with _open_text_path(destination, "r") as stream:
        assert stream.read() == "respondent,score\ncafé,1.5\n"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("suffix", [".txt", ".csv.gz", ".json.bz2", ".txt.xz"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_failed_stream_preserves_destination_and_cleans_staging(
    tmp_path: Path, suffix: str, existing: bool, error: type[BaseException]
) -> None:
    destination = tmp_path / f"result{suffix}"
    if existing:
        destination.write_bytes(b"previous complete result")
    with pytest.raises(error, match="interrupted write"), _output_stream(destination) as stream:
        stream.write("partial output")
        stream.flush()
        raise error("interrupted write")
    assert destination.exists() == existing
    if existing:
        assert destination.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == ([destination] if existing else [])


def test_late_json_validation_failure_preserves_existing_result(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    destination.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float"):
        _write_stream_output(
            destination,
            lambda stream: _write_json_value(stream, {"valid": [1, 2, 3], "invalid": float("inf")}),
        )
    assert destination.read_text(encoding="utf-8") == '{"previous": true}'
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("suffix", [".txt", ".txt.gz", ".txt.bz2", ".txt.xz"])
def test_encoding_failure_preserves_existing_result(tmp_path: Path, suffix: str) -> None:
    destination = tmp_path / f"result{suffix}"
    destination.write_bytes(b"previous complete result")
    with pytest.raises(UnicodeEncodeError), _output_stream(destination) as stream:
        stream.write("valid first chunk\n")
        stream.flush()
        stream.write("\ud800")
    assert destination.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("operation", ["chmod", "replace"])
def test_commit_failure_preserves_existing_result(tmp_path: Path, operation: str) -> None:
    destination = tmp_path / "result.txt"
    destination.write_bytes(b"previous complete result")
    with (
        patch.object(Path, operation, side_effect=OSError("commit failed")),
        pytest.raises(OSError, match="commit failed"),
    ):
        _write_output("complete new result", destination)
    assert destination.read_bytes() == b"previous complete result"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are required")
def test_existing_permission_bits_are_preserved(tmp_path: Path) -> None:
    destination = tmp_path / "result.csv"
    destination.write_bytes(b"previous complete result")
    destination.chmod(0o640)
    _write_output("complete new result", destination)
    assert S_IMODE(destination.stat().st_mode) == 0o640


def test_new_file_uses_normal_creation_permissions(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary.txt"
    ordinary.write_text("normal creation", encoding="utf-8")
    destination = tmp_path / "result.txt"
    _write_output("new result", destination)
    assert S_IMODE(destination.stat().st_mode) == S_IMODE(ordinary.stat().st_mode)


@pytest.mark.skipif(os.name != "posix", reason="symbolic links are required")
@pytest.mark.parametrize("existing", [False, True])
def test_symlink_target_is_replaced_using_link_compression(tmp_path: Path, existing: bool) -> None:
    target = tmp_path / "target"
    if existing:
        target.write_bytes(b"previous complete result")
    link = tmp_path / "result.csv.gz"
    link.symlink_to(target.name)
    _write_output("complete new result", link)
    assert link.is_symlink()
    with _open_text_path(link, "r") as stream:
        assert stream.read() == "complete new result"
    assert set(tmp_path.iterdir()) == {target, link}


@pytest.mark.skipif(os.name != "posix", reason="a POSIX null device is required")
def test_special_file_is_streamed_without_replacement(tmp_path: Path) -> None:
    link = tmp_path / "null-output.txt"
    link.symlink_to(os.devnull)
    before = Path(os.devnull).stat()
    _write_output("discard this output", link)
    assert link.is_symlink()
    after = Path(os.devnull).stat()
    # Device timestamps can advance on writes, including on macOS.
    assert os.path.samestat(before, after)
    assert after.st_mode == before.st_mode
    assert list(tmp_path.iterdir()) == [link]


@pytest.mark.skipif(os.name != "posix", reason="POSIX named pipes are required")
def test_named_pipe_is_streamed_without_replacement(tmp_path: Path) -> None:
    destination = tmp_path / "result.csv"
    os.mkfifo(destination)
    reader = os.open(destination, os.O_RDONLY | os.O_NONBLOCK)
    try:
        _write_output("respondent,score\n0,1.5", destination)
        assert os.read(reader, 1024) == b"respondent,score\n0,1.5"
    finally:
        os.close(reader)
    assert S_ISFIFO(destination.stat().st_mode)
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="procfs descriptor links required")
def test_descriptor_symlinks_retain_direct_streaming() -> None:
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    try:
        _write_output("descriptor output", Path(f"/proc/self/fd/{writer}"))
        assert os.read(reader, 1024) == b"descriptor output"
    finally:
        os.close(reader)
        os.close(writer)


@pytest.mark.parametrize("destination", [None, Path("-")])
def test_standard_output_stays_open_and_streams_immediately(destination: Path | None) -> None:
    output = StringIO()
    with patch("sys.stdout", output), _output_stream(destination) as stream:
        stream.write("first chunk")
        assert output.getvalue() == "first chunk"
    assert not output.closed


@pytest.mark.parametrize("command", ["indices", "screen", "composite", "response-time"])
@pytest.mark.parametrize("output_format", ["text", "csv", "json"])
def test_cli_reports_finalization_failure_without_losing_previous_result(
    tmp_path: Path, command: str, output_format: str
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("a,b,c\n1,2,3\n3,3,3\n2,4,1\n", encoding="utf-8")
    destination = tmp_path / f"result.{output_format}.gz"
    destination.write_bytes(b"previous complete result")
    arguments = [command]
    if command != "indices":
        arguments.append(str(source))
    if command in {"screen", "composite"}:
        arguments.extend(["--indices", "irv", "longstring"])
    arguments.extend(["--format", output_format, "--output", str(destination)])

    @contextmanager
    def fail_finalization(path: Path, mode: Literal["r", "w"]) -> Iterator[TextIO]:
        with _open_text_path(path, mode) as stream:
            yield stream
        raise OSError("finalization failed")

    error_output = StringIO()
    with (
        patch("ier._cli_output._open_text_path", fail_finalization),
        patch("sys.stderr", error_output),
    ):
        assert main(arguments) == 1
    assert "error: finalization failed" in error_output.getvalue()
    assert "Traceback" not in error_output.getvalue()
    assert destination.read_bytes() == b"previous complete result"
    assert set(tmp_path.iterdir()) == {source, destination}
