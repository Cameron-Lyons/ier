"""Keep actual benchmark mappings closed before temporary-file cleanup."""

from __future__ import annotations

import mmap
import weakref
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType


@pytest.fixture
def checked_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, list[tuple[bool, weakref.ReferenceType[mmap.mmap]]]]:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    benchmark = import_module("bench_cli_input")
    original_open = np.lib.format.open_memmap
    original_directory = benchmark.tempfile.TemporaryDirectory
    mappings: list[tuple[bool, weakref.ReferenceType[mmap.mmap]]] = []

    def track_mapping(*args: object, **kwargs: object) -> np.memmap:
        values = original_open(*args, **kwargs)
        assert isinstance(values, np.memmap)
        assert isinstance(values.base, mmap.mmap)
        mappings.append((values.flags.writeable, weakref.ref(values.base)))
        return values

    @contextmanager
    def checked_directory() -> Iterator[str]:
        with original_directory() as directory:
            try:
                yield directory
            finally:
                # Windows refuses to delete files while these real mmap handles
                # remain open. Check the same condition on every platform.
                live = [handle for _, reference in mappings if (handle := reference()) is not None]
                open_handles = [handle for handle in live if not handle.closed]
                try:
                    assert not open_handles, "benchmark left file mappings open before cleanup"
                finally:
                    # Also clean up the deliberately failing old implementation.
                    for handle in open_handles:
                        handle.close()

    monkeypatch.setattr(np.lib.format, "open_memmap", track_mapping)
    monkeypatch.setattr(benchmark.tempfile, "TemporaryDirectory", checked_directory)
    monkeypatch.setattr(
        "sys.argv", ["bench_cli_input.py", "--respondents", "33", "--items", "7", "--masks"]
    )
    return benchmark, mappings


@pytest.mark.parametrize("repeats", [1, 3])
def test_mask_benchmark_closes_owned_maps_before_cleanup(
    checked_benchmark: tuple[ModuleType, list[tuple[bool, weakref.ReferenceType[mmap.mmap]]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    repeats: int,
) -> None:
    benchmark, mappings = checked_benchmark
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_cli_input.py",
            "--respondents",
            "33",
            "--items",
            "7",
            "--masks",
            "--repeats",
            str(repeats),
        ],
    )
    benchmark.main()
    output = capsys.readouterr().out
    assert "mask.csv: median=" in output and "mask.npy: median=" in output
    assert sum(not writable for writable, _ in mappings) == repeats + 1
    assert sum(writable for writable, _ in mappings) == 1


def test_mask_benchmark_preserves_value_validation_error_and_closes_maps(
    checked_benchmark: tuple[ModuleType, list[tuple[bool, weakref.ReferenceType[mmap.mmap]]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, mappings = checked_benchmark
    original_fixture = benchmark._write_mask_fixture

    def corrupt_fixture(path: Path, n_respondents: int, n_items: int) -> None:
        original_fixture(path, n_respondents, n_items)
        if path.suffix == ".npy":
            values = np.load(path, mmap_mode="r+", allow_pickle=False)
            assert isinstance(values.base, mmap.mmap)
            try:
                values[0, 0] = not values[0, 0]
                values.flush()
            finally:
                values.base.close()

    monkeypatch.setattr(benchmark, "_write_mask_fixture", corrupt_fixture)
    with pytest.raises(AssertionError, match="Arrays are not equal"):
        benchmark.main()
    assert any(not writable for writable, _ in mappings)


def test_mask_fixture_flush_error_preserves_error_and_closes_writer(
    checked_benchmark: tuple[ModuleType, list[tuple[bool, weakref.ReferenceType[mmap.mmap]]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark, mappings = checked_benchmark

    def failed_flush(values: np.memmap) -> None:
        raise OSError("cannot flush mapped fixture")

    monkeypatch.setattr(np.memmap, "flush", failed_flush)
    with pytest.raises(OSError, match="cannot flush mapped fixture"):
        benchmark.main()
    assert len(mappings) == 1 and mappings[0][0]
