"""Shared text-file streams for command-line input and output."""

from __future__ import annotations

import bz2
import gzip
import lzma
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import TextIO

_COMPRESSION_SUFFIXES = frozenset({".bz2", ".gz", ".xz"})


def _is_compressed_npy_path(path: Path) -> bool:
    """Return whether a path names a compressed NumPy binary matrix."""
    return (
        path.suffix.casefold() in _COMPRESSION_SUFFIXES
        and path.with_suffix("").suffix.casefold() == ".npy"
    )


@contextmanager
def _open_text_path(path: Path, mode: Literal["r", "w"]) -> Iterator[TextIO]:
    """Open a UTF-8 text path with compression selected from its suffix."""
    text_mode: Literal["rt", "wt"] = "rt" if mode == "r" else "wt"
    suffix = path.suffix.casefold()
    if suffix == ".bz2":
        with bz2.open(path, mode=text_mode, newline="", encoding="utf-8") as handle:
            yield handle
        return
    if suffix == ".gz":
        with gzip.open(path, mode=text_mode, newline="", encoding="utf-8") as handle:
            yield handle
        return
    if suffix == ".xz":
        if mode == "w":
            with lzma.open(
                path,
                mode="wt",
                newline="",
                encoding="utf-8",
                preset=1,
            ) as handle:
                yield handle
            return
        with lzma.open(path, mode="rt", newline="", encoding="utf-8") as handle:
            yield handle
        return
    with path.open(mode=mode, newline="", encoding="utf-8") as handle:
        yield handle
