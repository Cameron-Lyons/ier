"""Shared text-file streams for command-line input and output."""

from __future__ import annotations

import bz2
import gzip
import lzma
import zlib
from contextlib import ExitStack, contextmanager
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
    """Open a UTF-8 text path and report compressed transport failures by path."""
    text_mode: Literal["rt", "wt"] = "rt" if mode == "r" else "wt"
    suffix = path.suffix.casefold()
    try:
        with ExitStack() as stack:
            handle: TextIO
            if suffix == ".bz2":
                handle = stack.enter_context(
                    bz2.open(path, mode=text_mode, newline="", encoding="utf-8")
                )
            elif suffix == ".gz":
                handle = stack.enter_context(
                    gzip.open(path, mode=text_mode, newline="", encoding="utf-8")
                )
            elif suffix == ".xz" and mode == "w":
                handle = stack.enter_context(
                    lzma.open(path, mode="wt", newline="", encoding="utf-8", preset=1)
                )
            elif suffix == ".xz":
                handle = stack.enter_context(
                    lzma.open(path, mode="rt", newline="", encoding="utf-8")
                )
            else:
                handle = stack.enter_context(path.open(mode=mode, newline="", encoding="utf-8"))
            yield handle
    except (EOFError, OSError, lzma.LZMAError, zlib.error) as err:
        if mode != "r" or suffix not in _COMPRESSION_SUFFIXES:
            raise
        raise ValueError(f"failed to read compressed text from {path}: {err}") from err
