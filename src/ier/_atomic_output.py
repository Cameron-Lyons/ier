"""Shared atomic replacement for streamed result files."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from stat import S_IMODE, S_ISREG
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def atomic_output_path(path: Path) -> Iterator[Path]:
    """Stage regular files beside their destination until writing completes.

    Callers must close their streams before exiting the context. Existing
    permission bits and symbolic links are preserved. Nonregular destinations
    retain direct streaming, since devices and pipes cannot be replaced.
    """
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        mode = None

    if mode is not None and not S_ISREG(mode):
        yield path
        return

    destination = path.resolve(strict=False) if path.is_symlink() else path
    with TemporaryDirectory(prefix=".ier-output-", dir=destination.parent) as directory:
        # Keep the requested suffix so aliases still select the same compressor.
        staged_path = Path(directory) / path.name
        yield staged_path
        if mode is not None:
            staged_path.chmod(S_IMODE(mode))
        staged_path.replace(destination)
