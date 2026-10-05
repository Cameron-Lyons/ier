"""Bound NPZ member allocation by the data actually stored in the ZIP container."""

from __future__ import annotations

from io import BytesIO
from math import prod
from typing import TYPE_CHECKING, cast
from zipfile import ZIP_STORED

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import IO, Literal, Protocol

    from numpy.lib.npyio import NpzFile

    class _HeaderReader(Protocol):
        def __call__(
            self, stream: IO[bytes], *, max_header_size: int
        ) -> tuple[tuple[int, ...], bool, np.dtype]: ...


_MAX_HEADER_SIZE = 10_000


def read_npz_member(archive: NpzFile, name: str) -> np.ndarray:
    """Check an NPY header and its byte count before NumPy allocates the array."""
    try:
        info = archive.zip.getinfo(name)
    except KeyError:
        info = archive.zip.getinfo(f"{name}.npy")

    with archive.zip.open(info) as member:
        # NumPy's public format functions do not expose complete annotations.
        read_magic = cast("Callable[[IO[bytes]], tuple[int, int]]", np.lib.format.read_magic)
        version = read_magic(member)
        if version not in {(1, 0), (2, 0), (3, 0)}:
            raise ValueError(f"unsupported NPY format version: {version}")

        length_size = 2 if version == (1, 0) else 4
        length_bytes = member.read(length_size)
        if len(length_bytes) != length_size:
            raise ValueError("truncated NPY header length")
        header_size = int.from_bytes(length_bytes, "little")
        if header_size > _MAX_HEADER_SIZE:
            raise ValueError(f"NPY header exceeds {_MAX_HEADER_SIZE} bytes")
        header_bytes = member.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError("truncated NPY header")

        # NumPy exposes readers for v1/v2 only. V3 uses the v2 layout with
        # UTF-8 rather than Latin-1; escaping Unicode inside its Python string
        # literals preserves field names for the public v2 header parser.
        if version == (3, 0):
            header_bytes = header_bytes.decode("utf-8").encode("ascii", "backslashreplace")
            length_bytes = len(header_bytes).to_bytes(length_size, "little")
        header_stream = BytesIO(length_bytes + header_bytes)
        reader = cast(
            "_HeaderReader",
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0,
        )
        shape, fortran_order, dtype = reader(header_stream, max_header_size=_MAX_HEADER_SIZE)
        if any(isinstance(dimension, bool) for dimension in shape):
            raise ValueError("NPY shape dimensions must be integers, not booleans")
        if any(dimension < 0 for dimension in shape):
            raise ValueError("NPY shape dimensions must be nonnegative")
        element_count = prod(shape)
        index_limit = np.iinfo(np.intp).max
        if any(dimension > index_limit for dimension in shape) or element_count > index_limit:
            raise ValueError("NPY shape exceeds the platform index range")
        if dtype.hasobject:
            raise ValueError("Object arrays cannot be loaded when allow_pickle=False")
        if dtype.itemsize == 0 and element_count:
            raise ValueError("nonempty NPY arrays must have a positive item size")

        # Python integer multiplication avoids the fixed-width overflow used
        # by NumPy's element count. Check even metadata before decoding: a tiny
        # ZIP member may claim an enormous shape or string width.
        expected_bytes = element_count * dtype.itemsize
        stored_bytes = info.file_size - member.tell()
        if expected_bytes != stored_bytes:
            raise ValueError(
                f"NPY header declares {expected_bytes} payload bytes; "
                f"ZIP member contains {stored_bytes}"
            )
        if info.compress_type == ZIP_STORED:
            # ZipFile.open also checks compressed bytes against the physical
            # boundaries of this member. Stored lengths must therefore agree.
            if info.file_size != info.compress_size:
                raise ValueError("stored ZIP member sizes are inconsistent")
            member.seek(0)
            read_array = cast("Callable[[IO[bytes], bool], np.ndarray]", np.lib.format.read_array)
            return read_array(member, False)

        # The central directory's uncompressed size alone is not evidence that
        # the compressed stream contains the claimed data. Decode once in bounded
        # chunks, so allocation grows only with bytes actually present, and keep
        # the verified payload instead of decompressing it again. Reading to the
        # end of the stream also lets ZipFile verify the member's CRC.
        payload = bytearray()
        while chunk := member.read(min(1 << 20, expected_bytes - len(payload) + 1)):
            payload += chunk
            if len(payload) > expected_bytes:
                raise ValueError(
                    f"NPY header declares {expected_bytes} payload bytes; "
                    "decompressed member contains more"
                )
        if len(payload) != expected_bytes:
            raise ValueError(
                f"NPY header declares {expected_bytes} payload bytes; "
                f"decompressed member contains {len(payload)}"
            )
        order: Literal["C", "F"] = "F" if fortran_order else "C"
        if element_count == 0:
            # Zero-width dtypes such as <U0 cannot be viewed from a buffer.
            return np.empty(shape, dtype=dtype, order=order)
        # The bytearray keeps the decoded array writable, matching read_array.
        array = np.frombuffer(payload, dtype=dtype, count=element_count)
        return array.reshape(shape, order=order)
