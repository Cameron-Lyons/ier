"""Handcrafted invalid NPZ containers exercise the actual parser and ZIP layer."""

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest

from ier import load_response_time_archive, load_score_archive

ArchiveLoader = Callable[[Path], Any]


def _payload(result_type: str) -> dict[str, np.ndarray]:
    payload = {
        "schema_version": np.asarray(1),
        "result_type": np.asarray(result_type),
        "n_respondents": np.asarray(2),
    }
    if result_type == "screen":
        payload.update({"index_names": np.asarray(["irv"]), "score__irv": np.asarray([1.0, 2.0])})
    else:
        payload.update(
            {
                "metric": np.asarray("median"),
                "flag_direction": np.asarray("low"),
                "threshold": np.asarray(1.5),
                "scores": np.asarray([1.0, 2.0]),
                "flags": np.asarray([True, False]),
            }
        )
    return payload


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


@pytest.mark.parametrize(
    ("loader", "result_type", "member"),
    [
        (load_score_archive, "screen", "score__irv"),
        (load_response_time_archive, "response_time", "scores"),
    ],
)
@pytest.mark.parametrize("dtype", [np.str_, np.bool_, np.complex128])
def test_archive_loaders_reject_non_real_score_member_types(
    tmp_path: Path,
    loader: ArchiveLoader,
    result_type: str,
    member: str,
    dtype: type,
) -> None:
    payload = _payload(result_type)
    payload[member] = np.asarray([1, 2], dtype=dtype)
    destination = tmp_path / "typed.npz"
    np.savez(destination, **payload)
    with pytest.raises(ValueError, match="real numeric array"):
        loader(destination)


@pytest.mark.parametrize(
    ("loader", "result_type", "member"),
    [
        (load_score_archive, "screen", "score__irv"),
        (load_response_time_archive, "response_time", "scores"),
    ],
)
def test_truncated_npy_member_has_contextual_corruption_error(
    tmp_path: Path,
    loader: ArchiveLoader,
    result_type: str,
    member: str,
) -> None:
    destination = tmp_path / "truncated-member.npz"
    with ZipFile(destination, "w") as archive:
        for name, value in _payload(result_type).items():
            encoded = _npy_bytes(value)
            if name == member:
                encoded = encoded[:-1]
            archive.writestr(f"{name}.npy", encoded)
    with pytest.raises(ValueError, match=f"member {member} is malformed"):
        loader(destination)


@pytest.mark.parametrize(
    ("loader", "result_type", "member"),
    [
        (load_score_archive, "screen", "score__irv"),
        (load_response_time_archive, "response_time", "scores"),
    ],
)
def test_bad_member_checksum_has_contextual_corruption_error(
    tmp_path: Path,
    loader: ArchiveLoader,
    result_type: str,
    member: str,
) -> None:
    destination = tmp_path / "bad-checksum.npz"
    payload = _payload(result_type)
    with ZipFile(destination, "w", compression=ZIP_STORED) as archive:
        for name, value in payload.items():
            archive.writestr(f"{name}.npy", _npy_bytes(value))
    encoded = _npy_bytes(payload[member])
    damaged = bytearray(destination.read_bytes())
    damaged[damaged.index(encoded) + len(encoded) - 1] ^= 1
    destination.write_bytes(damaged)
    with pytest.raises(ValueError, match=f"member {member} cannot be read:.*CRC"):
        loader(destination)


@pytest.mark.parametrize("loader", [load_score_archive, load_response_time_archive])
@pytest.mark.parametrize("content", [b"", b"not NumPy data", b"PK\x03\x04broken ZIP"])
def test_invalid_containers_have_contextual_load_errors(
    tmp_path: Path,
    loader: ArchiveLoader,
    content: bytes,
) -> None:
    destination = tmp_path / "invalid.npz"
    destination.write_bytes(content)
    with pytest.raises(ValueError, match="archive could not be read as an NPZ archive"):
        loader(destination)


@pytest.mark.parametrize("loader", [load_score_archive, load_response_time_archive])
def test_missing_archive_preserves_filesystem_error(tmp_path: Path, loader: ArchiveLoader) -> None:
    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "missing.npz")


@pytest.mark.parametrize("loader", [load_score_archive, load_response_time_archive])
def test_wrong_format_npy_is_rejected_before_decoding_its_payload(
    tmp_path: Path,
    loader: ArchiveLoader,
) -> None:
    destination = tmp_path / "large-matrix.npz"
    with destination.open("wb") as stream:
        # Declares an irrelevant 8 GB matrix without actually storing its data.
        np.lib.format.write_array_header_1_0(
            stream,
            {"descr": "<f8", "fortran_order": False, "shape": (1_000_000_000,)},
        )
    with pytest.raises(ValueError, match="archive must be an NPZ archive"):
        loader(destination)
