"""Handcrafted invalid NPZ containers exercise the actual parser and ZIP layer."""

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

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


@pytest.mark.parametrize(
    ("loader", "result_type", "member"),
    [
        (load_score_archive, "screen", "score__irv"),
        (load_response_time_archive, "response_time", "scores"),
    ],
)
@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
@pytest.mark.parametrize("shape", [(1_000_000_000,), (2**32, 2**32), (-1,)])
def test_invalid_shape_is_rejected_before_numpy_allocates_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: ArchiveLoader,
    result_type: str,
    member: str,
    compression: int,
    shape: tuple[int, ...],
) -> None:
    destination = tmp_path / "inflated-shape.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": shape}
    )
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in _payload(result_type).items():
            encoded = header.getvalue() if name == member else _npy_bytes(value)
            archive.writestr(f"{name}.npy", encoded)

    read_array = np.lib.format.read_array

    def guarded_read(stream: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        # Do not actually allocate the claimed 8 GB (or overflowed product).
        # The real container must be rejected before NumPy reaches this decoder.
        assert stream.name != f"{member}.npy", "invalid payload reached NumPy allocation"
        return read_array(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", guarded_read)
    with pytest.raises(ValueError, match=f"member {member} is malformed"):
        loader(destination)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_npy_versions_byte_orders_and_compression_remain_reusable(
    tmp_path: Path,
    version: tuple[int, int],
    compression: int,
) -> None:
    destination = tmp_path / "versioned.npz"
    values = np.asarray([0.25, 2.0], dtype=">f8")
    payload = _payload("screen")
    payload["score__irv"] = values
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in payload.items():
            encoded = io.BytesIO()
            np.lib.format.write_array(encoded, value, version=version, allow_pickle=False)
            archive.writestr(f"{name}.npy", encoded.getvalue())

    loaded = load_score_archive(destination)
    np.testing.assert_array_equal(loaded["scores"]["irv"], [0.25, 2.0])


def test_numpy_members_without_filename_suffix_remain_reusable(tmp_path: Path) -> None:
    destination = tmp_path / "extensionless.npz"
    with ZipFile(destination, "w") as archive:
        for name, value in _payload("screen").items():
            archive.writestr(name, _npy_bytes(value))

    loaded = load_score_archive(destination)
    np.testing.assert_array_equal(loaded["scores"]["irv"], [1.0, 2.0])


@pytest.mark.parametrize(
    "encoded",
    [
        np.lib.format.magic(99, 0),
        np.lib.format.magic(1, 0) + b"\x01",
        np.lib.format.magic(2, 0) + (1_000_000_000).to_bytes(4, "little"),
        np.lib.format.magic(1, 0) + b"\x04\x00{}",
    ],
    ids=["unknown-version", "short-length", "oversized-header", "short-header"],
)
def test_invalid_npy_headers_fail_before_payload_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, encoded: bytes
) -> None:
    destination = tmp_path / "invalid-header.npz"
    with ZipFile(destination, "w") as archive:
        for name, value in _payload("screen").items():
            archive.writestr(f"{name}.npy", encoded if name == "score__irv" else _npy_bytes(value))

    read_array = np.lib.format.read_array

    def guarded_read(stream: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        assert stream.name != "score__irv.npy", "invalid header reached payload decoding"
        return read_array(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", guarded_read)
    with pytest.raises(ValueError, match="member score__irv is malformed"):
        load_score_archive(destination)


@pytest.mark.parametrize(
    ("loader", "result_type", "member"),
    [
        (load_score_archive, "screen", "index_names"),
        (load_score_archive, "screen", "error_names"),
        (load_score_archive, "screen", "error_messages"),
        (load_score_archive, "screen", "respondent_ids"),
        (load_response_time_archive, "response_time", "respondent_ids"),
    ],
)
@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_zero_width_metadata_is_rejected_before_python_list_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: ArchiveLoader,
    result_type: str,
    member: str,
    compression: int,
) -> None:
    destination = tmp_path / "zero-width-metadata.npz"
    payload = _payload(result_type)
    payload.update(
        {
            "error_names": np.asarray(["mad"]),
            "error_messages": np.asarray(["unconfigured item pairs"]),
            "respondent_ids": np.asarray(["A", "B"]),
        }
        if result_type == "screen"
        else {"respondent_ids": np.asarray(["A", "B"])}
    )
    header = io.BytesIO()
    # NumPy preserves U0 in its NPY reader. The empty binary payload would
    # otherwise permit a billion Python strings to be created by .tolist().
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<U0", "fortran_order": False, "shape": (1_000_000_000,)}
    )
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in payload.items():
            encoded = header.getvalue() if name == member else _npy_bytes(value)
            archive.writestr(f"{name}.npy", encoded)
    read_array = np.lib.format.read_array

    def guarded_read(stream: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        assert stream.name != f"{member}.npy", "zero-width metadata reached NumPy decoder"
        return read_array(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", guarded_read)
    with pytest.raises(ValueError, match=f"member {member} is malformed:.*positive item size"):
        loader(destination)


@pytest.mark.parametrize("shape", [(0, 2**64), (2**64, 0)])
@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_zero_element_shape_with_excessive_dimension_is_rejected_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    compression: int,
) -> None:
    destination = tmp_path / "excessive-dimension.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<U1", "fortran_order": False, "shape": shape}
    )
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in _payload("screen").items():
            encoded = header.getvalue() if name == "index_names" else _npy_bytes(value)
            archive.writestr(f"{name}.npy", encoded)
    read_array = np.lib.format.read_array

    def guarded_read(stream: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        assert stream.name != "index_names.npy", "excessive dimension reached NumPy decoder"
        return read_array(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", guarded_read)
    with pytest.raises(ValueError, match="member index_names is malformed:.*platform index range"):
        load_score_archive(destination)


@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_empty_zero_width_error_vectors_remain_supported(tmp_path: Path, compression: int) -> None:
    destination = tmp_path / "empty-errors.npz"
    payload = _payload("screen")
    payload["error_names"] = np.ndarray(0, dtype="U0")
    payload["error_messages"] = np.ndarray(0, dtype="U0")
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in payload.items():
            archive.writestr(f"{name}.npy", _npy_bytes(value))
    loaded = load_score_archive(destination)
    assert loaded["errors"] == {}
    np.testing.assert_array_equal(loaded["scores"]["irv"], [1.0, 2.0])


@pytest.mark.parametrize(
    ("loader", "result_type", "member"),
    [
        (load_score_archive, "screen", "score__irv"),
        (load_response_time_archive, "response_time", "scores"),
    ],
)
@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_forged_zip_uncompressed_size_cannot_authorize_numpy_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loader: ArchiveLoader,
    result_type: str,
    member: str,
    compression: int,
) -> None:
    destination = tmp_path / "forged-zip-size.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<f8", "fortran_order": False, "shape": (100_000_000,)}
    )
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in _payload(result_type).items():
            encoded = header.getvalue() if name == member else _npy_bytes(value)
            archive.writestr(f"{name}.npy", encoded)
    member_name = f"{member}.npy"
    with ZipFile(destination) as archive:
        local_offset = archive.getinfo(member_name).header_offset
    damaged = bytearray(destination.read_bytes())
    central_offset = damaged.rindex(member_name.encode()) - 46
    assert damaged[central_offset : central_offset + 4] == b"PK\x01\x02"
    declared_size = len(header.getvalue()) + 800_000_000
    # Both ZIP headers now agree with the NPY shape. The physical member still
    # contains only its valid NPY header, with the original (valid) checksum.
    damaged[local_offset + 22 : local_offset + 26] = declared_size.to_bytes(4, "little")
    damaged[central_offset + 24 : central_offset + 28] = declared_size.to_bytes(4, "little")
    destination.write_bytes(damaged)
    with ZipFile(destination) as archive:
        info = archive.getinfo(member_name)
        assert info.file_size == declared_size
        assert info.compress_size < 1_000
    read_array = np.lib.format.read_array

    def guarded_read(stream: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        assert stream.name != member_name, "forged ZIP size reached NumPy allocation"
        return read_array(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", guarded_read)
    with pytest.raises(ValueError, match=f"member {member} is malformed"):
        loader(destination)


@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_boolean_shape_dimension_has_contextual_error_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compression: int,
) -> None:
    destination = tmp_path / "boolean-dimension.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": "<U1", "fortran_order": False, "shape": (True,)}
    )
    header.write("i".encode("utf-32-le"))
    with ZipFile(destination, "w", compression=compression) as archive:
        for name, value in _payload("screen").items():
            encoded = header.getvalue() if name == "index_names" else _npy_bytes(value)
            archive.writestr(f"{name}.npy", encoded)
    read_array = np.lib.format.read_array

    def guarded_read(stream: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        assert stream.name != "index_names.npy", "boolean dimension reached NumPy decoder"
        return read_array(stream, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "read_array", guarded_read)
    with pytest.raises(ValueError, match="member index_names is malformed"):
        load_score_archive(destination)
