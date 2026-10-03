"""Reject release artifacts that are incomplete or contradict the project metadata."""

from __future__ import annotations

import base64
import csv
import hashlib
import struct
import tarfile
import zipfile
from io import BytesIO, StringIO
from types import SimpleNamespace
from typing import TYPE_CHECKING
from zlib import crc32

import pytest
from scripts.check_dist import _verify_metadata, _verify_wheel
from scripts.check_dist import _verify_sdist as _verify_sdist_with_support

if TYPE_CHECKING:
    from pathlib import Path

PROJECT: dict[str, object] = {
    "name": "insufficient-effort",
    "version": "1.2.3",
    "license": "MIT",
    "requires-python": ">=3.11",
    "dependencies": ["numpy>=1.26.0,<2.5"],
    "scripts": {"ier": "ier.cli:main"},
}
PACKAGE_FILES = {
    "ier/__init__.py": b'__version__ = "1.2.3"\n',
    "ier/cli.py": b"def main():\n    return 0\n",
    "ier/_cli_input.py": b"# Input parsing\n",
    "ier/py.typed": b"",
}
DIST_INFO = "insufficient_effort-1.2.3.dist-info"
SDIST_ROOT = "insufficient_effort-1.2.3"
WHEEL_FILENAME = "insufficient_effort-1.2.3-py3-none-any.whl"


def _verify_sdist(path: Path, project: dict[str, object], package_files: dict[str, bytes]) -> None:
    _verify_sdist_with_support(path, project, package_files, support_files={})


def _metadata(**overrides: str) -> bytes:
    headers = {
        "Metadata-Version": "2.4",
        "Name": "insufficient-effort",
        "Version": "1.2.3",
        "License-Expression": "MIT",
        "Requires-Python": ">=3.11",
        "License-File": "LICENSE",
        "Requires-Dist": "numpy<2.5,>=1.26.0",
    }
    headers.update(overrides)
    return ("\n".join(f"{name}: {value}" for name, value in headers.items()) + "\n\n").encode()


def _record(members: dict[str, bytes], *, algorithm: str = "sha256") -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output)
    record_name = f"{DIST_INFO}/RECORD"
    for name, contents in members.items():
        if not name.endswith("/") and name not in {
            record_name,
            f"{record_name}.jws",
            f"{record_name}.p7s",
        }:
            digest = hashlib.new(algorithm, contents).digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
            writer.writerow((name, f"{algorithm}={encoded}", len(contents)))
    writer.writerow((record_name, "", ""))
    return output.getvalue().encode("utf-8")


def _wheel_members() -> dict[str, bytes]:
    members = {
        **PACKAGE_FILES,
        f"{DIST_INFO}/METADATA": _metadata(),
        f"{DIST_INFO}/WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"),
        f"{DIST_INFO}/entry_points.txt": b"[console_scripts]\nier = ier.cli:main\n",
        f"{DIST_INFO}/licenses/LICENSE": b"MIT license",
    }
    return members | {f"{DIST_INFO}/RECORD": _record(members)}


def _write_wheel(path: Path, members: dict[str, bytes], *, update_record: bool = True) -> None:
    if update_record and f"{DIST_INFO}/RECORD" in members:
        members = members | {f"{DIST_INFO}/RECORD": _record(members)}
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            member = zipfile.ZipInfo(name)
            # Preserve malformed names verbatim: ZipInfo normalizes backslashes
            # on Windows and truncates NUL suffixes on every platform.
            member.filename = name
            archive.writestr(member, contents)


def _sdist_members() -> dict[str, bytes]:
    return {
        **{f"{SDIST_ROOT}/src/{name}": contents for name, contents in PACKAGE_FILES.items()},
        f"{SDIST_ROOT}/LICENSE": b"MIT license",
        f"{SDIST_ROOT}/README.md": b"IER readme",
        f"{SDIST_ROOT}/PKG-INFO": _metadata(),
        f"{SDIST_ROOT}/pyproject.toml": (
            b'[project]\nname = "insufficient-effort"\nversion = "1.2.3"\n'
            b'license = "MIT"\nrequires-python = ">=3.11"\n'
            b'dependencies = ["numpy>=1.26.0,<2.5"]\n'
            b'[project.scripts]\nier = "ier.cli:main"\n'
        ),
    }


def _write_sdist(path: Path, members: dict[str, bytes], *, directory: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in members.items():
            member = tarfile.TarInfo(name)
            if name == directory:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(contents)
                archive.addfile(member, BytesIO(contents))


def test_complete_distributions_pass(tmp_path: Path) -> None:
    wheel = tmp_path / WHEEL_FILENAME
    sdist = tmp_path / "example.tar.gz"
    _write_wheel(wheel, _wheel_members())
    _write_sdist(sdist, _sdist_members())

    _verify_wheel(wheel, PROJECT, PACKAGE_FILES)
    _verify_sdist(sdist, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("algorithm", ["sha256", "sha512", "sha3_256", "blake2s"])
def test_wheel_verifies_secure_record_hashes_and_csv_quoted_resource_names(
    tmp_path: Path, algorithm: str
) -> None:
    path = tmp_path / WHEEL_FILENAME
    members = _wheel_members() | {"ier/data/entrée,values.csv": b"1,2\n"}
    members[f"{DIST_INFO}/RECORD"] = _record(members, algorithm=algorithm)
    _write_wheel(path, members, update_record=False)

    _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_record_excludes_directories_and_legacy_signature_files(tmp_path: Path) -> None:
    path = tmp_path / WHEEL_FILENAME
    members = _wheel_members() | {
        "ier/data/": b"",
        f"{DIST_INFO}/RECORD.jws": b"legacy signature",
        f"{DIST_INFO}/RECORD.p7s": b"legacy signature",
    }
    _write_wheel(path, members)

    _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "record,error",
    [
        (b"\xff", "invalid RECORD"),
        (b'"unterminated', "invalid RECORD"),
        (b"one,two\n", "three columns"),
        (b"one,two,three,four\n", "three columns"),
        (b"\n", "three columns"),
        (b"same,,\nsame,,\n", "duplicate entries"),
    ],
)
def test_wheel_rejects_malformed_integrity_manifest(
    tmp_path: Path, record: bytes, error: str
) -> None:
    path = tmp_path / "malformed-record.whl"
    members = _wheel_members() | {f"{DIST_INFO}/RECORD": record}
    _write_wheel(path, members, update_record=False)

    with pytest.raises(ValueError, match=error):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("mutation", ["missing", "extra", "self-missing", "signature"])
def test_wheel_requires_record_to_describe_exactly_the_installed_files(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "incomplete-record.whl"
    members = _wheel_members()
    record_name = f"{DIST_INFO}/RECORD"
    if mutation == "missing":
        members[record_name] = b"\n".join(
            row
            for row in members[record_name].splitlines()
            if not row.startswith(f"{DIST_INFO}/licenses/LICENSE,".encode())
        )
    elif mutation == "self-missing":
        members[record_name] = b"\n".join(
            row
            for row in members[record_name].splitlines()
            if not row.startswith(f"{record_name},".encode())
        )
    else:
        name = "ier/ghost.txt" if mutation == "extra" else f"{record_name}.jws"
        if mutation == "signature":
            members[name] = b"legacy signature"
        members[record_name] += f"{name},sha256=hash,4\n".encode()
    _write_wheel(path, members, update_record=False)

    with pytest.raises(ValueError, match="RECORD does not match archive members"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "digest,size,error",
    [
        ("", "11", "missing a secure hash"),
        ("sha256", "11", "missing a secure hash"),
        ("sha256=", "11", "missing a secure hash"),
        ("not-a-hash=value", "11", "unsupported hash algorithm"),
        ("md5=value", "11", "sha256 or stronger"),
        ("sha1=value", "11", "sha256 or stronger"),
        ("sha224=value", "11", "sha256 or stronger"),
        ("shake_256=value", "11", "sha256 or stronger"),
        ("sha256=wrong-hash", "11", "hash does not match"),
        ("sha256=hash", "-11", "invalid file size"),
        ("sha256=hash", "11.0", "invalid file size"),
        ("sha256=hash", " 11", "invalid file size"),
        ("sha256=hash", "１１", "invalid file size"),
        ("sha256=hash", "12", "size does not match"),
    ],
)
def test_wheel_rejects_unusable_hashes_and_false_record_sizes(
    tmp_path: Path, digest: str, size: str, error: str
) -> None:
    path = tmp_path / "false-integrity.whl"
    members = _wheel_members()
    record_name = f"{DIST_INFO}/RECORD"
    target = f"{DIST_INFO}/licenses/LICENSE"
    record = members[record_name].decode()
    lines = record.splitlines(keepends=True)
    members[record_name] = "".join(
        f"{target},{digest},{size}\n" if line.startswith(f"{target},") else line for line in lines
    ).encode()
    _write_wheel(path, members, update_record=False)

    with pytest.raises(ValueError, match=error):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("suffix", [",sha256=hash,\n", ",,1\n"])
def test_wheel_record_does_not_hash_or_size_itself(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / "self-record.whl"
    members = _wheel_members()
    name = f"{DIST_INFO}/RECORD"
    members[name] = members[name].replace(f"{name},,\r\n".encode(), f"{name}{suffix}".encode())
    _write_wheel(path, members, update_record=False)

    with pytest.raises(ValueError, match="cannot hash or size itself"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_accepts_record_with_optional_sizes_omitted(tmp_path: Path) -> None:
    path = tmp_path / WHEEL_FILENAME
    members = _wheel_members()
    name = f"{DIST_INFO}/RECORD"
    rows = list(csv.reader(StringIO(members[name].decode())))
    output = StringIO(newline="")
    csv.writer(output).writerows((entry, digest, "") for entry, digest, _ in rows)
    members[name] = output.getvalue().encode()
    _write_wheel(path, members, update_record=False)

    _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_hashes_all_resource_chunks_even_when_sizes_and_zip_checksums_match(
    tmp_path: Path,
) -> None:
    path = tmp_path / WHEEL_FILENAME
    name = "ier/data/resource.bin"
    members = _wheel_members() | {name: b"x" * (2 * 1024 * 1024 + 1)}
    members[f"{DIST_INFO}/RECORD"] = _record(members)
    _write_wheel(path, members, update_record=False)
    _verify_wheel(path, PROJECT, PACKAGE_FILES)

    members[name] = members[name][:-1] + b"y"
    _write_wheel(path, members, update_record=False)
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    with pytest.raises(ValueError, match="RECORD hash does not match ier/data/resource.bin"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "metadata,error",
    [
        (b"not wheel metadata", "Wheel-Version"),
        (b"Root-Is-Purelib: true\nTag: py3-none-any\n", "Wheel-Version"),
        (b"Wheel-Version: 2.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n", "Wheel-Version"),
        (
            b"Wheel-Version: 1.0\nWheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            "Wheel-Version",
        ),
        (b"Wheel-Version: 1.0\nTag: py3-none-any\n", "Root-Is-Purelib"),
        (b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-any\n", "Root-Is-Purelib"),
        (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n",
            "Root-Is-Purelib",
        ),
        (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n", "WHEEL Tag"),
        (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: malformed\n", "WHEEL Tag"),
        (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: cp311-cp311-win_amd64\n",
            "WHEEL Tag",
        ),
        (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\nTag: py3-none-any\n",
            "WHEEL Tag",
        ),
        (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\nBuild: 1\n",
            "WHEEL Build",
        ),
    ],
)
def test_wheel_rejects_invalid_installation_metadata_even_with_matching_record(
    tmp_path: Path, metadata: bytes, error: str
) -> None:
    path = tmp_path / WHEEL_FILENAME
    members = _wheel_members() | {f"{DIST_INFO}/WHEEL": metadata}
    _write_wheel(path, members)

    with pytest.raises(ValueError, match=error):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "filename,error",
    [
        ("example.whl", "invalid wheel filename"),
        ("different_name-1.2.3-py3-none-any.whl", "distribution does not match project.name"),
        ("insufficient_effort-1.2.4-py3-none-any.whl", "version does not match project.version"),
        ("insufficient_effort-1.2.3-py311-none-any.whl", "WHEEL Tag"),
    ],
)
def test_wheel_filename_must_match_its_contents(tmp_path: Path, filename: str, error: str) -> None:
    path = tmp_path / filename
    _write_wheel(path, _wheel_members())

    with pytest.raises(ValueError, match=error):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_rejects_platform_specific_tags_for_this_pure_python_package(tmp_path: Path) -> None:
    path = tmp_path / "insufficient_effort-1.2.3-cp311-cp311-win_amd64.whl"
    members = _wheel_members()
    members[f"{DIST_INFO}/WHEEL"] = members[f"{DIST_INFO}/WHEEL"].replace(
        b"py3-none-any", b"cp311-cp311-win_amd64"
    )
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="platform-independent Python package"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_expands_filename_tags_and_preserves_build_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "insufficient_effort-1.2.3-002abc-py3.py311-none-any.whl"
    metadata = (
        b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\n"
        b"Tag: py3-none-any\nTag: py311-none-any\nBuild: 002abc\n"
    )
    members = _wheel_members() | {f"{DIST_INFO}/WHEEL": metadata}
    _write_wheel(path, members)

    _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("build", [b"", b"Build: 2abc\n", b"Build: 002abc\nBuild: 002abc\n"])
def test_wheel_build_metadata_must_match_filename(tmp_path: Path, build: bytes) -> None:
    path = tmp_path / "insufficient_effort-1.2.3-002abc-py3-none-any.whl"
    members = _wheel_members()
    members[f"{DIST_INFO}/WHEEL"] += build
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="WHEEL Build"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("missing", sorted(_wheel_members()))
def test_wheel_rejects_missing_source_and_metadata(tmp_path: Path, missing: str) -> None:
    path = tmp_path / "incomplete.whl"
    members = _wheel_members()
    del members[missing]
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="missing required files"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("missing", sorted(_sdist_members()))
def test_sdist_rejects_missing_source_and_metadata(tmp_path: Path, missing: str) -> None:
    path = tmp_path / "incomplete.tar.gz"
    members = _sdist_members()
    del members[missing]
    _write_sdist(path, members)

    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "header,value",
    [
        ("Name", "another-package"),
        ("Version", "1.2.2"),
        ("License-Expression", "Apache-2.0"),
        ("Requires-Python", ">=3.14"),
        ("License-File", "COPYING"),
    ],
)
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_incorrect_metadata(
    tmp_path: Path, header: str, value: str, kind: str
) -> None:
    if kind == "wheel":
        path = tmp_path / "incorrect.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = _metadata(**{header: value})
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "incorrect.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = _metadata(**{header: value})
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="expected|does not declare LICENSE"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "entry_points",
    [
        "[console_scripts]\n# ier = ier.cli:main\n",
        "[unrelated]\nier = ier.cli:main\n",
        "[console_scripts]\nier = ier.cli:missing\n",
        "[console_scripts]\nIER = ier.cli:main\n",
        "[console_scripts]\nier = ier.cli:main\nstale = ier.cli:removed\n",
        "[DEFAULT]\nier = ier.cli:main\n[console_scripts]\n",
        "[console_scripts]\nier: ier.cli:main\n",
    ],
)
def test_wheel_requires_real_console_entry_point(tmp_path: Path, entry_points: str) -> None:
    path = tmp_path / "no-cli.whl"
    members = _wheel_members()
    members[f"{DIST_INFO}/entry_points.txt"] = entry_points.encode()
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="CLI entry point"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_checks_all_declared_console_scripts(tmp_path: Path) -> None:
    path = tmp_path / WHEEL_FILENAME
    project = PROJECT | {"scripts": {"ier": "ier.cli:main", "ier-extra": "ier.cli:extra"}}
    members = _wheel_members()
    _write_wheel(path, members)
    with pytest.raises(ValueError, match="CLI entry points"):
        _verify_wheel(path, project, PACKAGE_FILES)

    members[f"{DIST_INFO}/entry_points.txt"] += b"ier-extra = ier.cli:extra\n"
    _write_wheel(path, members)
    _verify_wheel(path, project, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "name",
    [
        "ier/stale.py",
        "ier/nested/stale.py",
        "ier/stale.pyc",
        "IER/stale.py",
        "ier/STALE.PY",
        "ier/stale.so",
        "ier/stale.pyd",
    ],
)
def test_distributions_reject_removed_package_modules(tmp_path: Path, kind: str, name: str) -> None:
    if kind == "wheel":
        path = tmp_path / "stale-module.whl"
        _write_wheel(path, _wheel_members() | {name: b"# leftover build module\n"})
        verifier = _verify_wheel
    else:
        path = tmp_path / "stale-module.tar.gz"
        _write_sdist(
            path, _sdist_members() | {f"{SDIST_ROOT}/src/{name}": b"# leftover build module\n"}
        )
        verifier = _verify_sdist
    with pytest.raises(ValueError, match="unexpected package files"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("source_directory", ["SRC", "Src"])
def test_sdist_rejects_stale_modules_under_source_directory_case_aliases(
    tmp_path: Path, source_directory: str
) -> None:
    path = tmp_path / "source-alias.tar.gz"
    members = _sdist_members() | {
        f"{SDIST_ROOT}/{source_directory}/ier/stale.py": b"# leftover build module\n"
    }
    _write_sdist(path, members)
    with pytest.raises(ValueError, match="unexpected package files"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_duplicate_archive_members(tmp_path: Path, kind: str) -> None:
    if kind == "wheel":
        path = tmp_path / "duplicate.whl"
        _write_wheel(path, _wheel_members())
        with zipfile.ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("ier/cli.py", PACKAGE_FILES["ier/cli.py"])
        verifier = _verify_wheel
    else:
        path = tmp_path / "duplicate.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            members = _sdist_members()
            for name in [*members, f"{SDIST_ROOT}/src/ier/cli.py"]:
                member = tarfile.TarInfo(name)
                contents = members[name]
                member.size = len(contents)
                archive.addfile(member, BytesIO(contents))
        verifier = _verify_sdist
    with pytest.raises(ValueError, match="duplicate archive members"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("name", ["IER/cli.py", "ier/CLI.py", "ier/cli.py ", "ier/cli.py."])
def test_distributions_reject_windows_aliases_of_verified_sources(
    tmp_path: Path, kind: str, name: str
) -> None:
    if kind == "wheel":
        path = tmp_path / "alias.whl"
        _write_wheel(path, _wheel_members() | {name: b"# alias of verified source\n"})
        verifier = _verify_wheel
    else:
        path = tmp_path / "alias.tar.gz"
        _write_sdist(
            path, _sdist_members() | {f"{SDIST_ROOT}/src/{name}": b"# alias of verified source\n"}
        )
        verifier = _verify_sdist
    with pytest.raises(ValueError, match="duplicate archive members"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "name",
    [
        "/escaped.py",
        "../escaped.py",
        "ier/../escaped.py",
        "ier/./escaped.py",
        "ier//escaped.py",
        "ier\\evil.py",
        "C:/escaped.py",
        "C:escaped.py",
        "ier/cli.py:evil",
        "ier/NUL.txt",
        "ier/CON.py",
        "ier/lPt1.foo",
        "ier/COM¹.txt",
        "ier/name?.txt",
        "ier/name\t.txt",
        "ier/directory./resource.txt",
    ],
)
def test_distributions_reject_paths_that_escape_or_depend_on_the_extractor(
    tmp_path: Path, kind: str, name: str
) -> None:
    if kind == "wheel":
        path = tmp_path / "unsafe-path.whl"
        _write_wheel(path, _wheel_members() | {name: b"# unsafe path\n"})
        verifier = _verify_wheel
    else:
        path = tmp_path / "unsafe-path.tar.gz"
        _write_sdist(path, _sdist_members() | {name: b"# unsafe path\n"})
        verifier = _verify_sdist
    with pytest.raises(ValueError, match="unsafe archive paths"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("name", [r"ier\resource.txt", r"ier\cli.py"])
def test_wheel_checks_original_names_when_windows_normalizes_backslashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    path = tmp_path / "windows-path.whl"
    _write_wheel(path, _wheel_members() | {name: b"# unsafe path\n"})
    # Exercise Windows ZipInfo behavior on every CI platform without changing
    # the separators used by pathlib, pytest, or other users of the os module.
    windows_os = SimpleNamespace(**(vars(zipfile.os) | {"sep": "\\", "altsep": "/"}))
    monkeypatch.setattr(zipfile, "os", windows_os)

    with zipfile.ZipFile(path) as archive:
        member = archive.infolist()[-1]
        assert member.orig_filename == name
        assert member.filename == name.replace("\\", "/")
    with pytest.raises(ValueError, match="unsafe archive paths"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("name", ["unrelated\0resource.txt", "ier/resource.txt\0suffix"])
def test_wheel_rejects_names_truncated_at_nul(tmp_path: Path, name: str) -> None:
    path = tmp_path / "nul-path.whl"
    _write_wheel(path, _wheel_members() | {name: b"# unsafe path\n"})

    with zipfile.ZipFile(path) as archive:
        member = archive.infolist()[-1]
        assert member.orig_filename == name
        assert member.filename == name.split("\0", 1)[0]
    with pytest.raises(ValueError, match="unsafe archive paths"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_wheel_checks_effective_names_from_unicode_path_extra_fields(tmp_path: Path) -> None:
    path = tmp_path / "unicode-path.whl"
    _write_wheel(path, _wheel_members())
    member = zipfile.ZipInfo("ier/resource.txt")
    unicode_name = b"ier/resource?.txt"
    member.extra = (
        struct.pack("<HHBL", 0x7075, 5 + len(unicode_name), 1, crc32(member.filename.encode()))
        + unicode_name
    )
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(member, b"# unsafe effective path\n")

    with zipfile.ZipFile(path) as archive:
        member = archive.infolist()[-1]
        assert member.orig_filename == "ier/resource.txt"
        if member.filename == member.orig_filename:
            pytest.skip("This Python does not interpret Unicode Path extra fields")
        assert member.filename == unicode_name.decode()
    with pytest.raises(ValueError, match="unsafe archive paths"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_sdist_rejects_links_even_when_required_source_files_are_complete(
    tmp_path: Path, link_type: bytes
) -> None:
    path = tmp_path / "linked-source.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in _sdist_members().items():
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, BytesIO(contents))
        link = tarfile.TarInfo(f"{SDIST_ROOT}/src/ier/linked.py")
        link.type = link_type
        link.linkname = f"{SDIST_ROOT}/src/ier/cli.py"
        archive.addfile(link)
    with pytest.raises(ValueError, match="archive links"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("header", ["Name", "Version", "License-Expression", "Requires-Python"])
def test_distributions_reject_ambiguous_single_value_metadata(
    tmp_path: Path, kind: str, header: str
) -> None:
    from email.parser import BytesParser

    metadata = _metadata()
    value = BytesParser().parsebytes(metadata)[header]
    metadata = metadata.rstrip(b"\n") + f"\n{header}: {value}\n\n".encode()
    if kind == "wheel":
        path = tmp_path / "duplicate-header.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = metadata
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "duplicate-header.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = metadata
        _write_sdist(path, members)
        verifier = _verify_sdist
    with pytest.raises(ValueError, match=header):
        verifier(path, PROJECT, PACKAGE_FILES)


def test_sdist_rejects_bundled_project_version_drift(tmp_path: Path) -> None:
    path = tmp_path / "stale-project.tar.gz"
    members = _sdist_members()
    name = f"{SDIST_ROOT}/pyproject.toml"
    members[name] = members[name].replace(b'"1.2.3"', b'"1.2.2"')
    _write_sdist(path, members)

    with pytest.raises(ValueError, match="bundled project.version"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


def test_sdist_does_not_accept_directory_as_required_file(tmp_path: Path) -> None:
    path = tmp_path / "bad-source.tar.gz"
    _write_sdist(path, _sdist_members(), directory=f"{SDIST_ROOT}/src/ier/_cli_input.py")

    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("name", PACKAGE_FILES)
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_stale_source_contents(tmp_path: Path, name: str, kind: str) -> None:
    if kind == "wheel":
        path = tmp_path / "stale-source.whl"
        members = _wheel_members()
        members[name] = b"# stale artifact from a different checkout\n"
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "stale-source.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/src/{name}"] = b"# stale artifact from a different checkout\n"
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="source file .* does not match the checkout"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "before,after,key",
    [
        (b"numpy>=1.26.0,<2.5", b"numpy>=1.26.0", "dependencies"),
        (b'dependencies = ["numpy>=1.26.0,<2.5"]\n', b"", "dependencies"),
        (b'ier = "ier.cli:main"', b'ier = "ier.cli:missing"', "scripts"),
        (b"[project.scripts]", b'dynamic = ["description"]\n[project.scripts]', "dynamic"),
    ],
)
def test_sdist_rejects_bundled_dependency_and_entry_point_drift(
    tmp_path: Path, before: bytes, after: bytes, key: str
) -> None:
    path = tmp_path / "stale-configuration.tar.gz"
    members = _sdist_members()
    name = f"{SDIST_ROOT}/pyproject.toml"
    members[name] = members[name].replace(before, after)
    _write_sdist(path, members)

    with pytest.raises(ValueError, match=rf"bundled project\.{key}"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "requirement",
    [
        None,
        "numpy>=1.26.0",
        "numpy>=1.27.0,<2.5",
        'numpy>=1.26.0,<2.5; extra == "plot"',
        "unrelated-package>=1.26.0,<2.5",
        "numpy=>1.26.0",
        "numpy<2.5,>=1.26.0\nRequires-Dist: numpy<2.5,>=1.26.0",
    ],
)
def test_distributions_reject_missing_wrong_or_duplicate_runtime_requirements(
    tmp_path: Path, kind: str, requirement: str | None
) -> None:
    metadata = _metadata(**{"Requires-Dist": requirement or "numpy<2.5,>=1.26.0"})
    if requirement is None:
        metadata = metadata.replace(b"Requires-Dist: numpy<2.5,>=1.26.0\n", b"")
    if kind == "wheel":
        path = tmp_path / "wrong-dependency.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = metadata
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "wrong-dependency.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = metadata
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="Requires-Dist"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_undeclared_provided_extras(tmp_path: Path, kind: str) -> None:
    metadata = _metadata(**{"Provides-Extra": "unexpected-extra"})
    if kind == "wheel":
        path = tmp_path / "wrong-extra.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = metadata
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "wrong-extra.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = metadata
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="Provides-Extra"):
        verifier(path, PROJECT, PACKAGE_FILES)


def test_dependency_metadata_supports_pep508_extras_direct_references_and_markers() -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {
        "dependencies": [
            'Demo[fast] >= 1, != 2; python_version >= "3.11"',
            'direct @ https://example.com/direct.whl ; sys_platform == "linux"',
        ],
        "optional-dependencies": {
            "Fancy_Plot": ['plotter>=2; python_version >= "3.11" or platform_system == "Linux"'],
            "empty": [],
        },
    }
    metadata = BytesParser().parsebytes(_metadata())
    del metadata["Requires-Dist"]
    metadata["Requires-Dist"] = 'demo[fast]!=2,>=1; python_version >= "3.11"'
    metadata["Requires-Dist"] = 'direct @ https://example.com/direct.whl ; sys_platform == "linux"'
    metadata["Requires-Dist"] = (
        'plotter>=2; (python_version >= "3.11" or platform_system == "Linux") '
        'and extra == "fancy-plot"'
    )
    metadata["Provides-Extra"] = "empty"
    metadata["Provides-Extra"] = "fancy-plot"
    _verify_metadata(Path("valid.whl"), metadata, project)

    # Omitting parentheses broadens the optional requirement into a mandatory
    # dependency for some environments; that change must not pass verification.
    del metadata["Requires-Dist"]
    metadata["Requires-Dist"] = 'demo[fast]!=2,>=1; python_version >= "3.11"'
    metadata["Requires-Dist"] = 'direct @ https://example.com/direct.whl ; sys_platform == "linux"'
    metadata["Requires-Dist"] = (
        'plotter>=2; python_version >= "3.11" or platform_system == "Linux" '
        'and extra == "fancy-plot"'
    )
    with pytest.raises(ValueError, match="Requires-Dist"):
        _verify_metadata(Path("wrong-marker.whl"), metadata, project)


@pytest.mark.parametrize("provided", [[], ["plot", "plot"], ["wrong"]])
def test_declared_optional_extras_require_complete_unique_metadata(provided: list[str]) -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {"optional-dependencies": {"plot": []}}
    metadata = BytesParser().parsebytes(_metadata())
    for extra in provided:
        metadata["Provides-Extra"] = extra
    with pytest.raises(ValueError, match="Provides-Extra"):
        _verify_metadata(Path("wrong-extra.whl"), metadata, project)


def test_direct_reference_metadata_must_keep_the_declared_url() -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {"dependencies": ["direct @ https://example.com/direct.whl"]}
    metadata = BytesParser().parsebytes(
        _metadata(**{"Requires-Dist": "direct @ https://example.com/other.whl"})
    )
    with pytest.raises(ValueError, match="Requires-Dist"):
        _verify_metadata(Path("wrong-reference.whl"), metadata, project)


@pytest.mark.parametrize(
    "requirement",
    [
        'matplotlib>=3.8; extra == "plot"',
        'matplotlib>=3.8,<4; extra == "wrong"',
        "matplotlib>=3.8,<4",
    ],
)
def test_optional_dependencies_keep_their_constraints_and_extra_marker(requirement: str) -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {"optional-dependencies": {"plot": ["matplotlib>=3.8,<4"]}}
    metadata = BytesParser().parsebytes(_metadata(**{"Provides-Extra": "plot"}))
    metadata["Requires-Dist"] = requirement
    with pytest.raises(ValueError, match="Requires-Dist"):
        _verify_metadata(Path("wrong-optional.whl"), metadata, project)


@pytest.mark.parametrize(
    "support",
    [
        "tests/fixtures/parity/reference.json",
        "tests/fixtures/distributions/README.txt",
        "tests/fixtures/distributions/reference.whl",
        "tests/fixtures/distributions/reference.tar.gz",
        "tests/__init__.py",
        "tests/test_property_invariants.py",
        "scripts/check_dist.py",
        "scripts/check.sh",
        "benchmarks/_measurement.py",
        "benchmarks/bench_screen.py",
        "uv.lock",
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
    ],
)
def test_sdist_requires_original_test_support(tmp_path: Path, support: str) -> None:
    path = tmp_path / "test-support.tar.gz"
    members = _sdist_members()
    expected = {support: members.pop(f"{SDIST_ROOT}/{support}", b"test support")}
    _write_sdist(path, members)
    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist_with_support(path, PROJECT, PACKAGE_FILES, support_files=expected)

    members[f"{SDIST_ROOT}/{support}"] = expected[support]
    _write_sdist(path, members)
    _verify_sdist_with_support(path, PROJECT, PACKAGE_FILES, support_files=expected)

    members[f"{SDIST_ROOT}/{support}"] = b"stale test support"
    _write_sdist(path, members)
    with pytest.raises(ValueError, match="source file .* does not match the checkout"):
        _verify_sdist_with_support(path, PROJECT, PACKAGE_FILES, support_files=expected)
