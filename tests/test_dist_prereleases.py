"""Verify real setuptools artifacts preserve SemVer sources and normalized metadata."""

from __future__ import annotations

import base64
import csv
import hashlib
import tarfile
import zipfile
from functools import partial
from io import StringIO
from pathlib import Path

import pytest
from scripts.check_dist import _verify_sdist, _verify_wheel

FIXTURES = Path(__file__).parent / "fixtures" / "distributions"
PROJECT: dict[str, object] = {
    "name": "insufficient-effort",
    "license": "MIT",
    "license-files": ["LICENSE"],
    "requires-python": ">=3.11",
    "dependencies": ["numpy>=1.26.0,<2.5"],
    "scripts": {"ier": "ier.cli:main"},
}
SOURCES = {
    "ier/__init__.py": b'"""Distribution verifier fixture."""\n',
    "ier/cli.py": b"def main():\n    return 0\n",
    "ier/py.typed": b"",
}
VERSIONS = [
    ("1.2.3", "1.2.3"),
    ("1.2.3-alpha.2", "1.2.3a2"),
    ("1.2.3-beta.3", "1.2.3b3"),
    ("1.2.3-rc.1", "1.2.3rc1"),
    ("1.2.3-dev.6", "1.2.3.dev6"),
    ("1.2.3+Build-Tag.007", "1.2.3+build.tag.7"),
    ("1.2.3-rc.1+Build.007", "1.2.3rc1+build.7"),
]


@pytest.mark.parametrize("source,normalized", VERSIONS)
def test_real_setuptools_artifacts_accept_only_their_project_version(
    source: str, normalized: str
) -> None:
    project = PROJECT | {"version": source}
    wheel = FIXTURES / f"insufficient_effort-{normalized}-py3-none-any.whl"
    sdist = FIXTURES / f"insufficient_effort-{normalized}.tar.gz"

    _verify_wheel(wheel, project, SOURCES)
    _verify_sdist(sdist, project, SOURCES, support_files={})

    # A different SemVer cannot pass merely because the artifact itself is valid.
    wrong_project = project | {"version": "9.9.9-rc.1"}
    with pytest.raises(ValueError, match="missing required files"):
        _verify_wheel(wheel, wrong_project, SOURCES)
    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist(sdist, wrong_project, SOURCES, support_files={})


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_real_prerelease_artifact_rejects_wrong_metadata_version(tmp_path: Path, kind: str) -> None:
    project = PROJECT | {"version": "1.2.3-rc.1"}
    root = "insufficient_effort-1.2.3rc1"
    if kind == "wheel":
        original = FIXTURES / f"{root}-py3-none-any.whl"
        corrupt = tmp_path / original.name
        with zipfile.ZipFile(original) as archive, zipfile.ZipFile(corrupt, "w") as output:
            for member in archive.infolist():
                contents = archive.read(member)
                if member.filename.endswith("/METADATA"):
                    contents = contents.replace(b"Version: 1.2.3rc1\n", b"Version: 1.2.3rc2\n")
                output.writestr(member, contents)
        verifier = _verify_wheel
    else:
        original = FIXTURES / f"{root}.tar.gz"
        corrupt = tmp_path / original.name
        _rewrite_sdist(original, corrupt, f"{root}/PKG-INFO", b"1.2.3rc1", b"1.2.3rc2")
        verifier = partial(_verify_sdist, support_files={})

    with pytest.raises(ValueError, match=r"Version=.*1\.2\.3rc2.*expected.*1\.2\.3rc1"):
        verifier(corrupt, project, SOURCES)


def test_real_setuptools_wheel_rejects_content_corruption_with_valid_zip_checksums(
    tmp_path: Path,
) -> None:
    original = FIXTURES / "insufficient_effort-1.2.3-py3-none-any.whl"
    corrupt = tmp_path / original.name
    with zipfile.ZipFile(original) as archive, zipfile.ZipFile(corrupt, "w") as output:
        for member in archive.infolist():
            contents = archive.read(member)
            if member.filename.endswith("/licenses/LICENSE"):
                contents = b"X" + contents[1:]
            output.writestr(member, contents)
    with zipfile.ZipFile(corrupt) as archive:
        assert archive.testzip() is None

    with pytest.raises(ValueError, match="RECORD hash does not match .*licenses/LICENSE"):
        _verify_wheel(corrupt, PROJECT | {"version": "1.2.3"}, SOURCES)


@pytest.mark.parametrize(
    "before,after,error",
    [
        (b"Wheel-Version: 1.0", b"Wheel-Version: 999.0", "Wheel-Version"),
        (b"Root-Is-Purelib: true", b"Root-Is-Purelib: false", "Root-Is-Purelib"),
        (b"Tag: py3-none-any", b"Tag: cp311-cp311-win_amd64", "WHEEL Tag"),
    ],
)
def test_real_setuptools_wheel_rejects_invalid_installation_metadata_after_rehashing(
    tmp_path: Path, before: bytes, after: bytes, error: str
) -> None:
    original = FIXTURES / "insufficient_effort-1.2.3-py3-none-any.whl"
    corrupt = tmp_path / original.name
    dist_info = "insufficient_effort-1.2.3.dist-info"
    record_name = f"{dist_info}/RECORD"
    with zipfile.ZipFile(original) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    wheel_metadata = members[f"{dist_info}/WHEEL"]
    assert before in wheel_metadata
    members[f"{dist_info}/WHEEL"] = wheel_metadata.replace(before, after)
    record = StringIO(newline="")
    writer = csv.writer(record)
    for name, contents in members.items():
        if name != record_name:
            digest = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=")
            writer.writerow((name, f"sha256={digest.decode('ascii')}", len(contents)))
    writer.writerow((record_name, "", ""))
    members[record_name] = record.getvalue().encode()
    with zipfile.ZipFile(corrupt, "w") as output:
        for name, contents in members.items():
            output.writestr(name, contents)

    with pytest.raises(ValueError, match=error):
        _verify_wheel(corrupt, PROJECT | {"version": "1.2.3"}, SOURCES)


def _rewrite_sdist(original: Path, output: Path, name: str, before: bytes, after: bytes) -> None:
    from io import BytesIO

    with tarfile.open(original) as archive, tarfile.open(output, "w:gz") as rewritten:
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                with stream:
                    contents = stream.read()
                if member.name == name:
                    contents = contents.replace(before, after)
                member.size = len(contents)
                rewritten.addfile(member, BytesIO(contents))
            else:
                rewritten.addfile(member)


def test_prerelease_sdist_keeps_the_original_source_version(tmp_path: Path) -> None:
    original = FIXTURES / "insufficient_effort-1.2.3rc1.tar.gz"
    corrupt = tmp_path / original.name
    _rewrite_sdist(
        original,
        corrupt,
        "insufficient_effort-1.2.3rc1/pyproject.toml",
        b'"1.2.3-rc.1"',
        b'"1.2.3rc1"',
    )

    with pytest.raises(ValueError, match=r"bundled project\.version"):
        _verify_sdist(corrupt, PROJECT | {"version": "1.2.3-rc.1"}, SOURCES, support_files={})
