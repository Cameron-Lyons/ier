"""Reject release artifacts that are incomplete or contradict the project metadata."""

from __future__ import annotations

import tarfile
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from scripts.check_dist import _verify_sdist, _verify_wheel

if TYPE_CHECKING:
    from pathlib import Path

PROJECT: dict[str, object] = {
    "name": "insufficient-effort",
    "version": "1.2.3",
    "license": "MIT",
    "requires-python": ">=3.11",
}
PACKAGE_FILES = {"ier/__init__.py", "ier/cli.py", "ier/_cli_input.py", "ier/py.typed"}
DIST_INFO = "insufficient_effort-1.2.3.dist-info"
SDIST_ROOT = "insufficient_effort-1.2.3"


def _metadata(**overrides: str) -> bytes:
    headers = {
        "Metadata-Version": "2.4",
        "Name": "insufficient-effort",
        "Version": "1.2.3",
        "License-Expression": "MIT",
        "Requires-Python": ">=3.11",
        "License-File": "LICENSE",
    }
    headers.update(overrides)
    return ("\n".join(f"{name}: {value}" for name, value in headers.items()) + "\n\n").encode()


def _wheel_members() -> dict[str, bytes]:
    return {
        **dict.fromkeys(PACKAGE_FILES, b""),
        f"{DIST_INFO}/METADATA": _metadata(),
        f"{DIST_INFO}/entry_points.txt": b"[console_scripts]\nier = ier.cli:main\n",
        f"{DIST_INFO}/licenses/LICENSE": b"MIT license",
    }


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def _sdist_members() -> dict[str, bytes]:
    return {
        **dict.fromkeys((f"{SDIST_ROOT}/src/{name}" for name in PACKAGE_FILES), b""),
        f"{SDIST_ROOT}/LICENSE": b"MIT license",
        f"{SDIST_ROOT}/README.md": b"IER readme",
        f"{SDIST_ROOT}/PKG-INFO": _metadata(),
        f"{SDIST_ROOT}/pyproject.toml": (
            b'[project]\nname = "insufficient-effort"\nversion = "1.2.3"\n'
            b'license = "MIT"\nrequires-python = ">=3.11"\n'
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
    wheel = tmp_path / "example.whl"
    sdist = tmp_path / "example.tar.gz"
    _write_wheel(wheel, _wheel_members())
    _write_sdist(sdist, _sdist_members())

    _verify_wheel(wheel, PROJECT, PACKAGE_FILES)
    _verify_sdist(sdist, PROJECT, PACKAGE_FILES)


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
    ],
)
def test_wheel_requires_real_console_entry_point(tmp_path: Path, entry_points: str) -> None:
    path = tmp_path / "no-cli.whl"
    members = _wheel_members()
    members[f"{DIST_INFO}/entry_points.txt"] = entry_points.encode()
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="CLI entry point"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


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
