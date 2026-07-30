"""Verify metadata and required files in built wheel and source distributions."""

from __future__ import annotations

import argparse
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from email.message import Message


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _project_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if not isinstance(project, dict):
        raise ValueError("pyproject.toml [project] table is invalid")
    return project


def _project_value(project: dict[str, object], key: str) -> str:
    value = project.get(key)
    if not isinstance(value, str):
        raise ValueError(f"pyproject.toml project.{key} must be a string")
    return value


def _wheel_metadata(archive: zipfile.ZipFile, dist_info: str) -> Message:
    metadata = archive.read(f"{dist_info}/METADATA")
    return BytesParser().parsebytes(metadata)


def _verify_wheel(path: Path, project: dict[str, object]) -> None:
    name = _project_value(project, "name")
    version = _project_value(project, "version")
    license_expression = _project_value(project, "license")
    requires_python = _project_value(project, "requires-python")
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"

    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        required_members = {
            "ier/py.typed",
            f"{dist_info}/METADATA",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
        }
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")

        metadata = _wheel_metadata(archive, dist_info)
        expected_headers = {
            "Name": name,
            "Version": version,
            "License-Expression": license_expression,
            "Requires-Python": requires_python,
        }
        for header, expected in expected_headers.items():
            actual = metadata.get(header)
            _require(
                actual == expected,
                f"{path.name} has {header}={actual!r}; expected {expected!r}",
            )

        license_files = metadata.get_all("License-File", failobj=[])
        _require("LICENSE" in license_files, f"{path.name} does not declare LICENSE metadata")
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        _require(
            "ier = ier.cli:main" in entry_points,
            f"{path.name} is missing the ier CLI entry point",
        )


def _verify_sdist(path: Path, project: dict[str, object]) -> None:
    name = _project_value(project, "name").replace("-", "_")
    version = _project_value(project, "version")
    root = f"{name}-{version}"
    required_members = {
        f"{root}/LICENSE",
        f"{root}/README.md",
        f"{root}/pyproject.toml",
        f"{root}/src/ier/py.typed",
    }

    with tarfile.open(path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers()}
    missing = sorted(required_members - members)
    _require(not missing, f"{path.name} is missing required files: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    project = _project_metadata()
    wheel_count = 0
    sdist_count = 0

    for artifact in args.artifacts:
        if artifact.suffix == ".whl":
            _verify_wheel(artifact, project)
            wheel_count += 1
        elif artifact.name.endswith(".tar.gz"):
            _verify_sdist(artifact, project)
            sdist_count += 1
        else:
            raise ValueError(f"unsupported distribution artifact: {artifact}")
        print(f"verified {artifact}")

    _require(wheel_count > 0, "no wheel artifact was provided")
    _require(sdist_count > 0, "no source distribution artifact was provided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
