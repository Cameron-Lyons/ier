"""Verify metadata and required files in built wheel and source distributions."""

from __future__ import annotations

import argparse
import configparser
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from email.message import Message


class _EntryPointParser(configparser.ConfigParser):
    """Preserve the case of console-script names, as installers do."""

    def optionxform(self, optionstr: str) -> str:
        return optionstr


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


def _package_files() -> set[str]:
    source = Path(__file__).resolve().parents[1] / "src"
    return {path.relative_to(source).as_posix() for path in (source / "ier").rglob("*.py")} | {
        "ier/py.typed"
    }


def _verify_metadata(path: Path, metadata: Message, project: dict[str, object]) -> None:
    expected_headers = {
        "Name": _project_value(project, "name"),
        "Version": _project_value(project, "version"),
        "License-Expression": _project_value(project, "license"),
        "Requires-Python": _project_value(project, "requires-python"),
    }
    for header, expected in expected_headers.items():
        actual = metadata.get(header)
        _require(
            actual == expected,
            f"{path.name} has {header}={actual!r}; expected {expected!r}",
        )

    license_files = metadata.get_all("License-File", failobj=[])
    _require("LICENSE" in license_files, f"{path.name} does not declare LICENSE metadata")


def _verify_wheel(
    path: Path, project: dict[str, object], package_files: set[str] | None = None
) -> None:
    name = _project_value(project, "name")
    version = _project_value(project, "version")
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"
    if package_files is None:
        package_files = _package_files()

    with zipfile.ZipFile(path) as archive:
        members = {member.filename for member in archive.infolist() if not member.is_dir()}
        required_members = package_files | {
            f"{dist_info}/METADATA",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
        }
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")

        _verify_metadata(path, _wheel_metadata(archive, dist_info), project)
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        configuration = _EntryPointParser(interpolation=None)
        configuration.read_string(entry_points)
        _require(
            configuration.get("console_scripts", "ier", fallback=None) == "ier.cli:main",
            f"{path.name} is missing the ier CLI entry point",
        )


def _verify_sdist(
    path: Path, project: dict[str, object], package_files: set[str] | None = None
) -> None:
    name = _project_value(project, "name").replace("-", "_")
    version = _project_value(project, "version")
    root = f"{name}-{version}"
    if package_files is None:
        package_files = _package_files()
    required_members = {f"{root}/src/{member}" for member in package_files} | {
        f"{root}/LICENSE",
        f"{root}/PKG-INFO",
        f"{root}/README.md",
        f"{root}/pyproject.toml",
    }

    with tarfile.open(path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")
        metadata_file = archive.extractfile(f"{root}/PKG-INFO")
        if metadata_file is None:
            raise ValueError(f"{path.name} is missing PKG-INFO contents")
        with metadata_file:
            _verify_metadata(path, BytesParser().parsebytes(metadata_file.read()), project)

        project_file = archive.extractfile(f"{root}/pyproject.toml")
        if project_file is None:
            raise ValueError(f"{path.name} is missing pyproject.toml contents")
        with project_file:
            bundled_project = tomllib.loads(project_file.read().decode("utf-8")).get("project", {})
        if not isinstance(bundled_project, dict):
            raise ValueError(f"{path.name} bundled pyproject.toml has no [project] table")
        for key in ("name", "version", "license", "requires-python"):
            _require(
                bundled_project.get(key) == project.get(key),
                f"{path.name} bundled project.{key} does not match pyproject.toml",
            )


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
