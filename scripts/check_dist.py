"""Verify built distributions preserve the checkout's sources and project metadata."""

from __future__ import annotations

import argparse
import configparser
import tarfile
import tomllib
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

if __package__:
    from .check_version import normalized_distribution_version
else:
    from check_version import normalized_distribution_version

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


def _package_files() -> dict[str, bytes]:
    source = Path(__file__).resolve().parents[1] / "src"
    paths = [*(source / "ier").rglob("*.py"), source / "ier" / "py.typed"]
    return {path.relative_to(source).as_posix(): path.read_bytes() for path in paths}


def _sdist_support_files() -> dict[str, bytes]:
    """Collect bundled tests and every local resource they import or execute."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        *(root / "tests").rglob("*.py"),
        *(root / "tests" / "fixtures").rglob("*"),
        *(root / "scripts").rglob("*.py"),
        *(root / "scripts").rglob("*.sh"),
        *(root / "benchmarks").rglob("*.py"),
        root / "uv.lock",
    ]
    return {
        path.relative_to(root).as_posix(): path.read_bytes() for path in paths if path.is_file()
    }


def _expected_requirements(project: dict[str, object]) -> tuple[Counter[Requirement], set[str]]:
    """Build PEP 508 requirements, preserving optional-marker boolean grouping."""
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list) or any(
        not isinstance(value, str) for value in dependencies
    ):
        raise ValueError("pyproject.toml project.dependencies must be an array of strings")
    requirements = Counter(Requirement(value) for value in dependencies)
    optional = project.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ValueError("pyproject.toml project.optional-dependencies must be a table")
    extras: set[str] = set()
    for name, values in optional.items():
        extra = canonicalize_name(name, validate=True)
        if extra in extras:
            raise ValueError(f"pyproject.toml contains duplicate normalized extra {extra!r}")
        extras.add(extra)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(
                f"pyproject.toml optional dependency {name!r} must be an array of strings"
            )
        for value in values:
            requirement = Requirement(value)
            condition = f'extra == "{extra}"'
            if requirement.marker is not None:
                condition = f"({requirement.marker}) and {condition}"
            requirement.marker = Marker(condition)
            requirements[requirement] += 1
    return requirements, extras


def _verify_dependencies(path: Path, metadata: Message, project: dict[str, object]) -> None:
    expected, extras = _expected_requirements(project)
    try:
        actual = Counter(Requirement(value) for value in metadata.get_all("Requires-Dist", []))
    except InvalidRequirement as error:
        raise ValueError(f"{path.name} has invalid Requires-Dist metadata: {error}") from error
    missing = sorted(str(value) for value in (expected - actual).elements())
    unexpected = sorted(str(value) for value in (actual - expected).elements())
    _require(
        not missing and not unexpected,
        f"{path.name} Requires-Dist does not match pyproject.toml: "
        f"missing {missing}; unexpected {unexpected}",
    )
    provided = metadata.get_all("Provides-Extra", [])
    _require(
        Counter(provided) == Counter(extras),
        f"{path.name} Provides-Extra={provided!r}; expected {sorted(extras)!r}",
    )


def _verify_metadata(path: Path, metadata: Message, project: dict[str, object]) -> None:
    expected_headers = {
        "Name": _project_value(project, "name"),
        "Version": normalized_distribution_version(_project_value(project, "version")),
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
    _verify_dependencies(path, metadata, project)


def _verify_wheel(
    path: Path, project: dict[str, object], package_files: dict[str, bytes] | None = None
) -> None:
    name = _project_value(project, "name")
    version = normalized_distribution_version(_project_value(project, "version"))
    distribution = name.replace("-", "_")
    dist_info = f"{distribution}-{version}.dist-info"
    if package_files is None:
        package_files = _package_files()

    with zipfile.ZipFile(path) as archive:
        members = {member.filename for member in archive.infolist() if not member.is_dir()}
        required_members = package_files.keys() | {
            f"{dist_info}/METADATA",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
        }
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")

        for name, expected in package_files.items():
            _require(
                archive.read(name) == expected,
                f"{path.name} source file {name} does not match the checkout",
            )

        _verify_metadata(path, _wheel_metadata(archive, dist_info), project)
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        configuration = _EntryPointParser(interpolation=None)
        configuration.read_string(entry_points)
        _require(
            configuration.get("console_scripts", "ier", fallback=None) == "ier.cli:main",
            f"{path.name} is missing the ier CLI entry point",
        )


def _verify_sdist(
    path: Path,
    project: dict[str, object],
    package_files: dict[str, bytes] | None = None,
    *,
    support_files: dict[str, bytes] | None = None,
) -> None:
    name = _project_value(project, "name").replace("-", "_")
    version = normalized_distribution_version(_project_value(project, "version"))
    root = f"{name}-{version}"
    if package_files is None:
        package_files = _package_files()
    if support_files is None:
        support_files = _sdist_support_files()
    source_files = {
        f"src/{name}": contents for name, contents in package_files.items()
    } | support_files
    required_members = {f"{root}/{member}" for member in source_files} | {
        f"{root}/LICENSE",
        f"{root}/PKG-INFO",
        f"{root}/README.md",
        f"{root}/pyproject.toml",
    }

    with tarfile.open(path, mode="r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")
        for name, expected in source_files.items():
            member_name = f"{root}/{name}"
            source_file = archive.extractfile(member_name)
            if source_file is None:
                raise ValueError(f"{path.name} is missing {member_name} contents")
            with source_file:
                _require(
                    source_file.read() == expected,
                    f"{path.name} source file {name} does not match the checkout",
                )
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
        for key in sorted(project.keys() | bundled_project.keys()):
            _require(
                bundled_project.get(key) == project.get(key),
                f"{path.name} bundled project.{key} does not match pyproject.toml",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    project = _project_metadata()
    package_files = _package_files()
    support_files = _sdist_support_files()
    wheel_count = 0
    sdist_count = 0

    for artifact in args.artifacts:
        if artifact.suffix == ".whl":
            _verify_wheel(artifact, project, package_files)
            wheel_count += 1
        elif artifact.name.endswith(".tar.gz"):
            _verify_sdist(artifact, project, package_files, support_files=support_files)
            sdist_count += 1
        else:
            raise ValueError(f"unsupported distribution artifact: {artifact}")
        print(f"verified {artifact}")

    _require(wheel_count > 0, "no wheel artifact was provided")
    _require(sdist_count > 0, "no source distribution artifact was provided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
