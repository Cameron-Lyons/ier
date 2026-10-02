"""Verify built distributions preserve the checkout's sources and project metadata."""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import tarfile
import tomllib
import zipfile
from collections import Counter
from email.parser import BytesParser
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import Version

if TYPE_CHECKING or __package__:
    from .check_version import normalized_distribution_version
else:
    from check_version import normalized_distribution_version

if TYPE_CHECKING:
    from email.message import Message

_WINDOWS_RESERVED_NAMES = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in "123456789¹²³"
}


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
        root / "pyproject.toml",
        root / "MANIFEST.in",
        root / "README.md",
        root / "LICENSE",
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
        actual = metadata.get_all(header, [])
        _require(
            actual == [expected],
            f"{path.name} has {header}={actual!r}; expected {expected!r}",
        )

    license_files = metadata.get_all("License-File", failobj=[])
    _require("LICENSE" in license_files, f"{path.name} does not declare LICENSE metadata")
    _verify_dependencies(path, metadata, project)


def _verify_archive_members(path: Path, names: list[str]) -> None:
    """Reject ambiguous names and paths that differ across archive extractors."""
    normalized = [
        "/".join(part.rstrip(" .").casefold() for part in name.removesuffix("/").split("/"))
        for name in names
    ]
    duplicates = sorted(name for name, count in Counter(normalized).items() if count > 1)
    _require(not duplicates, f"{path.name} has duplicate archive members: {duplicates}")
    invalid = [
        name
        for name in names
        if any(character in '<>:"\\|?*' or ord(character) < 32 for character in name)
        or any(
            part in {"", ".", ".."}
            or part.endswith((" ", "."))
            or part.split(".", 1)[0].rstrip(" ").casefold() in _WINDOWS_RESERVED_NAMES
            for part in name.removesuffix("/").split("/")
        )
    ]
    _require(not invalid, f"{path.name} has unsafe archive paths: {sorted(invalid)}")


def _verify_package_members(path: Path, members: set[str], package_files: dict[str, bytes]) -> None:
    """Catch deleted package modules accidentally retained by an old build tree."""
    package_roots = {name.split("/", 1)[0].casefold() for name in package_files}
    unexpected = sorted(
        name
        for name in members - package_files.keys()
        if name.split("/", 1)[0].casefold() in package_roots
        and (name.casefold().endswith((".py", ".pyc", ".so", ".pyd", "/py.typed")))
    )
    _require(not unexpected, f"{path.name} has unexpected package files: {unexpected}")


def _verify_record(path: Path, archive: zipfile.ZipFile, dist_info: str, members: set[str]) -> None:
    """Check the wheel's complete integrity manifest before publication."""
    record_name = f"{dist_info}/RECORD"
    signatures = {f"{record_name}.jws", f"{record_name}.p7s"}
    try:
        record = archive.read(record_name).decode("utf-8")
        rows = list(csv.reader(StringIO(record, newline=""), strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"{path.name} has invalid RECORD: {error}") from error
    entries: dict[str, tuple[str, str]] = {}
    for row in rows:
        _require(len(row) == 3, f"{path.name} RECORD rows must have three columns: {row!r}")
        name, digest, size = row
        _require(name not in entries, f"{path.name} RECORD has duplicate entries: {name!r}")
        entries[name] = (digest, size)
    expected = members - signatures
    _require(
        entries.keys() == expected,
        f"{path.name} RECORD does not match archive members: "
        f"missing {sorted(expected - entries.keys())}; "
        f"unexpected {sorted(entries.keys() - expected)}",
    )
    for name, (digest, size) in entries.items():
        if name == record_name:
            _require(
                not digest and not size,
                f"{path.name} RECORD cannot hash or size itself",
            )
            continue
        algorithm, separator, encoded_digest = digest.partition("=")
        _require(
            bool(separator and encoded_digest),
            f"{path.name} RECORD is missing a secure hash for {name}",
        )
        _require(
            algorithm in hashlib.algorithms_guaranteed,
            f"{path.name} RECORD has unsupported hash algorithm {algorithm!r} for {name}",
        )
        try:
            hasher = hashlib.new(algorithm)
        except ValueError as error:
            raise ValueError(
                f"{path.name} RECORD has unsupported hash algorithm {algorithm!r} for {name}"
            ) from error
        _require(
            hasher.digest_size >= 32,
            f"{path.name} RECORD requires sha256 or stronger hashes for {name}",
        )
        _require(
            not size or (size.isascii() and size.isdecimal()),
            f"{path.name} RECORD has invalid file size for {name}: {size!r}",
        )
        _require(
            not size or int(size) == archive.getinfo(name).file_size,
            f"{path.name} RECORD size does not match {name}",
        )
        with archive.open(name) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        expected_digest = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode("ascii")
        _require(
            encoded_digest == expected_digest,
            f"{path.name} RECORD hash does not match {name}",
        )


def _verify_wheel_format(
    path: Path, archive: zipfile.ZipFile, dist_info: str, project: dict[str, object]
) -> None:
    """Ensure installers see the same version and compatibility as the filename."""
    try:
        distribution, version, build, tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as error:
        raise ValueError(f"{path.name} has an invalid wheel filename: {error}") from error
    _require(
        distribution == canonicalize_name(_project_value(project, "name")),
        f"{path.name} wheel filename distribution does not match project.name",
    )
    _require(
        version == Version(normalized_distribution_version(_project_value(project, "version"))),
        f"{path.name} wheel filename version does not match project.version",
    )
    metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/WHEEL"))
    wheel_versions = metadata.get_all("Wheel-Version", [])
    _require(
        wheel_versions == ["1.0"],
        f"{path.name} has unsupported or ambiguous Wheel-Version={wheel_versions!r}; "
        "expected '1.0'",
    )
    purelib = metadata.get_all("Root-Is-Purelib", [])
    _require(
        purelib == ["true"],
        f"{path.name} has Root-Is-Purelib={purelib!r}; expected pure Python package metadata",
    )
    declared_tags = metadata.get_all("Tag", [])
    _require(
        Counter(declared_tags) == Counter(str(tag) for tag in tags),
        f"{path.name} WHEEL Tag={declared_tags!r} does not match filename tags "
        f"{sorted(str(tag) for tag in tags)!r}",
    )
    _require(
        all(tag.abi == "none" and tag.platform == "any" for tag in tags),
        f"{path.name} wheel filename tags must describe a platform-independent Python package",
    )
    declared_build = metadata.get_all("Build", [])
    expected_build = [path.stem.split("-")[-4]] if build else []
    _require(
        declared_build == expected_build,
        f"{path.name} WHEEL Build={declared_build!r} does not match filename build "
        f"{expected_build!r}",
    )


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
        # ZipInfo normalizes separators on Windows and truncates NUL suffixes.
        # Validate original names before those repairs can conceal unsafe paths.
        _verify_archive_members(path, [member.orig_filename for member in archive.infolist()])
        # Unicode path extras can independently replace the effective names.
        _verify_archive_members(path, [member.filename for member in archive.infolist()])
        members = {member.filename for member in archive.infolist() if not member.is_dir()}
        required_members = package_files.keys() | {
            f"{dist_info}/METADATA",
            f"{dist_info}/WHEEL",
            f"{dist_info}/RECORD",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
        }
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")
        _verify_package_members(path, members, package_files)

        for name, expected in package_files.items():
            _require(
                archive.read(name) == expected,
                f"{path.name} source file {name} does not match the checkout",
            )

        _verify_metadata(path, _wheel_metadata(archive, dist_info), project)
        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        configuration = _EntryPointParser(interpolation=None, delimiters=("=",))
        try:
            configuration.read_string(entry_points)
        except configparser.Error as error:
            raise ValueError(f"{path.name} has invalid CLI entry points: {error}") from error
        _require(
            not configuration.defaults(),
            f"{path.name} CLI entry points cannot use [DEFAULT] settings",
        )
        expected_scripts = project.get("scripts", {})
        if not isinstance(expected_scripts, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in expected_scripts.items()
        ):
            raise ValueError("pyproject.toml project.scripts must be a table of strings")
        actual_scripts = (
            dict(configuration.items("console_scripts"))
            if configuration.has_section("console_scripts")
            else {}
        )
        _require(
            actual_scripts == expected_scripts,
            f"{path.name} CLI entry points {actual_scripts!r} do not match "
            f"project.scripts {expected_scripts!r}",
        )
        _verify_record(path, archive, dist_info, members)
        _verify_wheel_format(path, archive, dist_info, project)


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
        _verify_archive_members(path, [member.name for member in archive.getmembers()])
        links = sorted(
            member.name for member in archive.getmembers() if member.issym() or member.islnk()
        )
        _require(not links, f"{path.name} contains archive links: {links}")
        members = {member.name for member in archive.getmembers() if member.isfile()}
        missing = sorted(required_members - members)
        _require(not missing, f"{path.name} is missing required files: {missing}")
        source_prefix = f"{root}/src/"
        package_members = {
            name[len(source_prefix) :]
            for name in members
            if name.casefold().startswith(source_prefix.casefold())
        }
        _verify_package_members(path, package_members, package_files)
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
