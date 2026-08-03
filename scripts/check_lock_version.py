"""Require the editable project version in uv.lock to match pyproject.toml."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

if __package__:
    from .check_version import parse_semver, read_project_version
else:
    from check_version import parse_semver, read_project_version


def _read_project_name(path: Path) -> str:
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{path} does not contain a [project] table")
    name = project.get("name")
    if not isinstance(name, str):
        raise ValueError(f"{path} does not contain a string project.name")
    return name


def validate_lock_version(pyproject: Path, lockfile: Path) -> str:
    """Return the shared version after validating the editable lock entry."""
    project_name = _read_project_name(pyproject)
    project_version = read_project_version(pyproject)
    parse_semver(project_version)

    with lockfile.open("rb") as handle:
        lock_document = tomllib.load(handle)
    packages = lock_document.get("package")
    if not isinstance(packages, list):
        raise ValueError(f"{lockfile} does not contain package entries")

    matches = []
    for package in packages:
        if not isinstance(package, dict) or package.get("name") != project_name:
            continue
        source = package.get("source")
        if isinstance(source, dict) and source.get("editable") == ".":
            matches.append(package)

    if len(matches) != 1:
        raise ValueError(
            f"{lockfile} must contain exactly one editable package entry for "
            f"{project_name!r}; found {len(matches)}"
        )

    lock_version = matches[0].get("version")
    if not isinstance(lock_version, str):
        raise ValueError(f"editable package {project_name!r} has no string version in {lockfile}")
    if lock_version != project_version:
        raise ValueError(
            f"editable package {project_name!r} version {lock_version!r} in {lockfile} "
            f"does not match project.version {project_version!r}"
        )
    return project_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", nargs="?", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("lockfile", nargs="?", type=Path, default=Path("uv.lock"))
    args = parser.parse_args(argv)

    try:
        version = validate_lock_version(args.pyproject, args.lockfile)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"uv.lock editable project version matches {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
