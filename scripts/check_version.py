#!/usr/bin/env python3
"""Require a candidate pyproject version to be valid SemVer and newer than a base."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

Identifier: TypeAlias = int | str

_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@dataclass(frozen=True)
class SemVer:
    """A parsed semantic version with precedence comparison."""

    major: int
    minor: int
    patch: int
    prerelease: tuple[Identifier, ...] | None = None

    def __lt__(self, other: SemVer) -> bool:
        own_core = (self.major, self.minor, self.patch)
        other_core = (other.major, other.minor, other.patch)
        if own_core != other_core:
            return own_core < other_core
        return _prerelease_is_lower(self.prerelease, other.prerelease)


def _prerelease_is_lower(
    left: tuple[Identifier, ...] | None,
    right: tuple[Identifier, ...] | None,
) -> bool:
    if left is None:
        return False
    if right is None:
        return True

    for left_part, right_part in zip(left, right, strict=False):
        if left_part == right_part:
            continue
        if isinstance(left_part, int) and isinstance(right_part, str):
            return True
        if isinstance(left_part, str) and isinstance(right_part, int):
            return False
        return left_part < right_part
    return len(left) < len(right)


def parse_semver(value: str) -> SemVer:
    """Parse a strict Semantic Versioning 2.0.0 value."""
    match = _SEMVER_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid semantic version: {value!r}")

    prerelease_text = match.group(4)
    prerelease: tuple[Identifier, ...] | None = None
    if prerelease_text is not None:
        identifiers: list[Identifier] = []
        for part in prerelease_text.split("."):
            if part.isdigit():
                if len(part) > 1 and part.startswith("0"):
                    raise ValueError(
                        f"invalid semantic version {value!r}: numeric prerelease identifiers "
                        "cannot contain leading zeroes"
                    )
                identifiers.append(int(part))
            else:
                identifiers.append(part)
        prerelease = tuple(identifiers)

    return SemVer(
        major=int(match.group(1)),
        minor=int(match.group(2)),
        patch=int(match.group(3)),
        prerelease=prerelease,
    )


def read_project_version(path: Path) -> str:
    """Read project.version from a pyproject.toml file."""
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{path} does not contain a [project] table")
    version = project.get("version")
    if not isinstance(version, str):
        raise ValueError(f"{path} does not contain a string project.version")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_pyproject", type=Path)
    parser.add_argument("candidate_pyproject", type=Path)
    args = parser.parse_args(argv)

    try:
        base_text = read_project_version(args.base_pyproject)
        candidate_text = read_project_version(args.candidate_pyproject)
        base = parse_semver(base_text)
        candidate = parse_semver(candidate_text)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if not base < candidate:
        print(
            f"error: candidate version {candidate_text} must be greater than "
            f"base version {base_text}",
            file=sys.stderr,
        )
        return 1

    print(f"version increased from {base_text} to {candidate_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
