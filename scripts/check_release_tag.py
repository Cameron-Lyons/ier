"""Require a release tag to match the project version exactly."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .check_version import parse_semver, read_project_version
else:
    from check_version import parse_semver, read_project_version


def validate_release_tag(tag: str, pyproject: Path) -> str:
    """Return the project version when ``tag`` is its exact ``v``-prefixed form."""
    version = read_project_version(pyproject)
    parse_semver(version)
    expected = f"v{version}"
    if tag != expected:
        raise ValueError(
            f"release tag {tag!r} must match project.version {version!r} exactly as {expected!r}"
        )
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, for example v2.1.2")
    parser.add_argument(
        "pyproject",
        nargs="?",
        type=Path,
        default=Path("pyproject.toml"),
    )
    args = parser.parse_args(argv)

    try:
        version = validate_release_tag(args.tag, args.pyproject)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"release tag {args.tag} matches project version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
