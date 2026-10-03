"""Fail the aggregate CI gate unless every prerequisite succeeded."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection


def validate_ci_results(results: object, *, allow_skipped: Collection[str] = ()) -> None:
    """Check GitHub's ``toJSON(needs)`` payload, including newly added jobs."""
    if not isinstance(results, dict) or not results:
        raise ValueError("CI results must contain at least one prerequisite job")

    failures = []
    for name, job in results.items():
        if not isinstance(name, str) or not name:
            raise ValueError("CI prerequisite names must be nonempty strings")
        if not isinstance(job, dict):
            raise ValueError(f"CI prerequisite {name!r} has no result object")
        result = job.get("result")
        if result not in ("success", "failure", "cancelled", "skipped"):
            raise ValueError(f"CI prerequisite {name!r} has an invalid result: {result!r}")
        if result != "success" and not (result == "skipped" and name in allow_skipped):
            failures.append(f"{name}={result}")

    if failures:
        raise ValueError(f"required CI jobs did not succeed: {', '.join(sorted(failures))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-skipped",
        action="append",
        default=[],
        metavar="JOB",
        help="Permit this prerequisite to be skipped; failure and cancellation still fail",
    )
    args = parser.parse_args(argv)
    try:
        validate_ci_results(json.load(sys.stdin), allow_skipped=args.allow_skipped)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("All required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
