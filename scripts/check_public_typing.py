"""Check valid consumer workflows and require deliberate type errors to be rejected."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "typing"
_EXPECTED = re.compile(r"# expect: ([a-z-]+)$")
_DIAGNOSTIC = re.compile(r"^(.+?):([0-9]+): error: .* \[([a-z-]+)\]$")


def _expected_errors(fixture: Path) -> set[tuple[int, str]]:
    """Read declared diagnostic positions from the deliberate-mistake fixture."""
    expected = set()
    source = fixture.read_text(encoding="utf-8")
    statements = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Return))
    ]
    for line_number, line in enumerate(source.splitlines(), 1):
        match = _EXPECTED.search(line)
        if match is not None:
            positions = [
                node.lineno
                for node in statements
                if node.lineno <= line_number <= (node.end_lineno or node.lineno)
            ]
            if not positions:
                raise ValueError(f"{fixture}:{line_number} has no statement for its expected error")
            expected.add((max(positions), match.group(1)))
    if not expected:
        raise ValueError(f"{fixture} has no expected type errors")
    return expected


def _verify_negative_result(fixture: Path, result: subprocess.CompletedProcess[str]) -> None:
    """Reject infrastructure failures, missing errors, and unrelated diagnostics."""
    expected = _expected_errors(fixture)
    actual = []
    for line in result.stdout.splitlines():
        match = _DIAGNOSTIC.fullmatch(line)
        if match is not None:
            reported = Path(match.group(1))
            if not reported.is_absolute():
                reported = _ROOT / reported
            if reported.resolve() != fixture.resolve():
                raise ValueError(f"unexpected diagnostic outside typing fixture: {line}")
            actual.append((int(match.group(2)), match.group(3)))
        elif ": error:" in line:
            raise ValueError(f"unrecognized typing diagnostic: {line}")
    if (
        result.returncode != 1
        or result.stderr
        or len(actual) != len(expected)
        or set(actual) != expected
    ):
        raise ValueError(
            "negative typing fixture did not produce exactly its expected errors: "
            f"expected {sorted(expected)}, received {actual}, exit={result.returncode}\n"
            f"{result.stdout}{result.stderr}"
        )


def _run_mypy(fixture: Path) -> subprocess.CompletedProcess[str]:
    """Use the active locked interpreter and the repository's strict configuration."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(_ROOT / "pyproject.toml"),
            "--no-pretty",
            "--no-error-summary",
            "--show-error-codes",
            "--hide-error-context",
            fixture.relative_to(_ROOT).as_posix(),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def main() -> int:
    """Validate public type inference and its rejection of incorrect consumers."""
    positive = _run_mypy(_FIXTURES / "composite_valid.py")
    if positive.returncode != 0 or positive.stderr:
        print(positive.stdout + positive.stderr, file=sys.stderr, end="")
        return 1
    negative_path = _FIXTURES / "composite_invalid.py"
    try:
        _verify_negative_result(negative_path, _run_mypy(negative_path))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print("Public consumer typing passed, including deliberate return-shape errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
