"""The required aggregate CI check must fail closed for every prerequisite."""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest
from scripts.check_ci_results import validate_ci_results

_REQUIRED = ("ci", "lint", "security", "docs")
_STATES = ("success", "failure", "cancelled", "skipped")


def test_every_combination_of_prerequisite_outcomes() -> None:
    # Exercise all 1,024 combinations, including skipped and cancelled reusable
    # workflows. Only the optional version comparison may be skipped.
    for states in product(_STATES, repeat=5):
        names = (*_REQUIRED, "version-check")
        results = {
            name: {"result": state, "outputs": {}}
            for name, state in zip(names, states, strict=True)
        }
        acceptable = all(state == "success" for state in states[:4]) and states[4] in (
            "success",
            "skipped",
        )
        if acceptable:
            validate_ci_results(results, allow_skipped=("version-check",))
        else:
            with pytest.raises(ValueError, match="required CI jobs did not succeed"):
                validate_ci_results(results, allow_skipped=("version-check",))


@pytest.mark.parametrize("state", ["failure", "cancelled", "skipped"])
def test_new_prerequisite_cannot_escape_the_aggregate_gate(state: str) -> None:
    results = {name: {"result": "success"} for name in (*_REQUIRED, "version-check")}
    results["new-quality-gate"] = {"result": state}

    with pytest.raises(ValueError, match=f"new-quality-gate={state}"):
        validate_ci_results(results, allow_skipped=("version-check",))


@pytest.mark.parametrize(
    "results",
    [None, [], {}, {"": {"result": "success"}}, {1: {"result": "success"}}, {"ci": None}],
)
def test_missing_or_malformed_prerequisites_fail_closed(results: object) -> None:
    with pytest.raises(ValueError):
        validate_ci_results(results)


@pytest.mark.parametrize("state", [None, "", "pending", [], {}, True, 0])
def test_unknown_results_fail_closed_even_for_optional_jobs(state: object) -> None:
    with pytest.raises(ValueError, match="invalid result"):
        validate_ci_results({"version-check": {"result": state}}, allow_skipped=("version-check",))


def test_skipping_requires_an_explicit_exception() -> None:
    with pytest.raises(ValueError, match="version-check=skipped"):
        validate_ci_results({"version-check": {"result": "skipped"}})


@pytest.mark.parametrize(
    ("payload", "arguments", "expected_exit", "expected_message"),
    [
        (json.dumps({"ci": {"result": "success"}}), [], 0, "All required checks passed"),
        (
            json.dumps({"version-check": {"result": "skipped"}}),
            ["--allow-skipped", "version-check"],
            0,
            "All required checks passed",
        ),
        (json.dumps({"ci": {"result": "cancelled"}}), [], 1, "ci=cancelled"),
        ("{", [], 1, "error:"),
        ("", [], 1, "error:"),
        ("{}", [], 1, "at least one prerequisite"),
    ],
)
def test_real_ci_entrypoint_reads_stdin_and_returns_a_blocking_exit_code(
    tmp_path: Path,
    payload: str,
    arguments: list[str],
    expected_exit: int,
    expected_message: str,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_ci_results.py"
    completed = subprocess.run(
        [sys.executable, str(script), *arguments],
        input=payload,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == expected_exit, completed.stderr
    assert expected_message in (completed.stdout if expected_exit == 0 else completed.stderr)
    if expected_exit:
        assert "All required checks passed" not in completed.stdout
