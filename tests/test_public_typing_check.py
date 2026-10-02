"""A negative typing fixture cannot pass because of an unrelated checker failure."""

import subprocess
from pathlib import Path

import pytest
from scripts.check_public_typing import _expected_errors, _verify_negative_result


@pytest.fixture
def invalid_consumer(tmp_path: Path) -> Path:
    fixture = tmp_path / "consumer.py"
    fixture.write_text(
        "value: str = 3  # expect: assignment\nother: str = (\n    4\n)  # expect: assignment\n",
        encoding="utf-8",
    )
    return fixture


def _diagnostic(fixture: Path, line: int, code: str = "assignment") -> str:
    return f"{fixture}:{line}: error: Incorrect consumer return type  [{code}]\n"


def test_negative_gate_accepts_only_declared_errors_and_statement_locations(
    invalid_consumer: Path,
) -> None:
    assert _expected_errors(invalid_consumer) == {(1, "assignment"), (2, "assignment")}
    result = subprocess.CompletedProcess(
        [], 1, _diagnostic(invalid_consumer, 1) + _diagnostic(invalid_consumer, 2), ""
    )
    _verify_negative_result(invalid_consumer, result)


@pytest.mark.parametrize(
    "failure",
    [
        "success",
        "crash",
        "stderr",
        "missing",
        "duplicate",
        "wrong-code",
        "wrong-line",
        "other-file",
        "malformed",
        "additional-error",
    ],
)
def test_negative_gate_rejects_false_success_and_unrelated_failures(
    invalid_consumer: Path, failure: str
) -> None:
    output = _diagnostic(invalid_consumer, 1) + _diagnostic(invalid_consumer, 2)
    returncode, stderr = 1, ""
    if failure == "success":
        returncode = 0
    elif failure == "crash":
        returncode = 2
    elif failure == "stderr":
        stderr = "checker failed to load plugin"
    elif failure == "missing":
        output = _diagnostic(invalid_consumer, 1)
    elif failure == "duplicate":
        output += _diagnostic(invalid_consumer, 2)
    elif failure == "wrong-code":
        output = _diagnostic(invalid_consumer, 1, "import-not-found") + _diagnostic(
            invalid_consumer, 2
        )
    elif failure == "wrong-line":
        output = _diagnostic(invalid_consumer, 1) + _diagnostic(invalid_consumer, 4)
    elif failure == "other-file":
        output += _diagnostic(invalid_consumer.parent / "unrelated.py", 1)
    elif failure == "malformed":
        output += f"{invalid_consumer}:5: error: No diagnostic code\n"
    else:
        output += _diagnostic(invalid_consumer, 5)
    result = subprocess.CompletedProcess([], returncode, output, stderr)
    with pytest.raises(ValueError):
        _verify_negative_result(invalid_consumer, result)


@pytest.mark.parametrize("content", ["value = 3\n", "# expect: assignment\nvalue = 3\n"])
def test_negative_gate_requires_expected_errors_attached_to_statements(
    tmp_path: Path, content: str
) -> None:
    fixture = tmp_path / "missing-expectations.py"
    fixture.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="no (expected type errors|statement)"):
        _expected_errors(fixture)
