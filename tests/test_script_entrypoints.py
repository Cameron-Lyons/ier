"""Release validators support direct execution and package-module invocation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("mode", ["direct", "module"])
@pytest.mark.parametrize(
    "name", ["check_version", "check_lock_version", "check_release_tag", "check_dist"]
)
def test_release_validator_entrypoints_preserve_their_imports(
    tmp_path: Path, mode: str, name: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base.toml"
    project = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    base.write_text('[project]\nname = "example"\nversion = "1.2.3"\n', encoding="utf-8")
    project.write_text('[project]\nname = "example"\nversion = "1.2.4"\n', encoding="utf-8")
    lock.write_text(
        'version = 1\n[[package]]\nname = "example"\nversion = "1.2.4"\n'
        'source = { editable = "." }\n',
        encoding="utf-8",
    )
    commands = {
        "check_version": ([str(base), str(project)], "version increased from 1.2.3 to 1.2.4"),
        "check_lock_version": ([str(project), str(lock)], "editable project version matches 1.2.4"),
        "check_release_tag": (["v1.2.4", str(project)], "matches project version 1.2.4"),
        "check_dist": (["--help"], "Verify built distributions"),
    }
    arguments, expected = commands[name]
    invocation = (
        [str(root / "scripts" / f"{name}.py")] if mode == "direct" else ["-m", f"scripts.{name}"]
    )
    completed = subprocess.run(
        [sys.executable, *invocation, *arguments],
        cwd=tmp_path if mode == "direct" else root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected in completed.stdout
    assert not completed.stderr
