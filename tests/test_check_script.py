"""Integration tests for the unified local quality script."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestCheckScript(unittest.TestCase):
    def _run_checks(
        self,
        *,
        failed_command: str = "",
        skip_docs: bool = False,
        lock_version: str = "1.2.3",
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("requires Bash")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scripts = root / "scripts"
            scripts.mkdir()
            source = Path(__file__).resolve().parents[1] / "scripts"
            for name in ("check.sh", "check_lock_version.py", "check_version.py"):
                shutil.copyfile(source / name, scripts / name)
            (root / "pyproject.toml").write_text(
                '[project]\nname = "example"\nversion = "1.2.3"\n', encoding="utf-8"
            )
            (root / "uv.lock").write_text(
                'version = 1\n[[package]]\nname = "example"\n'
                f'version = "{lock_version}"\nsource = {{ editable = "." }}\n',
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "uv.log"
            fake_uv = fake_bin / "uv"
            fake_uv.write_text(
                '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$IER_TEST_UV_LOG"\n'
                'if [ "$*" = "$IER_TEST_FAILED_COMMAND" ]; then exit 17; fi\n',
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
            environment["IER_TEST_UV_LOG"] = log.as_posix()
            environment["IER_TEST_FAILED_COMMAND"] = failed_command
            environment["SKIP_DOCS"] = "1" if skip_docs else "0"

            result = subprocess.run(
                [bash, (scripts / "check.sh").as_posix()],
                cwd=root / "bin",
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            commands = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
            return result, commands

    def test_uv_bootstraps_locked_quality_groups_before_checks(self) -> None:
        result, commands = self._run_checks()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands[0], "sync --locked --all-groups")
        self.assertTrue(all(command.startswith("run --no-sync ") for command in commands[1:]))
        self.assertIn(
            "run --no-sync pytest tests/ -v --cov=ier --cov-report=term-missing",
            commands,
        )
        self.assertIn("run --no-sync mypy src/ier benchmarks", commands)
        self.assertIn("run --no-sync mkdocs build --strict", commands)

    def test_failed_sync_or_quality_gate_stops_later_checks(self) -> None:
        for command in (
            "sync --locked --all-groups",
            "run --no-sync pytest tests/ -v --cov=ier --cov-report=term-missing",
            "run --no-sync ruff check .",
            "run --no-sync ruff format --check .",
            "run --no-sync mypy src/ier benchmarks",
            "run --no-sync mkdocs build --strict",
        ):
            with self.subTest(command=command):
                result, commands = self._run_checks(failed_command=command)

                self.assertEqual(result.returncode, 17, result.stderr)
                self.assertEqual(commands[-1], command)
                self.assertNotIn("All checks passed", result.stdout)

    def test_lock_drift_stops_before_environment_mutation(self) -> None:
        result, commands = self._run_checks(lock_version="1.2.2")

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("does not match project.version", result.stderr)
        self.assertEqual(commands, [])

    def test_skip_docs_still_runs_every_other_quality_gate(self) -> None:
        result, commands = self._run_checks(skip_docs=True)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 5)
        self.assertEqual(commands[-1], "run --no-sync mypy src/ier benchmarks")
        self.assertNotIn("run --no-sync mkdocs build --strict", commands)
