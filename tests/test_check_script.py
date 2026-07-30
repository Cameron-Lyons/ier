"""Integration tests for the unified local quality script."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestCheckScript(unittest.TestCase):
    def test_uv_bootstraps_locked_quality_groups_before_checks(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("requires Bash")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "uv.log"
            fake_uv = fake_bin / "uv"
            fake_uv.write_text(
                '#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$IER_TEST_UV_LOG"\n',
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
            environment["IER_TEST_UV_LOG"] = log.as_posix()

            result = subprocess.run(
                [bash, "./scripts/check.sh"],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            commands = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(commands[0], "sync --locked --all-groups")
            self.assertTrue(all(command.startswith("run --no-sync ") for command in commands[1:]))
            self.assertIn(
                "run --no-sync pytest tests/ -v --cov=ier --cov-report=term-missing",
                commands,
            )
            self.assertIn("run --no-sync mkdocs build --strict", commands)
