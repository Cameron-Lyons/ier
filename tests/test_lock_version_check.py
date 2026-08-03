"""Tests for project and uv.lock version consistency."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.check_lock_version import main, validate_lock_version


class TestLockVersionCheck(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.pyproject = self.root / "pyproject.toml"
        self.lockfile = self.root / "uv.lock"
        self.pyproject.write_text(
            '[project]\nname = "example-package"\nversion = "2.4.0"\n',
            encoding="utf-8",
        )

    def _write_lock(self, version: str, *, editable: str = ".") -> None:
        self.lockfile.write_text(
            "version = 1\n"
            "[[package]]\n"
            'name = "example-package"\n'
            f'version = "{version}"\n'
            f'source = {{ editable = "{editable}" }}\n',
            encoding="utf-8",
        )

    def test_matching_version_passes(self) -> None:
        self._write_lock("2.4.0")

        self.assertEqual(validate_lock_version(self.pyproject, self.lockfile), "2.4.0")

        stdout = StringIO()
        with patch("sys.stdout", stdout):
            result = main([str(self.pyproject), str(self.lockfile)])
        self.assertEqual(result, 0)
        self.assertIn("matches 2.4.0", stdout.getvalue())

    def test_mismatched_version_fails(self) -> None:
        self._write_lock("2.3.0")
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            result = main([str(self.pyproject), str(self.lockfile)])

        self.assertEqual(result, 1)
        self.assertIn("does not match project.version '2.4.0'", stderr.getvalue())

    def test_missing_editable_entry_fails(self) -> None:
        self._write_lock("2.4.0", editable="../other")
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            result = main([str(self.pyproject), str(self.lockfile)])

        self.assertEqual(result, 1)
        self.assertIn("exactly one editable package entry", stderr.getvalue())

    def test_malformed_or_missing_files_fail_cleanly(self) -> None:
        cases = [
            (self.root / "missing.toml", self.lockfile),
            (self.pyproject, self.root / "missing.lock"),
        ]
        self._write_lock("2.4.0")
        for pyproject, lockfile in cases:
            with self.subTest(pyproject=pyproject, lockfile=lockfile):
                stderr = StringIO()
                with patch("sys.stderr", stderr):
                    result = main([str(pyproject), str(lockfile)])
                self.assertEqual(result, 1)
                self.assertIn("error:", stderr.getvalue())
