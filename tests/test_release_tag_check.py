"""Tests for the release tag and project version gate."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.check_release_tag import main


class TestReleaseTagGate(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

    def _pyproject(self, version: str) -> Path:
        path = self.root / "pyproject.toml"
        path.write_text(f'[project]\nname = "example"\nversion = "{version}"\n', encoding="utf-8")
        return path

    def test_matching_stable_and_prerelease_tags_pass(self) -> None:
        for version in ["2.1.2", "2.2.0-rc.1"]:
            with self.subTest(version=version):
                stdout = StringIO()

                with patch("sys.stdout", stdout):
                    result = main([f"v{version}", str(self._pyproject(version))])

                self.assertEqual(result, 0)
                self.assertIn(f"matches project version {version}", stdout.getvalue())

    def test_mismatched_or_unprefixed_tag_fails(self) -> None:
        pyproject = self._pyproject("2.1.2")
        for tag in ["v2.1.3", "2.1.2"]:
            with self.subTest(tag=tag):
                stderr = StringIO()

                with patch("sys.stderr", stderr):
                    result = main([tag, str(pyproject)])

                self.assertEqual(result, 1)
                self.assertIn("must match project.version", stderr.getvalue())
                self.assertIn("'v2.1.2'", stderr.getvalue())

    def test_invalid_project_version_fails(self) -> None:
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            result = main(["vnext", str(self._pyproject("next"))])

        self.assertEqual(result, 1)
        self.assertIn("invalid semantic version", stderr.getvalue())

    def test_missing_pyproject_fails(self) -> None:
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            result = main(["v2.1.2", str(self.root / "missing.toml")])

        self.assertEqual(result, 1)
        self.assertIn("error:", stderr.getvalue())
