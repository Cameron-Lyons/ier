"""Tests for the release version ordering gate."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from scripts.check_version import main, parse_semver


class TestSemVer(unittest.TestCase):
    def test_core_version_precedence(self) -> None:
        self.assertLess(parse_semver("2.1.0"), parse_semver("2.1.1"))
        self.assertLess(parse_semver("2.1.9"), parse_semver("2.2.0"))
        self.assertLess(parse_semver("2.9.9"), parse_semver("3.0.0"))

    def test_prerelease_precedence(self) -> None:
        ordered = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-beta.11",
            "1.0.0-rc.1",
            "1.0.0",
        ]
        parsed = [parse_semver(value) for value in ordered]
        self.assertTrue(all(left < right for left, right in pairwise(parsed)))

    def test_build_metadata_does_not_change_precedence(self) -> None:
        self.assertFalse(parse_semver("1.0.0+build.1") < parse_semver("1.0.0+build.2"))
        self.assertFalse(parse_semver("1.0.0+build.2") < parse_semver("1.0.0+build.1"))

    def test_invalid_versions_raise(self) -> None:
        invalid = ["1", "1.0", "v1.0.0", "01.0.0", "1.0.0-alpha.01", "1.0.0-"]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_semver(value)


class TestVersionGate(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)

    def _pyproject(self, name: str, version: str) -> Path:
        path = self.root / name
        path.write_text(f'[project]\nname = "example"\nversion = "{version}"\n', encoding="utf-8")
        return path

    def test_newer_version_passes(self) -> None:
        base = self._pyproject("base.toml", "2.1.0")
        candidate = self._pyproject("candidate.toml", "2.2.0-rc.1")
        stdout = StringIO()

        with patch("sys.stdout", stdout):
            result = main([str(base), str(candidate)])

        self.assertEqual(result, 0)
        self.assertIn("version increased from 2.1.0 to 2.2.0-rc.1", stdout.getvalue())

    def test_equal_or_older_version_fails(self) -> None:
        for candidate_version in ["2.1.0", "2.0.9"]:
            with self.subTest(candidate_version=candidate_version):
                base = self._pyproject("base.toml", "2.1.0")
                candidate = self._pyproject("candidate.toml", candidate_version)
                stderr = StringIO()

                with patch("sys.stderr", stderr):
                    result = main([str(base), str(candidate)])

                self.assertEqual(result, 1)
                self.assertIn("must be greater than base version", stderr.getvalue())

    def test_invalid_candidate_fails(self) -> None:
        base = self._pyproject("base.toml", "2.1.0")
        candidate = self._pyproject("candidate.toml", "next")
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            result = main([str(base), str(candidate)])

        self.assertEqual(result, 1)
        self.assertIn("invalid semantic version", stderr.getvalue())
