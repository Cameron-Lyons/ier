"""Tests for the release version ordering gate."""

from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from scripts.check_version import main, normalized_distribution_version, parse_semver


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
        invalid = [
            "1",
            "1.0",
            "v1.0.0",
            "01.0.0",
            "1.0.0-alpha.01",
            "1.0.0-",
            "1\u0660.0.0",
            "1.1\u0661.0",
            "1.0.1\u0662",
        ]
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

    def test_unbuildable_semver_candidate_fails(self) -> None:
        base = self._pyproject("base.toml", "2.1.0")
        candidate = self._pyproject("candidate.toml", "2.2.0-canary.1")
        stderr = StringIO()

        with patch("sys.stderr", stderr):
            result = main([str(base), str(candidate)])

        self.assertEqual(result, 1)
        self.assertIn("not a supported Python prerelease", stderr.getvalue())


class TestDistributionVersions(unittest.TestCase):
    def test_python_prerelease_and_local_version_normalization(self) -> None:
        versions = [
            ("1.2.3", "1.2.3"),
            ("1.2.3-alpha.2", "1.2.3a2"),
            ("1.2.3-beta.3", "1.2.3b3"),
            ("1.2.3-rc.1", "1.2.3rc1"),
            ("1.2.3-dev.6", "1.2.3.dev6"),
            ("1.2.3+Build-Tag.007", "1.2.3+build.tag.7"),
            ("1.2.3-rc.1+Build.007", "1.2.3rc1+build.7"),
            ("1.2.3-alpha", "1.2.3a0"),
            ("1.2.3-a.1", "1.2.3a1"),
            ("1.2.3-b.1", "1.2.3b1"),
            ("1.2.3-preview.1", "1.2.3rc1"),
            ("1.2.3-pre.1", "1.2.3rc1"),
            ("1.2.3-c.1", "1.2.3rc1"),
            ("1.2.3-RC01", "1.2.3rc1"),
            ("1.2.3-rc.1.dev.2", "1.2.3rc1.dev2"),
            ("1.2.3-dev", "1.2.3.dev0"),
        ]
        for source, expected in versions:
            with self.subTest(source=source):
                self.assertEqual(normalized_distribution_version(source), expected)

    def test_unbuildable_or_precedence_reversing_semver_is_rejected(self) -> None:
        versions = [
            "1.2.3-canary.1",
            "1.2.3-alpha.beta",
            "1.2.3-1",
            "1.2.3-post.1",
            "1.2.3+build--tag",
            "1.2.3+build-",
            "1.2.3+-build",
        ]
        for source in versions:
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "unsupported|supported"),
            ):
                normalized_distribution_version(source)
