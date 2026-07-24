"""Tests for optional dependency helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ier._optional_imports import require_matplotlib_pyplot, require_scipy, scipy_install_hint


class TestOptionalImports(unittest.TestCase):
    def test_scipy_install_hint_mentions_full_extra(self) -> None:
        hint = scipy_install_hint("demo feature")
        self.assertIn("demo feature", hint)
        self.assertIn("insufficient-effort[full]", hint)

    def test_require_scipy_raises_when_missing(self) -> None:
        with (
            patch.dict("sys.modules", {"scipy": None}),
            patch("builtins.__import__", side_effect=ImportError("no scipy")),
            self.assertRaises(RuntimeError) as ctx,
        ):
            require_scipy("unit test")
        self.assertIn("insufficient-effort[full]", str(ctx.exception))

    def test_require_matplotlib_raises_when_missing(self) -> None:
        with (
            patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}),
            patch("builtins.__import__", side_effect=ImportError("no matplotlib")),
            self.assertRaises(RuntimeError) as ctx,
        ):
            require_matplotlib_pyplot()
        self.assertIn("insufficient-effort[plot]", str(ctx.exception))
