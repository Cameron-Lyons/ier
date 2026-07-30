"""Tests for optional dependency helpers."""

from __future__ import annotations

import builtins
import sys
import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from ier._optional_imports import require_matplotlib_pyplot


@contextmanager
def _block_imports(*blocked: str):
    """Raise ImportError only for the named top-level packages."""
    real_import = builtins.__import__
    blocked_roots = set(blocked)

    def _import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        root = name.split(".", 1)[0]
        if root in blocked_roots:
            raise ImportError(f"no {root}")
        return real_import(name, globals, locals, fromlist, level)

    removed = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".", 1)[0] in blocked_roots
    }
    try:
        with patch("builtins.__import__", side_effect=_import):
            yield
    finally:
        sys.modules.update(removed)


class TestOptionalImports(unittest.TestCase):
    def test_require_matplotlib_raises_when_missing(self) -> None:
        with _block_imports("matplotlib"), self.assertRaises(RuntimeError) as ctx:
            require_matplotlib_pyplot()
        self.assertIn("insufficient-effort[plot]", str(ctx.exception))
