"""Run public module docstring examples so documented outputs stay accurate.

Examples must print identically under NumPy 1.26 and 2.x: show arrays with
``.tolist()`` or ``np.round(values, 2).tolist()`` rather than NumPy reprs or bare
NumPy scalars, and reserve ``# doctest: +SKIP`` for plotting or undefined data.
"""

import contextlib
import doctest
import importlib
import io
import pkgutil
import sys
from pathlib import Path

import pytest

import ier

# Modules whose examples still disagree with their output. The gate also fails
# once a listed module passes, so each fix must remove its module from this set.
_KNOWN_FAILING: frozenset[str] = frozenset()
_PUBLIC_MODULES = sorted(
    module.name
    for module in pkgutil.iter_modules(ier.__path__)
    if not module.name.startswith("_") and module.name != "cli"
)


def test_known_failures_name_public_modules() -> None:
    assert _KNOWN_FAILING.issubset(_PUBLIC_MODULES)


@pytest.mark.parametrize("name", _PUBLIC_MODULES)
def test_docstring_examples(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    importlib.import_module(f"ier.{name}")
    # Package attributes such as ``ier.lz`` are the same-named functions.
    module = sys.modules[f"ier.{name}"]
    # Archive examples write their files into the working directory.
    monkeypatch.chdir(tmp_path)
    report = io.StringIO()
    with contextlib.ExitStack() as cleanup:
        if name == "visualize":
            # Plot examples render headlessly and must not leave figures open.
            pytest.importorskip("matplotlib").use("Agg")
            cleanup.callback(importlib.import_module("matplotlib.pyplot").close, "all")
        cleanup.enter_context(contextlib.redirect_stdout(report))
        result = doctest.testmod(
            module,
            optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS,
            verbose=False,
        )
    if name in _KNOWN_FAILING:
        assert result.failed > 0, f"ier.{name} examples pass; remove it from _KNOWN_FAILING"
    else:
        assert result.failed == 0, report.getvalue()
