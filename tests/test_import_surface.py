"""`import ier` defers archive I/O and distribution metadata until they are used."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import pytest

import ier
import ier.archive
from ier.cli import main

_ARCHIVE_EXPORTS = [
    "load_response_time_archive",
    "load_score_archive",
    "load_screen_archive",
    "save_response_time_archive",
    "save_score_archive",
    "save_screen_archive",
]


def _run_python(*arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr
    return completed.stdout


def test_import_loads_neither_archive_io_nor_distribution_metadata() -> None:
    probe = """
import json
import sys

import numpy

# NumPy 1.26 imports numpy.random itself; NumPy 2 defers it until first use.
watched = ("zipfile", "importlib.metadata", "ier.archive", "numpy.random")
preloaded = {name for name in watched if name in sys.modules}
import ier

library = sorted(name for name in watched if name in sys.modules and name not in preloaded)
import ier.cli

cli = "importlib.metadata" in sys.modules and "importlib.metadata" not in preloaded
version = ier.__version__
print(json.dumps({"library": library, "cli_metadata": cli, "version": version}))
"""
    loaded = json.loads(_run_python("-c", probe))

    assert loaded == {
        "library": [],
        "cli_metadata": False,
        "version": version("insufficient-effort"),
    }


def test_lazy_exports_resolve_in_a_fresh_interpreter() -> None:
    probe = """
import json

import ier

before = sorted(name for name in ier.__all__ if name not in vars(ier))
first = ier.archive
exports = [getattr(ier, name) is getattr(first, name) for name in before if name != "__version__"]
print(json.dumps({"deferred": before, "identical": all(exports), "listed": sorted(dir(ier))}))
"""
    resolved = json.loads(_run_python("-c", probe))

    assert resolved["deferred"] == ["__version__", *_ARCHIVE_EXPORTS]
    assert resolved["identical"]
    assert set(ier.__all__) <= set(resolved["listed"])


def test_star_import_binds_every_public_name() -> None:
    probe = """
from ier import *
import ier
print(sorted(name for name in ier.__all__ if name not in globals()))
"""
    assert _run_python("-c", probe) == "[]\n"


@pytest.mark.parametrize("name", _ARCHIVE_EXPORTS)
def test_archive_exports_resolve_once_to_archive_functions(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delitem(vars(ier), name, raising=False)

    value = getattr(ier, name)

    assert value is getattr(ier.archive, name)
    assert vars(ier)[name] is value


def test_archive_submodule_attribute_remains_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(vars(ier), "archive", raising=False)

    assert ier.archive is sys.modules["ier.archive"]
    assert vars(ier)["archive"] is ier.archive


def test_version_reads_distribution_metadata_once(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = version("insufficient-effort")
    monkeypatch.delitem(vars(ier), "__version__", raising=False)
    calls: list[str] = []

    def recorded_version(distribution: str) -> str:
        calls.append(distribution)
        return expected

    monkeypatch.setattr("importlib.metadata.version", recorded_version)

    assert ier.__version__ == expected
    assert ier.__version__ == expected
    assert calls == ["insufficient-effort"]


def test_version_falls_back_without_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(ier.__version__, str)
    monkeypatch.delitem(vars(ier), "__version__")

    def missing_version(distribution: str) -> str:
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr("importlib.metadata.version", missing_version)

    assert ier.__version__ == "0.0.0"


def test_dir_lists_deferred_public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ["__version__", "archive", *_ARCHIVE_EXPORTS]:
        monkeypatch.delitem(vars(ier), name, raising=False)

    listed = dir(ier)

    assert listed == sorted(listed)
    assert set(ier.__all__) <= set(listed)
    assert "archive" in listed


def test_dir_lists_lazy_archive_submodule_in_a_fresh_interpreter() -> None:
    probe = """
import json
import sys

import ier

listed = dir(ier)
print(json.dumps({
    "listed": "archive" in listed,
    "sorted": listed == sorted(listed),
    "imported": "ier.archive" in sys.modules,
    "resolves": ier.archive is sys.modules.get("ier.archive"),
}))
"""
    observed = json.loads(_run_python("-c", probe))

    # Listing the deferred submodule must not import it; access still does.
    assert observed == {"listed": True, "sorted": True, "imported": False, "resolves": True}


def test_unknown_attributes_still_raise_attribute_error() -> None:
    with pytest.raises(AttributeError, match="module 'ier' has no attribute 'not_an_index'"):
        _ = ier.not_an_index


def test_version_option_prints_the_distribution_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--version"])

    assert raised.value.code == 0
    assert capsys.readouterr() == (f"ier {version('insufficient-effort')}\n", "")


def test_version_option_output_and_help_are_unchanged() -> None:
    expected = version("insufficient-effort")

    assert _run_python("-m", "ier.cli", "--version") == f"ier {expected}\n"
    help_text = _run_python("-m", "ier.cli", "--help")
    assert re.search(r"^  --version +show program's version number and exit$", help_text, re.M)
