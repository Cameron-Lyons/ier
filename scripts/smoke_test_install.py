"""Smoke-test an installed wheel's import, metadata version, and CLI entry point."""

from __future__ import annotations

import shutil
import subprocess
from importlib.metadata import version

import ier


def main() -> int:
    """Verify that the installed distribution and console script are usable."""
    distribution_version = version("insufficient-effort")
    if ier.__version__ != distribution_version:
        raise RuntimeError(
            f"ier.__version__={ier.__version__!r} does not match installed "
            f"distribution version {distribution_version!r}"
        )

    executable = shutil.which("ier")
    if executable is None:
        raise RuntimeError("installed distribution is missing the ier console script")

    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = f"ier {distribution_version}"
    actual = completed.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"ier --version returned {actual!r}; expected {expected!r}")

    print(f"verified installed insufficient-effort {distribution_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
