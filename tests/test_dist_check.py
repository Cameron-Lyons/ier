"""Reject release artifacts that are incomplete or contradict the project metadata."""

from __future__ import annotations

import tarfile
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from scripts.check_dist import _verify_metadata, _verify_wheel
from scripts.check_dist import _verify_sdist as _verify_sdist_with_support

if TYPE_CHECKING:
    from pathlib import Path

PROJECT: dict[str, object] = {
    "name": "insufficient-effort",
    "version": "1.2.3",
    "license": "MIT",
    "requires-python": ">=3.11",
    "dependencies": ["numpy>=1.26.0,<2.5"],
    "scripts": {"ier": "ier.cli:main"},
}
PACKAGE_FILES = {
    "ier/__init__.py": b'__version__ = "1.2.3"\n',
    "ier/cli.py": b"def main():\n    return 0\n",
    "ier/_cli_input.py": b"# Input parsing\n",
    "ier/py.typed": b"",
}
DIST_INFO = "insufficient_effort-1.2.3.dist-info"
SDIST_ROOT = "insufficient_effort-1.2.3"


def _verify_sdist(path: Path, project: dict[str, object], package_files: dict[str, bytes]) -> None:
    _verify_sdist_with_support(path, project, package_files, support_files={})


def _metadata(**overrides: str) -> bytes:
    headers = {
        "Metadata-Version": "2.4",
        "Name": "insufficient-effort",
        "Version": "1.2.3",
        "License-Expression": "MIT",
        "Requires-Python": ">=3.11",
        "License-File": "LICENSE",
        "Requires-Dist": "numpy<2.5,>=1.26.0",
    }
    headers.update(overrides)
    return ("\n".join(f"{name}: {value}" for name, value in headers.items()) + "\n\n").encode()


def _wheel_members() -> dict[str, bytes]:
    return {
        **PACKAGE_FILES,
        f"{DIST_INFO}/METADATA": _metadata(),
        f"{DIST_INFO}/entry_points.txt": b"[console_scripts]\nier = ier.cli:main\n",
        f"{DIST_INFO}/licenses/LICENSE": b"MIT license",
    }


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def _sdist_members() -> dict[str, bytes]:
    return {
        **{f"{SDIST_ROOT}/src/{name}": contents for name, contents in PACKAGE_FILES.items()},
        f"{SDIST_ROOT}/LICENSE": b"MIT license",
        f"{SDIST_ROOT}/README.md": b"IER readme",
        f"{SDIST_ROOT}/PKG-INFO": _metadata(),
        f"{SDIST_ROOT}/pyproject.toml": (
            b'[project]\nname = "insufficient-effort"\nversion = "1.2.3"\n'
            b'license = "MIT"\nrequires-python = ">=3.11"\n'
            b'dependencies = ["numpy>=1.26.0,<2.5"]\n'
            b'[project.scripts]\nier = "ier.cli:main"\n'
        ),
    }


def _write_sdist(path: Path, members: dict[str, bytes], *, directory: str | None = None) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, contents in members.items():
            member = tarfile.TarInfo(name)
            if name == directory:
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            else:
                member.size = len(contents)
                archive.addfile(member, BytesIO(contents))


def test_complete_distributions_pass(tmp_path: Path) -> None:
    wheel = tmp_path / "example.whl"
    sdist = tmp_path / "example.tar.gz"
    _write_wheel(wheel, _wheel_members())
    _write_sdist(sdist, _sdist_members())

    _verify_wheel(wheel, PROJECT, PACKAGE_FILES)
    _verify_sdist(sdist, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("missing", sorted(_wheel_members()))
def test_wheel_rejects_missing_source_and_metadata(tmp_path: Path, missing: str) -> None:
    path = tmp_path / "incomplete.whl"
    members = _wheel_members()
    del members[missing]
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="missing required files"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("missing", sorted(_sdist_members()))
def test_sdist_rejects_missing_source_and_metadata(tmp_path: Path, missing: str) -> None:
    path = tmp_path / "incomplete.tar.gz"
    members = _sdist_members()
    del members[missing]
    _write_sdist(path, members)

    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "header,value",
    [
        ("Name", "another-package"),
        ("Version", "1.2.2"),
        ("License-Expression", "Apache-2.0"),
        ("Requires-Python", ">=3.14"),
        ("License-File", "COPYING"),
    ],
)
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_incorrect_metadata(
    tmp_path: Path, header: str, value: str, kind: str
) -> None:
    if kind == "wheel":
        path = tmp_path / "incorrect.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = _metadata(**{header: value})
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "incorrect.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = _metadata(**{header: value})
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="expected|does not declare LICENSE"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "entry_points",
    [
        "[console_scripts]\n# ier = ier.cli:main\n",
        "[unrelated]\nier = ier.cli:main\n",
        "[console_scripts]\nier = ier.cli:missing\n",
        "[console_scripts]\nIER = ier.cli:main\n",
    ],
)
def test_wheel_requires_real_console_entry_point(tmp_path: Path, entry_points: str) -> None:
    path = tmp_path / "no-cli.whl"
    members = _wheel_members()
    members[f"{DIST_INFO}/entry_points.txt"] = entry_points.encode()
    _write_wheel(path, members)

    with pytest.raises(ValueError, match="CLI entry point"):
        _verify_wheel(path, PROJECT, PACKAGE_FILES)


def test_sdist_rejects_bundled_project_version_drift(tmp_path: Path) -> None:
    path = tmp_path / "stale-project.tar.gz"
    members = _sdist_members()
    name = f"{SDIST_ROOT}/pyproject.toml"
    members[name] = members[name].replace(b'"1.2.3"', b'"1.2.2"')
    _write_sdist(path, members)

    with pytest.raises(ValueError, match="bundled project.version"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


def test_sdist_does_not_accept_directory_as_required_file(tmp_path: Path) -> None:
    path = tmp_path / "bad-source.tar.gz"
    _write_sdist(path, _sdist_members(), directory=f"{SDIST_ROOT}/src/ier/_cli_input.py")

    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("name", PACKAGE_FILES)
@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_stale_source_contents(tmp_path: Path, name: str, kind: str) -> None:
    if kind == "wheel":
        path = tmp_path / "stale-source.whl"
        members = _wheel_members()
        members[name] = b"# stale artifact from a different checkout\n"
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "stale-source.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/src/{name}"] = b"# stale artifact from a different checkout\n"
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="source file .* does not match the checkout"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize(
    "before,after,key",
    [
        (b"numpy>=1.26.0,<2.5", b"numpy>=1.26.0", "dependencies"),
        (b'dependencies = ["numpy>=1.26.0,<2.5"]\n', b"", "dependencies"),
        (b'ier = "ier.cli:main"', b'ier = "ier.cli:missing"', "scripts"),
        (b"[project.scripts]", b'dynamic = ["description"]\n[project.scripts]', "dynamic"),
    ],
)
def test_sdist_rejects_bundled_dependency_and_entry_point_drift(
    tmp_path: Path, before: bytes, after: bytes, key: str
) -> None:
    path = tmp_path / "stale-configuration.tar.gz"
    members = _sdist_members()
    name = f"{SDIST_ROOT}/pyproject.toml"
    members[name] = members[name].replace(before, after)
    _write_sdist(path, members)

    with pytest.raises(ValueError, match=rf"bundled project\.{key}"):
        _verify_sdist(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "requirement",
    [
        None,
        "numpy>=1.26.0",
        "numpy>=1.27.0,<2.5",
        'numpy>=1.26.0,<2.5; extra == "plot"',
        "unrelated-package>=1.26.0,<2.5",
        "numpy=>1.26.0",
        "numpy<2.5,>=1.26.0\nRequires-Dist: numpy<2.5,>=1.26.0",
    ],
)
def test_distributions_reject_missing_wrong_or_duplicate_runtime_requirements(
    tmp_path: Path, kind: str, requirement: str | None
) -> None:
    metadata = _metadata(**{"Requires-Dist": requirement or "numpy<2.5,>=1.26.0"})
    if requirement is None:
        metadata = metadata.replace(b"Requires-Dist: numpy<2.5,>=1.26.0\n", b"")
    if kind == "wheel":
        path = tmp_path / "wrong-dependency.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = metadata
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "wrong-dependency.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = metadata
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="Requires-Dist"):
        verifier(path, PROJECT, PACKAGE_FILES)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_distributions_reject_undeclared_provided_extras(tmp_path: Path, kind: str) -> None:
    metadata = _metadata(**{"Provides-Extra": "unexpected-extra"})
    if kind == "wheel":
        path = tmp_path / "wrong-extra.whl"
        members = _wheel_members()
        members[f"{DIST_INFO}/METADATA"] = metadata
        _write_wheel(path, members)
        verifier = _verify_wheel
    else:
        path = tmp_path / "wrong-extra.tar.gz"
        members = _sdist_members()
        members[f"{SDIST_ROOT}/PKG-INFO"] = metadata
        _write_sdist(path, members)
        verifier = _verify_sdist

    with pytest.raises(ValueError, match="Provides-Extra"):
        verifier(path, PROJECT, PACKAGE_FILES)


def test_dependency_metadata_supports_pep508_extras_direct_references_and_markers() -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {
        "dependencies": [
            'Demo[fast] >= 1, != 2; python_version >= "3.11"',
            'direct @ https://example.com/direct.whl ; sys_platform == "linux"',
        ],
        "optional-dependencies": {
            "Fancy_Plot": ['plotter>=2; python_version >= "3.11" or platform_system == "Linux"'],
            "empty": [],
        },
    }
    metadata = BytesParser().parsebytes(_metadata())
    del metadata["Requires-Dist"]
    metadata["Requires-Dist"] = 'demo[fast]!=2,>=1; python_version >= "3.11"'
    metadata["Requires-Dist"] = 'direct @ https://example.com/direct.whl ; sys_platform == "linux"'
    metadata["Requires-Dist"] = (
        'plotter>=2; (python_version >= "3.11" or platform_system == "Linux") '
        'and extra == "fancy-plot"'
    )
    metadata["Provides-Extra"] = "empty"
    metadata["Provides-Extra"] = "fancy-plot"
    _verify_metadata(Path("valid.whl"), metadata, project)

    # Omitting parentheses broadens the optional requirement into a mandatory
    # dependency for some environments; that change must not pass verification.
    del metadata["Requires-Dist"]
    metadata["Requires-Dist"] = 'demo[fast]!=2,>=1; python_version >= "3.11"'
    metadata["Requires-Dist"] = 'direct @ https://example.com/direct.whl ; sys_platform == "linux"'
    metadata["Requires-Dist"] = (
        'plotter>=2; python_version >= "3.11" or platform_system == "Linux" '
        'and extra == "fancy-plot"'
    )
    with pytest.raises(ValueError, match="Requires-Dist"):
        _verify_metadata(Path("wrong-marker.whl"), metadata, project)


@pytest.mark.parametrize("provided", [[], ["plot", "plot"], ["wrong"]])
def test_declared_optional_extras_require_complete_unique_metadata(provided: list[str]) -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {"optional-dependencies": {"plot": []}}
    metadata = BytesParser().parsebytes(_metadata())
    for extra in provided:
        metadata["Provides-Extra"] = extra
    with pytest.raises(ValueError, match="Provides-Extra"):
        _verify_metadata(Path("wrong-extra.whl"), metadata, project)


def test_direct_reference_metadata_must_keep_the_declared_url() -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {"dependencies": ["direct @ https://example.com/direct.whl"]}
    metadata = BytesParser().parsebytes(
        _metadata(**{"Requires-Dist": "direct @ https://example.com/other.whl"})
    )
    with pytest.raises(ValueError, match="Requires-Dist"):
        _verify_metadata(Path("wrong-reference.whl"), metadata, project)


@pytest.mark.parametrize(
    "requirement",
    [
        'matplotlib>=3.8; extra == "plot"',
        'matplotlib>=3.8,<4; extra == "wrong"',
        "matplotlib>=3.8,<4",
    ],
)
def test_optional_dependencies_keep_their_constraints_and_extra_marker(requirement: str) -> None:
    from email.parser import BytesParser
    from pathlib import Path

    project = PROJECT | {"optional-dependencies": {"plot": ["matplotlib>=3.8,<4"]}}
    metadata = BytesParser().parsebytes(_metadata(**{"Provides-Extra": "plot"}))
    metadata["Requires-Dist"] = requirement
    with pytest.raises(ValueError, match="Requires-Dist"):
        _verify_metadata(Path("wrong-optional.whl"), metadata, project)


@pytest.mark.parametrize(
    "support",
    [
        "tests/fixtures/parity/reference.json",
        "tests/fixtures/distributions/README.txt",
        "tests/fixtures/distributions/reference.whl",
        "tests/fixtures/distributions/reference.tar.gz",
        "tests/__init__.py",
        "tests/test_property_invariants.py",
        "scripts/check_dist.py",
        "scripts/check.sh",
        "benchmarks/_measurement.py",
        "benchmarks/bench_screen.py",
        "uv.lock",
    ],
)
def test_sdist_requires_original_test_support(tmp_path: Path, support: str) -> None:
    path = tmp_path / "test-support.tar.gz"
    expected = {support: b"test support"}
    members = _sdist_members()
    _write_sdist(path, members)
    with pytest.raises(ValueError, match="missing required files"):
        _verify_sdist_with_support(path, PROJECT, PACKAGE_FILES, support_files=expected)

    members[f"{SDIST_ROOT}/{support}"] = expected[support]
    _write_sdist(path, members)
    _verify_sdist_with_support(path, PROJECT, PACKAGE_FILES, support_files=expected)

    members[f"{SDIST_ROOT}/{support}"] = b"stale test support"
    _write_sdist(path, members)
    with pytest.raises(ValueError, match="source file .* does not match the checkout"):
        _verify_sdist_with_support(path, PROJECT, PACKAGE_FILES, support_files=expected)
