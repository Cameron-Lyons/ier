# Contributing to IER

Thanks for contributing. This document covers local setup, quality checks, and release flow.

## Development Setup

### Recommended (`uv`)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
uv sync --extra dev
```

### Fallback (`pip`)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
python -m pip install -e ".[dev]"
```

## Run Quality Checks

Run these before opening a PR:

```bash
pytest tests/ -v --cov=ier --cov-report=term-missing
ruff check .
ruff format --check .
mypy src/ier
pylint src/ier
```

## Pull Request Expectations

- Add or update tests for behavioral changes.
- Keep public docs/examples aligned with API changes.
- Keep CI green (tests, lint, security workflows).
- Open an issue first for large API or behavior changes.

## Release Process

The repository supports two release paths:

- Tag-based GitHub release workflow (`vX.Y.Z`) to produce artifacts.
- Publish workflow (`Publish to PyPI`) for TestPyPI/PyPI upload.

### Publish to TestPyPI

1. Open GitHub Actions.
2. Run `Publish to PyPI` manually.
3. Set `target=testpypi`.

### Publish to PyPI

1. Open GitHub Actions.
2. Run `Publish to PyPI` manually with `target=pypi`,
   or publish a GitHub Release.

## Versioning Policy

- Use semantic versioning (`MAJOR.MINOR.PATCH`).
- Bump `/pyproject.toml` when preparing a new public package release.
- PyPI does not allow uploading new files for a version that already exists.
  If you need to correct packaging for the same code, publish a new patch version.
