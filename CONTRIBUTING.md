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

The `dev` extra composes `full`, `plot`, and `docs` plus test/lint tooling.

## Run Quality Checks

Run the unified suite before opening a PR:

```bash
./scripts/check.sh
```

When `uv` is available, the script first synchronizes the locked `dev` extra,
then runs every command without further environment mutation. Without `uv`, it
uses tools from the active environment.

Skip the docs build with `SKIP_DOCS=1 ./scripts/check.sh`.

Or run checks individually:

```bash
pytest tests/ -v --cov=ier --cov-report=term-missing
ruff check .
ruff format --check .
mypy src/ier
pylint src/ier
bandit -r src/ier -c pyproject.toml
uv run mkdocs build --strict
```

### Lint roles

- **Ruff** is the primary linter/formatter (fast, CI-blocking style + bugbear rules).
- **Pylint** remains a secondary depth check (`--fail-under=9.0`) for a few
  heuristics Ruff does not cover. Prefer fixing Ruff findings first.

Optional benchmarks:

```bash
uv run python benchmarks/bench_screen.py
uv run python benchmarks/bench_detection.py
```

Verify release artifacts after packaging changes:

```bash
uv build
uv run --no-project python scripts/check_dist.py dist/*
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for registry design, flagging
policy, NA handling, and composite score caveats.

## Pull Request Expectations

- Add or update tests for behavioral changes.
- Keep public docs/examples aligned with API changes.
- Keep CI green (tests, lint, security, docs workflows).
- Open an issue first for large API or behavior changes.
- Version bumps in `pyproject.toml` are required when `src/` changes
  (docs/CI-only PRs do not need a bump; use the `no-version-bump` label to skip).

## Release Process

The repository supports two release paths:

- Tag-based GitHub release workflow (`vX.Y.Z`) — runs the full CI suite, then
  produces artifacts and a GitHub Release.
- Publish workflow (`Publish to PyPI`) — runs tests, builds, then uploads to
  TestPyPI/PyPI.

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
- Source-changing PRs must set a valid semantic version strictly greater than
  the version on `main`; CI rejects unchanged versions and downgrades.
- Bump `/pyproject.toml` when preparing a new public package release.
- PyPI does not allow uploading new files for a version that already exists.
  If you need to correct packaging for the same code, publish a new patch version.
