# Contributing to IER

Thanks for contributing. This document covers local setup, quality checks, and release flow.

## Development Setup

### Recommended (`uv`)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
uv sync --all-groups
```

### Fallback (`pip`)

```bash
git clone https://github.com/Cameron-Lyons/ier.git
cd ier
python -m pip install -e . --group integration --group lint --group docs --group security
```

Development dependencies are split into `test`, `integration`, `lint`, `docs`,
and `security` groups. `integration` includes the test group plus pandas and
Polars compatibility checks; `uv sync --all-groups` installs the complete
contributor environment.

## Run Quality Checks

Run the unified suite before opening a PR:

```bash
./scripts/check.sh
```

The script first verifies that the editable project version in `uv.lock` matches
`pyproject.toml`. When `uv` is available, it then synchronizes all locked dependency
groups and runs every command without further environment mutation. Without `uv`,
it uses tools from the active environment.

Skip the docs build with `SKIP_DOCS=1 ./scripts/check.sh`.

Or run checks individually:

```bash
uv run --no-sync pytest tests/ -v --cov=ier --cov-report=term-missing
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src/ier
uv run --no-sync mkdocs build --strict
```

### Lint roles

- **Ruff** is the linter, formatter, and static security scanner. The selected
  rules include pycodestyle, Pyflakes, isort, pyupgrade, Bugbear, simplify,
  type-checking, flake8-bandit security checks, and Pylint error/convention/warning rules.
- **mypy** performs strict static type checking for the public and internal source.

Pre-commit hooks use isolated tool environments and do not need a project
dependency. Install them with `uvx pre-commit install` when desired.

Optional benchmarks:

```bash
uv run python benchmarks/bench_screen.py
uv run python benchmarks/bench_detection.py
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_markov.py --missing-rate 0.1
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_sequence_scoring.py --missing-rate 0.1
OPENBLAS_NUM_THREADS=1 uv run python benchmarks/bench_sequence_scoring.py --respondents 100 --items 1000 --operations longstring longstring_pattern
```

The sequence benchmark measures longstring, repeating-pattern, and Markov
indices alongside default screening and composite workflows. Compare with
`--missing-rate 0` to check complete-data performance. Use the same Python,
NumPy, BLAS thread count, and benchmark arguments for before/after comparisons;
wall-clock timing and peak allocation are measured in separate runs.
The Markov benchmark follows the same measurement policy and accepts `--states`
to compare small response scales with its sparse high-cardinality path.

Verify release artifacts after packaging changes:

```bash
uv build
uv run --no-project python scripts/check_dist.py dist/*
uv run --isolated --no-project --with dist/*.whl python scripts/smoke_test_install.py
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
  produces artifacts and a GitHub Release. The tag must exactly match
  `v<project.version>` from `pyproject.toml`.
- Publish workflow (`Publish to PyPI`) — runs tests, builds, then uploads to
  TestPyPI/PyPI. Release-triggered publishes enforce the same tag/version match.

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
- Keep the editable project version in `uv.lock` equal to `project.version`;
  local and CI checks reject drift before dependency synchronization.
- Bump `/pyproject.toml` when preparing a new public package release.
- PyPI does not allow uploading new files for a version that already exists.
  If you need to correct packaging for the same code, publish a new patch version.
