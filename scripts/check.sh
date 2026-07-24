#!/usr/bin/env bash
# Run the full local quality suite used before opening a PR.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
else
  RUN=()
fi

echo "==> pytest + coverage"
"${RUN[@]}" pytest tests/ -v --cov=ier --cov-report=term-missing

echo "==> ruff check"
"${RUN[@]}" ruff check .

echo "==> ruff format"
"${RUN[@]}" ruff format --check .

echo "==> mypy"
"${RUN[@]}" mypy src/ier

echo "==> pylint"
"${RUN[@]}" pylint src/ier --fail-under=9.0

echo "==> bandit"
"${RUN[@]}" bandit -r src/ier -c pyproject.toml

if [[ "${SKIP_DOCS:-0}" != "1" ]]; then
  echo "==> mkdocs"
  "${RUN[@]}" mkdocs build --strict
fi

echo "All checks passed."
