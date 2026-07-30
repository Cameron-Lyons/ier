#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  echo "==> sync locked quality environments"
  uv sync --locked --all-groups
  RUN=(uv run --no-sync)
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

if [[ "${SKIP_DOCS:-0}" != "1" ]]; then
  echo "==> mkdocs"
  "${RUN[@]}" mkdocs build --strict
fi

echo "All checks passed."
