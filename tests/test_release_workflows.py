"""Regression tests for distribution reuse across publishing workflows."""

from pathlib import Path

import pytest

WORKFLOW_ROOT = Path(__file__).parents[1] / ".github" / "workflows"
VALIDATED_BUILD_STEPS = (
    "uv build",
    "scripts/check_dist.py",
    "uvx twine check",
    "scripts/smoke_test_install.py",
    "actions/upload-artifact",
)
REPEATED_BUILD_STEPS = (
    "uv build",
    "scripts/check_dist.py",
    "uvx twine check",
    "actions/upload-artifact",
)


def test_ci_workflow_owns_the_validated_distribution_build() -> None:
    workflow = (WORKFLOW_ROOT / "ci.yml").read_text(encoding="utf-8")

    for step in VALIDATED_BUILD_STEPS:
        assert workflow.count(step) == 1


@pytest.mark.parametrize("workflow_name", ["release.yml", "python-publish.yml"])
def test_publish_workflows_reuse_the_ci_distribution(workflow_name: str) -> None:
    workflow = (WORKFLOW_ROOT / workflow_name).read_text(encoding="utf-8")

    assert "  ci:\n" in workflow
    assert "uses: ./.github/workflows/ci.yml" in workflow
    assert "needs: ci" in workflow
    assert "needs: build" not in workflow
    assert "  build:\n" not in workflow
    for step in REPEATED_BUILD_STEPS:
        assert step not in workflow
