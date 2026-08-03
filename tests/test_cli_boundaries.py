"""Architecture checks for command-line module responsibilities."""

from __future__ import annotations

import ast
from pathlib import Path


def test_command_entry_does_not_import_result_serializers() -> None:
    """Keep format-specific result routing behind the dedicated boundary."""
    source_path = Path(__file__).parents[1] / "src" / "ier" / "cli.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

    assert all(node.module != "ier._cli_output" for node in imports)
    npz_imports = [node for node in imports if node.module == "ier._cli_npz"]
    assert len(npz_imports) == 1
    assert [name.name for name in npz_imports[0].names] == ["_require_npz_output_path"]
