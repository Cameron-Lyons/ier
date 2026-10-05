"""Catalog option requirements stay consistent across the API and `ier indices`."""

from __future__ import annotations

import csv
import json
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from ier import IndexMetadata, IndexOptions, index_catalog, infrequency, screen
from ier._cli_output import (
    _emit_index_catalog_json,
    _emit_index_catalog_text,
    _write_index_catalog_csv,
)
from ier.cli import main

if TYPE_CHECKING:
    from pathlib import Path

    from ier import IndexCatalog

_ANSWER_FORMS = ("infrequency_expected_responses", "infrequency_acceptable_ranges")


def _run_indices(arguments: list[str]) -> str:
    stdout = StringIO()
    with patch("sys.stdout", stdout):
        assert main(["indices", *arguments]) == 0
    return stdout.getvalue()


def test_catalog_entries_carry_every_metadata_key() -> None:
    for name, metadata in index_catalog().items():
        assert set(metadata) == set(IndexMetadata.__annotations__), name
        assert all(len(group) > 0 for group in metadata["alternative_options"]), name


def test_infrequency_needs_item_indices_and_either_answer_form() -> None:
    entry = index_catalog()["infrequency"]
    assert entry["required_options"] == ("infrequency_item_indices",)
    assert entry["alternative_options"] == (_ANSWER_FORMS,)

    # Either form alone satisfies the advertised requirements, even under strict scoring.
    data = np.array([[1, 3, 5, 2], [4, 3, 1, 2], [2, 3, 2, 5]], dtype=float)
    ranges: list[tuple[float, float]] = [(1, 2), (4, float("inf"))]
    for options, expected in [
        (
            IndexOptions(infrequency_item_indices=[0, 2], infrequency_acceptable_ranges=ranges),
            infrequency(data, [0, 2], acceptable_ranges=ranges),
        ),
        (
            IndexOptions(infrequency_item_indices=[0, 2], infrequency_expected_responses=[1, 5]),
            infrequency(data, [0, 2], expected_responses=[1, 5]),
        ),
    ]:
        result = screen(data, indices=["infrequency"], options=options, strict=True)
        np.testing.assert_array_equal(result["scores"]["infrequency"], expected)


def test_indices_text_lists_alternative_option_groups() -> None:
    lines = _run_indices([]).splitlines()
    assert lines[0].split("\t")[-3:] == [
        "required_options",
        "alternative_options",
        "keyed_responses",
    ]
    rows = {line.split("\t")[0]: line.split("\t") for line in lines[1:]}
    assert all(len(row) == len(lines[0].split("\t")) for row in rows.values())
    assert rows["infrequency"][-3:-1] == ["infrequency_item_indices", "|".join(_ANSWER_FORMS)]
    assert rows["mad"][-3:-1] == ["mad_positive_items,mad_negative_items", "-"]
    assert rows["irv"][-3:-1] == ["-", "-"]


def test_indices_json_lists_alternative_option_groups(tmp_path: Path) -> None:
    out = tmp_path / "indices.json"
    _run_indices(["--format", "json", "--output", str(out)])
    indices = json.loads(out.read_text(encoding="utf-8"))["indices"]
    assert indices["infrequency"]["required_options"] == ["infrequency_item_indices"]
    assert indices["infrequency"]["alternative_options"] == [list(_ANSWER_FORMS)]
    assert indices["evenodd"]["alternative_options"] == []
    assert all("alternative_options" in metadata for metadata in indices.values())


def test_indices_csv_lists_alternative_option_groups(tmp_path: Path) -> None:
    out = tmp_path / "indices.csv"
    _run_indices(["--format", "csv", "--output", str(out)])
    reader = csv.DictReader(StringIO(out.read_text(encoding="utf-8")))
    assert reader.fieldnames is not None
    assert reader.fieldnames[-3:] == [
        "required_options",
        "alternative_options",
        "uses_keyed_responses",
    ]
    rows = {row["index"]: row for row in reader}
    assert rows["infrequency"]["required_options"] == "infrequency_item_indices"
    assert rows["infrequency"]["alternative_options"] == "|".join(_ANSWER_FORMS)
    assert rows["mad"]["required_options"] == "mad_positive_items,mad_negative_items"
    assert rows["irv"]["alternative_options"] == ""


@pytest.fixture
def grouped_catalog() -> IndexCatalog:
    return {
        "grouped": {
            "flag_direction": "high",
            "flag_mode": "percentile",
            "default_screen": False,
            "default_composite": False,
            "composite_enabled": True,
            "required_options": ("scale_min", "scale_max"),
            "alternative_options": (("a", "b", "c"), ("d",)),
            "uses_keyed_responses": True,
        }
    }


def test_catalog_formats_separate_groups_and_alternatives(grouped_catalog: IndexCatalog) -> None:
    text_row = _emit_index_catalog_text(grouped_catalog).splitlines()[1].split("\t")
    assert text_row[-3:] == ["scale_min,scale_max", "a|b|c;d", "yes"]

    handle = StringIO()
    _write_index_catalog_csv(handle, grouped_catalog)
    (row,) = csv.DictReader(StringIO(handle.getvalue()))
    assert row["required_options"] == "scale_min,scale_max"
    assert row["alternative_options"] == "a|b|c;d"

    payload = json.loads(_emit_index_catalog_json(grouped_catalog))
    assert payload["indices"]["grouped"]["alternative_options"] == [["a", "b", "c"], ["d"]]
