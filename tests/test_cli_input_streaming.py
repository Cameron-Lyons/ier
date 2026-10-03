"""Selected CSV cells retain numerical meaning across streaming transports."""

from __future__ import annotations

import csv
import gzip
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from ier._cli_input import _load_input


class _ForwardOnlyInput(StringIO):
    def read(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("matrix input must not be read into a complete string")

    def seek(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("matrix input must not be rewound")


@pytest.mark.parametrize("transport", ["plain", "gzip", "stdin"])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("missing", ["none", "blank", "tokens"])
def test_streamed_numeric_cells_match_known_matrix_beyond_the_sniffing_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    selected: bool,
    missing: str,
) -> None:
    values = (np.arange(600 * 5).reshape(600, 5) - 1000) / 8
    values[13, 1] = np.inf
    values[29, 4] = -np.inf
    values[51, 2] = np.nan
    if missing != "none":
        values[::17, 0] = np.nan
        values[::19, 3] = np.nan
    identifiers = [
        f'person,{index};\t"quoted"\ncontinued' if index % 31 == 0 else f"person{index}"
        for index in range(len(values))
    ]
    item_names = ["first", "item, quoted", "third", "fourth", "fifth"]
    selected_order = [4, 0, 1, 3, 2]
    content = StringIO(newline="")
    content.write("survey preamble\nexport details\n\ufeff")
    writer = csv.writer(content)
    writer.writerow(
        ["notes", item_names[2], "id", item_names[0], item_names[4], item_names[1], item_names[3]]
        if selected
        else item_names
    )
    for row_index, row in enumerate(values):
        cells = [
            f"{value:+.3f}" if column % 2 == 0 else f" {value:.6e} "
            for column, value in enumerate(row)
        ]
        if missing != "none":
            if row_index % 17 == 0:
                cells[0] = "MISSING" if missing == "tokens" else ""
            if row_index % 19 == 0:
                cells[3] = " N/A " if missing == "tokens" else " \t "
        writer.writerow(
            [
                "text, with\nquoted metadata",
                cells[2],
                identifiers[row_index],
                *[cells[i] for i in (0, 4, 1, 3)],
            ]
            if selected
            else cells
        )
    text = content.getvalue()
    assert len(text) > 4096 * 5
    stdin = _ForwardOnlyInput(text)
    if transport == "stdin":
        path = Path("-")
        monkeypatch.setattr("sys.stdin", stdin)
    elif transport == "gzip":
        path = tmp_path / "survey.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
            handle.write(text)
    else:
        path = tmp_path / "survey.csv"
        path.write_text(text, encoding="utf-8")

    actual, actual_ids = _load_input(
        path,
        ",",
        id_column="id" if selected else None,
        item_columns=[item_names[i] for i in selected_order] if selected else None,
        missing_values=["MISSING", "N/A"] if missing == "tokens" else None,
        skip_rows=2,
    )

    np.testing.assert_array_equal(actual, values[:, selected_order] if selected else values)
    assert actual.dtype == np.dtype(np.float64)
    assert actual_ids == (identifiers if selected else None)
    assert not stdin.closed
