"""Exported Literal aliases name every option value set restated across signatures."""

import re
from dataclasses import fields
from pathlib import Path
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

import ier
import ier.types
from ier._registry import IndexOptions
from ier.tables import index_agreement
from ier.visualize import plot_index_agreement

_ROOT = Path(__file__).resolve().parents[1]
_ALIASES = {
    "AgreementKind": ("overlap", "jaccard", "spearman"),
    "EvenOddMethod": ("item_pairs", "halves"),
    "ItemCorrelationMode": ("complete", "pairwise"),
}


@pytest.mark.parametrize(("name", "values"), _ALIASES.items(), ids=list(_ALIASES))
def test_option_aliases_are_exported_literals(name: str, values: tuple[str, ...]) -> None:
    alias = getattr(ier, name)
    assert alias is getattr(ier.types, name)
    assert name in ier.__all__
    assert get_origin(alias) is Literal
    assert get_args(alias) == values


def test_option_aliases_are_listed_in_the_api_reference() -> None:
    api = (_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    members = set(re.findall(r"^\s+- (\S+)$", api, re.MULTILINE))
    assert set(_ALIASES) <= members


def test_agreement_and_item_correlation_options_use_the_aliases() -> None:
    assert get_type_hints(index_agreement)["kind"] == ier.AgreementKind
    assert get_type_hints(plot_index_agreement)["kind"] == ier.AgreementKind
    hints = get_type_hints(IndexOptions)
    assert hints["psychsyn_item_correlations"] == ier.ItemCorrelationMode
    assert hints["evenodd_method"] == ier.EvenOddMethod
    defaults = {field.name: field.default for field in fields(IndexOptions)}
    assert defaults["psychsyn_item_correlations"] in get_args(ier.ItemCorrelationMode)
    for module, restated in (
        ("tables.py", 'Literal["overlap", "jaccard", "spearman"]'),
        ("visualize.py", 'Literal["overlap", "jaccard", "spearman"]'),
        ("_registry.py", 'Literal["complete", "pairwise"]'),
    ):
        assert restated not in (_ROOT / "src" / "ier" / module).read_text(encoding="utf-8")
