"""JSONL and YAML case loaders."""

import pytest

from llm_eval_obs.datasets import load_dataset
from llm_eval_obs.errors import DatasetError
from tests.factories import ROOT


def test_yaml_dataset_matches_jsonl() -> None:
    jsonl = load_dataset(ROOT / "datasets" / "support.jsonl")
    yaml_cases = load_dataset(ROOT / "datasets" / "support.yaml")
    assert yaml_cases == jsonl
    assert [case.id for case in jsonl] == [
        "refund-window",
        "shipping-delay",
        "password-reset",
        "damaged-item",
        "exact-ack",
    ]


def test_duplicate_case_id_is_rejected(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    line = '{"id": "a", "inputs": {"question": "q"}}\n'
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(path)


def test_non_string_input_is_rejected(tmp_path) -> None:
    path = tmp_path / "cases.yaml"
    path.write_text("cases:\n  - id: a\n    inputs:\n      question: 5\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="string"):
        load_dataset(path)


def test_invalid_jsonl_line_names_the_line(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="cases.jsonl:1"):
        load_dataset(path)


def test_unsupported_suffix_is_rejected(tmp_path) -> None:
    path = tmp_path / "cases.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(DatasetError, match="jsonl"):
        load_dataset(path)
