"""Load regression cases from JSONL or YAML."""

import json
from pathlib import Path
from typing import Any

from llm_eval_obs.errors import DatasetError
from llm_eval_obs.models import TestCase
from llm_eval_obs.yaml_util import load_yaml


def load_dataset(path: Path) -> list[TestCase]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        cases = _load_jsonl(path)
    elif suffix in {".yaml", ".yml"}:
        cases = _load_yaml(path)
    else:
        raise DatasetError(f"{path} must be a .jsonl, .yaml, or .yml file")
    if not cases:
        raise DatasetError(f"{path} does not contain any cases")
    return _unique(cases, path)


def parse_case(raw: Any, *, where: str) -> TestCase:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: case must be a mapping")
    case_id = _text_field(raw.get("id"), where=where, field="id")
    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, dict) or not inputs_raw:
        raise DatasetError(f"{where}: inputs must be a non-empty mapping")
    inputs: dict[str, str] = {}
    for key, value in inputs_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise DatasetError(f"{where}: input {key!r} must be a string")
        inputs[key] = value
    tags_raw = raw.get("tags", [])
    if not isinstance(tags_raw, list) or not all(isinstance(tag, str) for tag in tags_raw):
        raise DatasetError(f"{where}: tags must be a list of strings")
    return TestCase(
        id=case_id,
        inputs=inputs,
        expected=_optional_text(raw.get("expected"), where=where, field="expected"),
        expected_regex=_optional_text(
            raw.get("expected_regex"), where=where, field="expected_regex"
        ),
        reference=_optional_text(raw.get("reference"), where=where, field="reference"),
        tags=tuple(tags_raw),
    )


def _load_jsonl(path: Path) -> list[TestCase]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"could not read {path}: {exc}") from exc
    cases: list[TestCase] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{where}: {exc}") from exc
        cases.append(parse_case(raw, where=where))
    return cases


def _load_yaml(path: Path) -> list[TestCase]:
    raw = load_yaml(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("cases"), list):
        raise DatasetError(f"{path} must be a mapping with a cases list")
    cases: list[TestCase] = []
    for index, item in enumerate(raw["cases"], start=1):
        cases.append(parse_case(item, where=f"{path}:case {index}"))
    return cases


def _unique(cases: list[TestCase], path: Path) -> list[TestCase]:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"{path} contains a duplicate case id {case.id}")
        seen.add(case.id)
    return cases


def _text_field(value: Any, *, where: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{where}: {field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, *, where: str, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetError(f"{where}: {field} must be a string")
    stripped = value.strip()
    return stripped or None
