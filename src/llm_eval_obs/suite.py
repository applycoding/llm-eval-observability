"""Eval suite files bind a dataset, a model, and the scorers to run."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_eval_obs.errors import SuiteError
from llm_eval_obs.yaml_util import load_yaml


@dataclass(frozen=True)
class ExactSpec:
    type: Literal["exact"]
    name: str
    casefold: bool = True


@dataclass(frozen=True)
class RegexSpec:
    type: Literal["regex"]
    name: str
    ignore_case: bool = True


@dataclass(frozen=True)
class EmbeddingSpec:
    type: Literal["embedding"]
    name: str
    threshold: float = 0.15


@dataclass(frozen=True)
class JudgeSpec:
    type: Literal["llm_judge"]
    name: str
    rubric: Path


ScorerSpec = ExactSpec | RegexSpec | EmbeddingSpec | JudgeSpec


@dataclass(frozen=True)
class Suite:
    id: str
    prompt_id: str
    dataset: Path
    dataset_name: str
    model: str
    price_catalog: Path
    scorers: tuple[ScorerSpec, ...]


def load_suite(path: Path, *, root: Path) -> Suite:
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise SuiteError(f"{path} must contain a mapping")
    suite_id = _text(raw.get("id"), path=path, field="id")
    prompt_id = _text(raw.get("prompt_id"), path=path, field="prompt_id")
    dataset_name = _text(raw.get("dataset"), path=path, field="dataset")
    model = _text(raw.get("model"), path=path, field="model")
    catalog_name = _text(raw.get("price_catalog"), path=path, field="price_catalog")
    scorer_raw = raw.get("scorers")
    if not isinstance(scorer_raw, list) or not scorer_raw:
        raise SuiteError(f"{path} needs at least one scorer")
    scorers = tuple(_parse_scorer(item, path=path, root=root) for item in scorer_raw)
    names = [spec.name for spec in scorers]
    if len(names) != len(set(names)):
        raise SuiteError(f"{path} contains a duplicate scorer name")
    return Suite(
        id=suite_id,
        prompt_id=prompt_id,
        dataset=(root / dataset_name).resolve(),
        dataset_name=dataset_name,
        model=model,
        price_catalog=(root / catalog_name).resolve(),
        scorers=scorers,
    )


def _parse_scorer(raw: object, *, path: Path, root: Path) -> ScorerSpec:
    if not isinstance(raw, dict):
        raise SuiteError(f"{path} scorer entries must be mappings")
    name = raw.get("name")
    kind = raw.get("type")
    if not isinstance(name, str) or not name.strip():
        raise SuiteError(f"{path} scorer is missing a name")
    match kind:
        case "exact":
            return ExactSpec(type="exact", name=name.strip(), casefold=_flag(raw, "casefold", path))
        case "regex":
            return RegexSpec(
                type="regex",
                name=name.strip(),
                ignore_case=_flag(raw, "ignore_case", path),
            )
        case "embedding":
            threshold = raw.get("threshold", 0.15)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise SuiteError(f"{path} embedding threshold must be a number")
            if not 0 <= float(threshold) <= 1:
                raise SuiteError(f"{path} embedding threshold must be between 0 and 1")
            return EmbeddingSpec(type="embedding", name=name.strip(), threshold=float(threshold))
        case "llm_judge":
            rubric = raw.get("rubric")
            if not isinstance(rubric, str) or not rubric.strip():
                raise SuiteError(f"{path} llm_judge scorer needs a rubric path")
            return JudgeSpec(type="llm_judge", name=name.strip(), rubric=(root / rubric).resolve())
        case _:
            raise SuiteError(f"{path} has an unknown scorer type: {kind}")


def _text(value: object, *, path: Path, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise SuiteError(f"{path} is missing {field}")
    if not isinstance(value, (str, int, float)):
        raise SuiteError(f"{path} field {field} must be a string")
    text = str(value).strip()
    if not text:
        raise SuiteError(f"{path} field {field} is empty")
    return text


def _flag(raw: dict[str, object], key: str, path: Path) -> bool:
    value = raw.get(key, True)
    if not isinstance(value, bool):
        raise SuiteError(f"{path} scorer flag {key} must be a boolean")
    return value
