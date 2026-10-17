"""Judge rubrics: named criteria, weights, and an integer scale."""

from dataclasses import dataclass
from pathlib import Path

from llm_eval_obs.errors import RubricError
from llm_eval_obs.yaml_util import load_yaml


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str
    weight: float


@dataclass(frozen=True)
class Rubric:
    id: str
    scale_min: int
    scale_max: int
    criteria: tuple[Criterion, ...]

    def normalized(self, criterion_id: str, value: int) -> float:
        span = self.scale_max - self.scale_min
        return (value - self.scale_min) / span


def load_rubric(path: Path) -> Rubric:
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise RubricError(f"{path} must contain a mapping")
    rubric_id = raw.get("id")
    if not isinstance(rubric_id, str) or not rubric_id.strip():
        raise RubricError(f"{path} is missing an id")
    scale_min = _integer(raw.get("scale_min"), path=path, field="scale_min")
    scale_max = _integer(raw.get("scale_max"), path=path, field="scale_max")
    if scale_max <= scale_min:
        raise RubricError(f"{path} scale_max must be greater than scale_min")
    criteria_raw = raw.get("criteria")
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise RubricError(f"{path} needs at least one criterion")
    criteria = tuple(
        _criterion(item, path=path, index=index) for index, item in enumerate(criteria_raw, start=1)
    )
    ids = [item.id for item in criteria]
    if len(ids) != len(set(ids)):
        raise RubricError(f"{path} contains a duplicate criterion id")
    total = sum(item.weight for item in criteria)
    if total <= 0:
        raise RubricError(f"{path} criterion weights must sum to a positive number")
    weighted = tuple(Criterion(item.id, item.description, item.weight / total) for item in criteria)
    return Rubric(id=rubric_id.strip(), scale_min=scale_min, scale_max=scale_max, criteria=weighted)


def _criterion(raw: object, *, path: Path, index: int) -> Criterion:
    if not isinstance(raw, dict):
        raise RubricError(f"{path} criterion {index} must be a mapping")
    criterion_id = raw.get("id")
    description = raw.get("description")
    weight = raw.get("weight", 1)
    if not isinstance(criterion_id, str) or not criterion_id.strip():
        raise RubricError(f"{path} criterion {index} is missing an id")
    if not isinstance(description, str) or not description.strip():
        raise RubricError(f"{path} criterion {criterion_id} is missing a description")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
        raise RubricError(f"{path} criterion {criterion_id} weight must be a positive number")
    return Criterion(id=criterion_id.strip(), description=description.strip(), weight=float(weight))


def _integer(value: object, *, path: Path, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RubricError(f"{path} {field} must be an integer")
    return value
