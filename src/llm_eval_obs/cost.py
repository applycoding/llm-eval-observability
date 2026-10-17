"""Token-cost estimates from an explicit price catalog.

Prices are never inferred. A model that is absent from the catalog costs 0 and
is marked unknown so a report cannot present that zero as a measured price.
"""

from dataclasses import dataclass
from pathlib import Path

from llm_eval_obs.errors import EvalObsError
from llm_eval_obs.yaml_util import load_yaml


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


def load_catalog(path: Path) -> dict[str, ModelPrice]:
    raw = load_yaml(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise EvalObsError(f"{path} must contain a models mapping")
    catalog: dict[str, ModelPrice] = {}
    for name, spec in raw["models"].items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise EvalObsError(f"{path} has an invalid price entry")
        catalog[name] = ModelPrice(
            input_per_million=_non_negative(spec.get("input_per_million"), path, name),
            output_per_million=_non_negative(spec.get("output_per_million"), path, name),
        )
    return catalog


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    catalog: dict[str, ModelPrice],
) -> tuple[float, bool]:
    """Return `(usd, known)`. Unknown models return `(0.0, False)`."""

    price = catalog.get(model)
    if price is None:
        return 0.0, False
    usd = (
        input_tokens * price.input_per_million + output_tokens * price.output_per_million
    ) / 1_000_000
    return round(usd, 10), True


def _non_negative(value: object, path: Path, model: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise EvalObsError(f"{path} price for {model} must be a non-negative number")
    return float(value)
