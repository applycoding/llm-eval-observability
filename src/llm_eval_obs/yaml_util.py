"""Small YAML reader shared by prompts, suites, rubrics, and the price catalog."""

from pathlib import Path
from typing import Any

import yaml

from llm_eval_obs.errors import EvalObsError


def load_yaml(path: Path) -> Any:
    """Return the parsed YAML document, or raise EvalObsError with the path."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalObsError(f"could not read {path}: {exc}") from exc
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise EvalObsError(f"invalid YAML in {path}: {exc}") from exc
