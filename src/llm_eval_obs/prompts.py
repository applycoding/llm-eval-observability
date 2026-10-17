"""Load versioned prompt templates and render `{{ variable }}` placeholders."""

import re
from pathlib import Path

from llm_eval_obs.errors import PromptError
from llm_eval_obs.models import PromptTemplate
from llm_eval_obs.yaml_util import load_yaml

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def render_prompt(template: str, inputs: dict[str, str]) -> str:
    """Substitute placeholders. Missing names raise PromptError; extra inputs are ignored."""

    required = set(_PLACEHOLDER.findall(template))
    missing = sorted(required - set(inputs))
    if missing:
        names = ", ".join(missing)
        raise PromptError(f"missing template variables: {names}")

    def replace(match: re.Match[str]) -> str:
        return inputs[match.group(1)]

    return _PLACEHOLDER.sub(replace, template)


def load_prompt(path: Path) -> PromptTemplate:
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise PromptError(f"{path} must contain a mapping")
    prompt_id = _require_text(data, "id", path)
    version = _require_text(data, "version", path)
    template = _require_text(data, "template", path)
    description = data.get("description", "")
    if not isinstance(description, str):
        raise PromptError(f"{path} description must be a string")
    return PromptTemplate(
        id=prompt_id,
        version=version,
        template=template,
        description=description.strip(),
    )


class PromptRegistry:
    """Prompts keyed by id and version. Duplicate pairs are rejected at load time."""

    def __init__(self, prompts: list[PromptTemplate]) -> None:
        self._prompts: dict[tuple[str, str], PromptTemplate] = {}
        for prompt in prompts:
            key = (prompt.id, prompt.version)
            if key in self._prompts:
                raise PromptError(f"duplicate prompt {prompt.id} version {prompt.version}")
            self._prompts[key] = prompt

    def get(self, prompt_id: str, version: str) -> PromptTemplate:
        try:
            return self._prompts[(prompt_id, version)]
        except KeyError as exc:
            known = ", ".join(f"{pid}@{ver}" for pid, ver in sorted(self._prompts)) or "(none)"
            raise PromptError(
                f"unknown prompt {prompt_id} version {version}. Known prompts: {known}"
            ) from exc

    def versions(self, prompt_id: str) -> tuple[str, ...]:
        found = [version for pid, version in self._prompts if pid == prompt_id]
        return tuple(sorted(found, key=_version_sort_key))

    @classmethod
    def load(cls, directory: Path) -> "PromptRegistry":
        if not directory.is_dir():
            raise PromptError(f"prompt directory does not exist: {directory}")
        paths = sorted(
            path
            for path in directory.rglob("*")
            if path.suffix.lower() in {".yaml", ".yml"} and path.is_file()
        )
        if not paths:
            raise PromptError(f"no prompt files found in {directory}")
        return cls([load_prompt(path) for path in paths])


def _require_text(data: dict[str, object], key: str, path: Path) -> str:
    value = data.get(key)
    if value is None or not isinstance(value, (str, int, float)):
        raise PromptError(f"{path} is missing a {key}")
    text = str(value).strip()
    if not text:
        raise PromptError(f"{path} has an empty {key}")
    return text


def _version_sort_key(version: str) -> tuple[int, str]:
    if version.isdigit():
        return (int(version), version)
    return (10**9, version)
