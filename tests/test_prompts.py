"""Prompt loading and rendering."""

import pytest

from llm_eval_obs.errors import PromptError
from llm_eval_obs.prompts import PromptRegistry, load_prompt, render_prompt
from tests.factories import ROOT


def test_registry_loads_both_support_versions() -> None:
    registry = PromptRegistry.load(ROOT / "prompts" / "support")
    first = registry.get("support", "1")
    second = registry.get("support", "2")
    assert "policy" not in first.template.lower()
    assert "Policy:" in second.template
    assert registry.versions("support") == ("1", "2")


def test_render_replaces_placeholders_and_rejects_missing_names() -> None:
    rendered = render_prompt("Hello {{ name }}", {"name": "Ada", "unused": "x"})
    assert rendered == "Hello Ada"
    with pytest.raises(PromptError, match="missing template variables: name"):
        render_prompt("Hello {{ name }}", {})


def test_duplicate_version_is_rejected(tmp_path) -> None:
    body = "id: support\nversion: '1'\ntemplate: Hello\n"
    (tmp_path / "a.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(PromptError, match="duplicate prompt"):
        PromptRegistry.load(tmp_path)


def test_unknown_version_lists_known_prompts(tmp_path) -> None:
    (tmp_path / "one.yaml").write_text(
        "id: support\nversion: '1'\ntemplate: Hello\n",
        encoding="utf-8",
    )
    registry = PromptRegistry.load(tmp_path)
    with pytest.raises(PromptError, match="support@1"):
        registry.get("support", "9")


def test_prompt_file_requires_a_template(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("id: support\nversion: '1'\n", encoding="utf-8")
    with pytest.raises(PromptError, match="template"):
        load_prompt(path)
