"""Fake model and the OpenAI-compatible HTTP client."""

import json
import urllib.error
import urllib.request
from email.message import EmailMessage

import pytest

from llm_eval_obs.datasets import load_dataset
from llm_eval_obs.errors import LLMError
from llm_eval_obs.llm import FakeLLM, HttpChatLLM, demo_completion, grounded_reply, word_tokens
from llm_eval_obs.prompts import PromptRegistry, render_prompt
from tests.factories import ROOT


def test_fake_task_reply_quotes_a_policy_line() -> None:
    registry = PromptRegistry.load(ROOT / "prompts" / "support")
    cases = {case.id: case for case in load_dataset(ROOT / "datasets" / "support.jsonl")}
    case = cases["exact-ack"]
    rendered = render_prompt(registry.get("support", "2").template, case.inputs)
    response = FakeLLM().complete(rendered, model="fake-llm")
    assert response.text == grounded_reply(case.inputs["policy"])
    assert response.text == case.expected
    generic = render_prompt(registry.get("support", "1").template, case.inputs)
    assert FakeLLM().complete(generic, model="fake-llm").text != case.expected


def test_fake_reports_word_tokens_and_configured_latency() -> None:
    client = FakeLLM(latency_ms=4.0)
    response = client.complete("one two three", model="fake-llm")
    assert response.input_tokens == word_tokens("one two three")
    assert response.latency_ms == 4.0
    assert client.calls == ["one two three"]


def test_demo_judge_completion_is_json() -> None:
    prompt = (
        "RESPOND WITH JSON ONLY.\n"
        "Scale: 1 to 5.\n"
        "- faithfulness: facts only\n"
        "REFERENCE_START\npolicy text\nREFERENCE_END\n"
        "CANDIDATE_START\npolicy text\nCANDIDATE_END\n"
    )
    payload = json.loads(demo_completion(prompt))
    assert payload["scores"]["faithfulness"] == 5


def test_http_client_parses_usage_and_sends_the_key() -> None:
    seen: dict[str, str] = {}

    def transport(request: urllib.request.Request) -> bytes:
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization") or ""
        return json.dumps(
            {
                "model": "gpt-test",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        ).encode("utf-8")

    client = HttpChatLLM(api_key="secret", transport=transport)
    response = client.complete("hi", model="gpt-test")
    assert response.text == "hello"
    assert response.input_tokens == 3
    assert response.output_tokens == 1
    assert seen["auth"] == "Bearer secret"
    assert seen["url"].endswith("/chat/completions")


def test_http_client_does_not_invent_token_counts() -> None:
    def transport(request: urllib.request.Request) -> bytes:
        return json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode("utf-8")

    client = HttpChatLLM(api_key="secret", transport=transport)
    with pytest.raises(LLMError, match="usage"):
        client.complete("hi", model="gpt-test")


def test_http_client_wraps_http_errors() -> None:
    def transport(request: urllib.request.Request) -> bytes:
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            EmailMessage(),
            None,
        )

    client = HttpChatLLM(api_key="secret", transport=transport)
    with pytest.raises(LLMError, match="HTTP 401"):
        client.complete("hi", model="gpt-test")


def test_empty_api_key_is_rejected() -> None:
    with pytest.raises(LLMError, match="api_key"):
        HttpChatLLM(api_key="  ")
