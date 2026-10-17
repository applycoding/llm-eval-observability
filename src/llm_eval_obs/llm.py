"""Model clients.

`FakeLLM` is the default. It is a deterministic heuristic so tests and the
checked-in demo run without API keys or downloaded weights. It is not a
language model and its scores are not a benchmark.

`HttpChatLLM` calls an OpenAI-compatible chat completions endpoint. Tests pass
a transport; the toolkit does not open a socket unless a caller uses the
default transport.
"""

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol

from llm_eval_obs.errors import LLMError
from llm_eval_obs.models import LLMResponse

GENERIC_REPLY = "Thanks for reaching out. I will look into that and follow up shortly."
JUDGE_MARKER = "RESPOND WITH JSON ONLY"

_POLICY = re.compile(r"(?m)^Policy:\s*(.*)\nCustomer:\s*", re.S)
_SCALE = re.compile(r"Scale:\s*(\d+)\s+to\s+(\d+)")
_CRITERION = re.compile(r"(?m)^- ([a-z][a-z0-9_]*): ")
_WORD = re.compile(r"[a-z0-9]+")


class LLMClient(Protocol):
    def complete(self, prompt: str, *, model: str) -> LLMResponse:
        """Return the model text plus the usage the client is willing to report."""


def grounded_reply(policy: str) -> str:
    """The demo fake's reply when the rendered prompt has a Policy line."""

    compact = " ".join(policy.split())
    return f"Based on our policy: {compact} If this does not cover the question, I will escalate."


def word_tokens(text: str) -> int:
    """Offline stand-in for a tokenizer. Real clients should report provider usage."""

    return max(1, len(text.split()))


class FakeLLM:
    """Deterministic client used by tests, CI, and the sample report.

    Task prompts that contain a `Policy:` line followed by a `Customer:` line
    quote that policy. Other task prompts return a fixed generic reply.
    Judge prompts, recognized by `RESPOND WITH JSON ONLY`, return rubric JSON
    whose integer scores follow lexical overlap with the reference. The overlap
    rule is part of the demo double, not a grader you would ship.
    """

    def __init__(
        self,
        handler: Callable[[str], str] | None = None,
        *,
        latency_ms: float = 0.0,
    ) -> None:
        self.calls: list[str] = []
        self._handler = handler or demo_completion
        self.latency_ms = latency_ms

    def complete(self, prompt: str, *, model: str) -> LLMResponse:
        self.calls.append(prompt)
        text = self._handler(prompt)
        return LLMResponse(
            text=text,
            model=model,
            input_tokens=word_tokens(prompt),
            output_tokens=word_tokens(text),
            latency_ms=self.latency_ms,
        )


def demo_completion(prompt: str) -> str:
    if JUDGE_MARKER in prompt:
        return _judge_json(prompt)
    match = _POLICY.search(prompt)
    if match is not None:
        return grounded_reply(match.group(1))
    return GENERIC_REPLY


class HttpChatLLM:
    """OpenAI-compatible `/chat/completions` client with an injectable transport."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        transport: Callable[[urllib.request.Request], bytes] | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise LLMError("api_key is empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport or self._urlopen

    def complete(self, prompt: str, *, model: str) -> LLMResponse:
        body = json.dumps(
            {"model": model, "messages": [{"role": "user", "content": prompt}]}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            raw = self._transport(request)
        except urllib.error.HTTPError as exc:
            raise LLMError(f"chat completion failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"chat completion request failed: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        return _parse_chat_response(raw, fallback_model=model, latency_ms=latency_ms)

    def _urlopen(self, request: urllib.request.Request) -> bytes:
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            return response.read()


def _parse_chat_response(raw: bytes, *, fallback_model: str, latency_ms: float) -> LLMResponse:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMError("chat completion returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise LLMError("chat completion JSON must be an object")
    choices = payload.get("choices")
    usage = payload.get("usage")
    if not isinstance(choices, list) or not choices:
        raise LLMError("chat completion did not include choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise LLMError("chat completion content must be a string")
    if not isinstance(usage, dict):
        raise LLMError("chat completion did not include usage; token counts are not invented")
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if (
        isinstance(input_tokens, bool)
        or isinstance(output_tokens, bool)
        or not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
    ):
        raise LLMError("chat completion usage tokens must be integers")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        model = fallback_model
    return LLMResponse(
        text=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )


def _judge_json(prompt: str) -> str:
    scale_min, scale_max = _scale(prompt)
    criteria = _CRITERION.findall(prompt)
    reference = _between(prompt, "REFERENCE_START", "REFERENCE_END")
    candidate = _between(prompt, "CANDIDATE_START", "CANDIDATE_END")
    score = _overlap_score(candidate, reference, scale_min, scale_max)
    payload = {
        "scores": {criterion: score for criterion in criteria},
        "rationale": "FakeLLM lexical overlap between the candidate and the reference.",
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _scale(prompt: str) -> tuple[int, int]:
    match = _SCALE.search(prompt)
    if match is None:
        return 1, 5
    return int(match.group(1)), int(match.group(2))


def _between(prompt: str, start: str, end: str) -> str:
    start_at = prompt.find(start)
    end_at = prompt.find(end)
    if start_at < 0 or end_at < 0 or end_at < start_at:
        return ""
    return prompt[start_at + len(start) : end_at].strip()


def _overlap_score(candidate: str, reference: str, scale_min: int, scale_max: int) -> int:
    if scale_max < scale_min:
        return scale_min
    ref = set(_WORD.findall(reference.lower()))
    cand = set(_WORD.findall(candidate.lower()))
    if not ref:
        return scale_min
    overlap = len(ref & cand) / len(ref)
    value = scale_min + round(overlap * (scale_max - scale_min))
    return max(scale_min, min(scale_max, value))
