"""Regression scorers: exact text, regex, hashing similarity, and a rubric judge.

The embedding scorer hashes tokens into a fixed vector and takes cosine
similarity. It is a local lexical baseline so CI does not download an embedding
model. Swap in another object with the same `embed` method for a real model.
Negative cosine is clamped to 0 so every scorer reports a value on `[0, 1]`.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Protocol, assert_never

from llm_eval_obs.cost import ModelPrice, estimate_cost
from llm_eval_obs.errors import LLMError, ScorerError
from llm_eval_obs.llm import JUDGE_MARKER, LLMClient
from llm_eval_obs.models import Score, TestCase, Usage
from llm_eval_obs.rubric import Rubric, load_rubric
from llm_eval_obs.suite import EmbeddingSpec, ExactSpec, JudgeSpec, RegexSpec, ScorerSpec
from llm_eval_obs.tracing import Tracer, record_llm_span

EMBEDDING_DIM = 64


@dataclass(frozen=True)
class ScoreContext:
    llm: LLMClient
    model: str
    tracer: Tracer
    prompt_id: str
    prompt_version: str
    case_id: str
    catalog: dict[str, ModelPrice]


@dataclass(frozen=True)
class ScoreOutcome:
    score: Score
    usage: Usage | None = None


class Scorer(Protocol):
    name: str

    def applies(self, case: TestCase) -> bool:
        """False when the case has no field this scorer can use."""

    def score(self, case: TestCase, output: str, ctx: ScoreContext) -> ScoreOutcome:
        """Score one model output. LLM failures surface as LLMError."""


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return one dense vector."""


class HashingEmbedder:
    """Signed feature hashing. The same text always yields the same vector."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim < 8:
            raise ScorerError("embedding dimension must be at least 8")
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)


class ExactScorer:
    def __init__(self, spec: ExactSpec) -> None:
        self.name = spec.name
        self._casefold = spec.casefold

    def applies(self, case: TestCase) -> bool:
        return case.expected is not None

    def score(self, case: TestCase, output: str, ctx: ScoreContext) -> ScoreOutcome:
        expected = case.expected or ""
        matched = _normalize(output, self._casefold) == _normalize(expected, self._casefold)
        return ScoreOutcome(
            Score(
                name=self.name,
                value=1.0 if matched else 0.0,
                passed=matched,
                detail="exact match" if matched else "output differs from expected",
            )
        )


class RegexScorer:
    def __init__(self, spec: RegexSpec) -> None:
        self.name = spec.name
        self._flags = re.IGNORECASE if spec.ignore_case else 0

    def applies(self, case: TestCase) -> bool:
        return case.expected_regex is not None

    def score(self, case: TestCase, output: str, ctx: ScoreContext) -> ScoreOutcome:
        pattern = case.expected_regex or ""
        try:
            compiled = re.compile(pattern, self._flags)
        except re.error as exc:
            raise ScorerError(f"invalid regex on case {case.id}: {exc}") from exc
        matched = compiled.search(output) is not None
        return ScoreOutcome(
            Score(
                name=self.name,
                value=1.0 if matched else 0.0,
                passed=matched,
                detail="regex matched" if matched else f"regex did not match: {pattern}",
            )
        )


class EmbeddingScorer:
    def __init__(self, spec: EmbeddingSpec, embedder: Embedder | None = None) -> None:
        self.name = spec.name
        self._threshold = spec.threshold
        self._embedder = embedder or HashingEmbedder()

    def applies(self, case: TestCase) -> bool:
        return case.reference is not None

    def score(self, case: TestCase, output: str, ctx: ScoreContext) -> ScoreOutcome:
        reference = case.reference or ""
        similarity = cosine(self._embedder.embed(output), self._embedder.embed(reference))
        clamped = round(max(0.0, similarity), 6)
        passed = clamped >= self._threshold
        return ScoreOutcome(
            Score(
                name=self.name,
                value=clamped,
                passed=passed,
                detail=f"cosine={clamped:.6f} threshold={self._threshold:.2f}",
            )
        )


class JudgeScorer:
    def __init__(self, spec: JudgeSpec, rubric: Rubric | None = None) -> None:
        self.name = spec.name
        self._rubric = rubric or load_rubric(spec.rubric)

    def applies(self, case: TestCase) -> bool:
        return True

    def score(self, case: TestCase, output: str, ctx: ScoreContext) -> ScoreOutcome:
        prompt = judge_prompt(self._rubric, output=output, reference=case.reference)
        try:
            response = ctx.llm.complete(prompt, model=ctx.model)
        except LLMError as exc:
            cost_usd, cost_known = 0.0, False
            record_llm_span(
                ctx.tracer,
                model=ctx.model,
                prompt_text=prompt,
                completion=None,
                input_tokens=None,
                output_tokens=None,
                latency_ms=None,
                prompt_id=ctx.prompt_id,
                prompt_version=ctx.prompt_version,
                case_id=ctx.case_id,
                role="judge",
                cost_usd=cost_usd,
                cost_known=cost_known,
                error=str(exc),
            )
            raise
        cost_usd, cost_known = estimate_cost(
            response.model,
            response.input_tokens,
            response.output_tokens,
            ctx.catalog,
        )
        record_llm_span(
            ctx.tracer,
            model=response.model,
            prompt_text=prompt,
            completion=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            prompt_id=ctx.prompt_id,
            prompt_version=ctx.prompt_version,
            case_id=ctx.case_id,
            role="judge",
            cost_usd=cost_usd,
            cost_known=cost_known,
        )
        usage = Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=cost_usd,
            cost_known=cost_known,
        )
        try:
            parsed = parse_judge_payload(response.text, self._rubric)
        except ScorerError as exc:
            return ScoreOutcome(
                Score(name=self.name, value=0.0, passed=False, detail=str(exc)),
                usage,
            )
        weighted = 0.0
        for criterion in self._rubric.criteria:
            weighted += criterion.weight * self._rubric.normalized(
                criterion.id, parsed[criterion.id]
            )
        value = round(weighted, 6)
        midpoint = (self._rubric.scale_min + self._rubric.scale_max) / 2
        passed = all(parsed[item.id] >= midpoint for item in self._rubric.criteria)
        parts = [f"{item.id}={parsed[item.id]}" for item in self._rubric.criteria]
        return ScoreOutcome(
            Score(
                name=self.name,
                value=value,
                passed=passed,
                detail=", ".join(parts),
            ),
            usage,
        )


def build_scorer(spec: ScorerSpec) -> Scorer:
    match spec:
        case ExactSpec():
            return ExactScorer(spec)
        case RegexSpec():
            return RegexScorer(spec)
        case EmbeddingSpec():
            return EmbeddingScorer(spec)
        case JudgeSpec():
            return JudgeScorer(spec)
        case _ as unreachable:
            assert_never(unreachable)


def judge_prompt(rubric: Rubric, *, output: str, reference: str | None) -> str:
    criteria = "\n".join(f"- {item.id}: {item.description}" for item in rubric.criteria)
    shown_reference = reference if reference else "(none)"
    return (
        "You are grading a support reply.\n"
        f"{JUDGE_MARKER}.\n"
        "Schema: "
        '{"scores": {"<criterion>": <integer>}, "rationale": "<string>"}\n'
        f"Scale: {rubric.scale_min} to {rubric.scale_max}.\n\n"
        "RUBRIC:\n"
        f"{criteria}\n\n"
        "REFERENCE_START\n"
        f"{shown_reference}\n"
        "REFERENCE_END\n\n"
        "CANDIDATE_START\n"
        f"{output}\n"
        "CANDIDATE_END\n"
    )


def parse_judge_payload(text: str, rubric: Rubric) -> dict[str, int]:
    blob = _extract_json(text)
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ScorerError(f"judge returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("scores"), dict):
        raise ScorerError("judge JSON must contain a scores object")
    scores = payload["scores"]
    parsed: dict[str, int] = {}
    for criterion in rubric.criteria:
        value = scores.get(criterion.id)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ScorerError(f"judge score for {criterion.id} must be an integer")
        if value < rubric.scale_min or value > rubric.scale_max:
            raise ScorerError(
                f"judge score for {criterion.id} is outside {rubric.scale_min}..{rubric.scale_max}"
            )
        parsed[criterion.id] = value
    return parsed


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ScorerError("embedding vectors must be non-empty and the same length")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    value = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def skipped_score(name: str) -> Score:
    return Score(name=name, value=None, passed=None, detail="not applicable", skipped=True)


def failed_score(name: str, detail: str) -> Score:
    return Score(name=name, value=0.0, passed=False, detail=detail, skipped=False)


def _normalize(text: str, casefold: bool) -> str:
    collapsed = " ".join(text.split())
    return collapsed.casefold() if casefold else collapsed


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ScorerError("judge response did not contain a JSON object")
    return stripped[start : end + 1]
