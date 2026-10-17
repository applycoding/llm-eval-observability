"""Data records passed between the runner, scorers, report, and gate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: str
    template: str
    description: str = ""


@dataclass(frozen=True)
class TestCase:
    id: str
    inputs: dict[str, str]
    expected: str | None = None
    expected_regex: str | None = None
    reference: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class Score:
    name: str
    value: float | None
    passed: bool | None
    detail: str
    skipped: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cost_known: bool

    @classmethod
    def zero(cls) -> "Usage":
        return cls(0, 0, 0.0, 0.0, True)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    output: str
    scores: tuple[Score, ...]
    task: Usage
    judge: Usage | None
    error: str | None = None


@dataclass(frozen=True)
class EvalRun:
    prompt_id: str
    prompt_version: str
    prompt_description: str
    dataset: str
    model: str
    suite_id: str
    cases: tuple[CaseResult, ...]


@dataclass(frozen=True)
class Baseline:
    prompt_id: str
    prompt_version: str
    dataset: str
    model: str
    suite_id: str
    scorer_means: dict[str, float]
    note: str
