"""End-to-end runs against the fake model and the checked-in suite."""

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_eval_obs.aggregate import scorer_means
from llm_eval_obs.cost import ModelPrice
from llm_eval_obs.errors import LLMError
from llm_eval_obs.models import LLMResponse, PromptTemplate, TestCase
from llm_eval_obs.runner import evaluate
from llm_eval_obs.scoring import ExactScorer
from llm_eval_obs.service import run_prompt_version
from llm_eval_obs.suite import ExactSpec
from llm_eval_obs.tracing import add_span_exporter, build_provider
from tests.factories import ROOT


def test_policy_prompt_scores_at_least_as_high_as_the_generic_prompt() -> None:
    before = run_prompt_version(root=ROOT, version="1")
    after = run_prompt_version(root=ROOT, version="2")
    left = scorer_means(before)
    right = scorer_means(after)
    assert right["policy_regex"] > left["policy_regex"]
    assert right["exact_reply"] > left["exact_reply"]
    assert right["reference_similarity"] >= left["reference_similarity"]
    assert right["rubric_judge"] >= left["rubric_judge"]


def test_full_run_emits_one_task_span_and_one_judge_span_per_case() -> None:
    provider = build_provider()
    exporter = InMemorySpanExporter()
    add_span_exporter(provider, exporter)
    run = run_prompt_version(root=ROOT, version="2", tracer=provider.get_tracer("test"))
    spans = exporter.get_finished_spans()
    assert len(run.cases) == 5
    assert len(spans) == 10
    roles = [(span.attributes or {})["llm_eval_obs.role"] for span in spans]
    assert roles.count("task") == 5
    assert roles.count("judge") == 5
    assert all(case.error is None for case in run.cases)
    assert all(case.task.cost_known for case in run.cases)
    assert all(case.task.cost_usd == 0.0 for case in run.cases)


def test_task_failure_zeros_applicable_scores_and_marks_the_span() -> None:
    class Down:
        def complete(self, prompt: str, *, model: str) -> LLMResponse:
            raise LLMError("down")

    provider = build_provider()
    exporter = InMemorySpanExporter()
    add_span_exporter(provider, exporter)
    case = TestCase("c", {"name": "Ada"}, expected="hello")
    run = evaluate(
        prompt=PromptTemplate("p", "1", "Hello {{ name }}"),
        cases=[case],
        scorers=[ExactScorer(ExactSpec(type="exact", name="exact_reply"))],
        llm=Down(),
        model="example-model",
        catalog={"example-model": ModelPrice(3, 15)},
        tracer=provider.get_tracer("test"),
        dataset_name="memory",
        suite_id="unit",
    )
    assert run.cases[0].error == "down"
    assert run.cases[0].scores[0].value == 0.0
    assert span_rows_status(exporter) == "ERROR"


def test_example_catalog_cost_is_recorded_on_the_span() -> None:
    class Fixed:
        def complete(self, prompt: str, *, model: str) -> LLMResponse:
            return LLMResponse("hello", model, 1000, 500, 12.5)

    provider = build_provider()
    exporter = InMemorySpanExporter()
    add_span_exporter(provider, exporter)
    evaluate(
        prompt=PromptTemplate("p", "1", "Hello {{ name }}"),
        cases=[TestCase("c", {"name": "Ada"}, expected="hello")],
        scorers=[ExactScorer(ExactSpec(type="exact", name="exact_reply"))],
        llm=Fixed(),
        model="example-model",
        catalog={"example-model": ModelPrice(3, 15)},
        tracer=provider.get_tracer("test"),
        dataset_name="memory",
        suite_id="unit",
    )
    attributes = exporter.get_finished_spans()[0].attributes or {}
    assert attributes["llm_eval_obs.cost.usd"] == 0.0105
    assert attributes["gen_ai.usage.input_tokens"] == 1000
    assert attributes["llm_eval_obs.latency_ms"] == 12.5


def span_rows_status(exporter: InMemorySpanExporter) -> str:
    return exporter.get_finished_spans()[0].status.status_code.name
