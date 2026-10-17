"""Run one prompt version over a dataset and score each case."""

from collections.abc import Sequence

from opentelemetry.trace import Tracer

from llm_eval_obs.aggregate import sum_usage
from llm_eval_obs.cost import ModelPrice, estimate_cost
from llm_eval_obs.errors import LLMError, PromptError
from llm_eval_obs.llm import LLMClient
from llm_eval_obs.models import CaseResult, EvalRun, PromptTemplate, TestCase, Usage
from llm_eval_obs.prompts import render_prompt
from llm_eval_obs.scoring import ScoreContext, Scorer, failed_score, skipped_score
from llm_eval_obs.tracing import record_llm_span


def evaluate(
    *,
    prompt: PromptTemplate,
    cases: Sequence[TestCase],
    scorers: Sequence[Scorer],
    llm: LLMClient,
    model: str,
    catalog: dict[str, ModelPrice],
    tracer: Tracer,
    dataset_name: str,
    suite_id: str,
) -> EvalRun:
    results = [
        _evaluate_case(
            prompt=prompt,
            case=case,
            scorers=scorers,
            llm=llm,
            model=model,
            catalog=catalog,
            tracer=tracer,
        )
        for case in cases
    ]
    return EvalRun(
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        prompt_description=prompt.description,
        dataset=dataset_name,
        model=model,
        suite_id=suite_id,
        cases=tuple(results),
    )


def _evaluate_case(
    *,
    prompt: PromptTemplate,
    case: TestCase,
    scorers: Sequence[Scorer],
    llm: LLMClient,
    model: str,
    catalog: dict[str, ModelPrice],
    tracer: Tracer,
) -> CaseResult:
    try:
        rendered = render_prompt(prompt.template, case.inputs)
    except PromptError as exc:
        return _failed(case, scorers, str(exc), usage=Usage.zero())

    try:
        response = llm.complete(rendered, model=model)
    except LLMError as exc:
        record_llm_span(
            tracer,
            model=model,
            prompt_text=rendered,
            completion=None,
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            prompt_id=prompt.id,
            prompt_version=prompt.version,
            case_id=case.id,
            role="task",
            cost_usd=0.0,
            cost_known=model in catalog,
            error=str(exc),
        )
        return _failed(
            case,
            scorers,
            str(exc),
            usage=Usage(0, 0, 0.0, 0.0, model in catalog),
        )

    cost_usd, cost_known = estimate_cost(
        response.model, response.input_tokens, response.output_tokens, catalog
    )
    record_llm_span(
        tracer,
        model=response.model,
        prompt_text=rendered,
        completion=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        case_id=case.id,
        role="task",
        cost_usd=cost_usd,
        cost_known=cost_known,
    )
    ctx = ScoreContext(
        llm=llm,
        model=model,
        tracer=tracer,
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        case_id=case.id,
        catalog=catalog,
    )
    scores = []
    judge_parts: list[Usage] = []
    for scorer in scorers:
        if not scorer.applies(case):
            scores.append(skipped_score(scorer.name))
            continue
        try:
            outcome = scorer.score(case, response.text, ctx)
        except LLMError as exc:
            scores.append(failed_score(scorer.name, str(exc)))
            continue
        scores.append(outcome.score)
        if outcome.usage is not None:
            judge_parts.append(outcome.usage)
    return CaseResult(
        case_id=case.id,
        output=response.text,
        scores=tuple(scores),
        task=Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            cost_usd=cost_usd,
            cost_known=cost_known,
        ),
        judge=sum_usage(judge_parts) if judge_parts else None,
        error=None,
    )


def _failed(
    case: TestCase,
    scorers: Sequence[Scorer],
    detail: str,
    *,
    usage: Usage,
) -> CaseResult:
    scores = [
        failed_score(scorer.name, detail) if scorer.applies(case) else skipped_score(scorer.name)
        for scorer in scorers
    ]
    return CaseResult(
        case_id=case.id,
        output="",
        scores=tuple(scores),
        task=usage,
        judge=None,
        error=detail,
    )
