"""JSON shape for eval runs and baselines."""

import json
from pathlib import Path
from typing import Any

from llm_eval_obs.aggregate import scorer_means
from llm_eval_obs.errors import RunError
from llm_eval_obs.models import Baseline, CaseResult, EvalRun, Score, Usage


def run_to_dict(run: EvalRun) -> dict[str, Any]:
    return {
        "prompt_id": run.prompt_id,
        "prompt_version": run.prompt_version,
        "prompt_description": run.prompt_description,
        "dataset": run.dataset,
        "model": run.model,
        "suite_id": run.suite_id,
        "cases": [_case_to_dict(case) for case in run.cases],
    }


def run_from_dict(data: object) -> EvalRun:
    body = _mapping(data, "run")
    return EvalRun(
        prompt_id=_text(body, "prompt_id"),
        prompt_version=_text(body, "prompt_version"),
        prompt_description=_text(body, "prompt_description"),
        dataset=_text(body, "dataset"),
        model=_text(body, "model"),
        suite_id=_text(body, "suite_id"),
        cases=tuple(_case_from_dict(item) for item in _list(body, "cases")),
    )


def baseline_to_dict(baseline: Baseline) -> dict[str, Any]:
    return {
        "note": baseline.note,
        "suite_id": baseline.suite_id,
        "prompt_id": baseline.prompt_id,
        "prompt_version": baseline.prompt_version,
        "dataset": baseline.dataset,
        "model": baseline.model,
        "scorer_means": baseline.scorer_means,
    }


def baseline_from_dict(data: object) -> Baseline:
    body = _mapping(data, "baseline")
    means_raw = body.get("scorer_means")
    if not isinstance(means_raw, dict) or not means_raw:
        raise RunError("baseline scorer_means must be a non-empty object")
    means: dict[str, float] = {}
    for name, value in means_raw.items():
        if not isinstance(name, str):
            raise RunError("baseline scorer names must be strings")
        means[name] = _number(value, f"scorer_means.{name}")
    return Baseline(
        prompt_id=_text(body, "prompt_id"),
        prompt_version=_text(body, "prompt_version"),
        dataset=_text(body, "dataset"),
        model=_text(body, "model"),
        suite_id=_text(body, "suite_id"),
        scorer_means=means,
        note=_text(body, "note"),
    )


def baseline_from_run(run: EvalRun, note: str) -> Baseline:
    means = scorer_means(run)
    if not means:
        raise RunError("cannot freeze a baseline from a run with no scores")
    return Baseline(
        prompt_id=run.prompt_id,
        prompt_version=run.prompt_version,
        dataset=run.dataset,
        model=run.model,
        suite_id=run.suite_id,
        scorer_means=means,
        note=note,
    )


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunError(f"could not read {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RunError(f"invalid JSON in {path}: {exc}") from exc


def load_run(path: Path) -> EvalRun:
    return run_from_dict(load_json(path))


def load_baseline(path: Path) -> Baseline:
    return baseline_from_dict(load_json(path))


def _case_to_dict(case: CaseResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "output": case.output,
        "error": case.error,
        "scores": [_score_to_dict(score) for score in case.scores],
        "task": _usage_to_dict(case.task),
        "judge": None if case.judge is None else _usage_to_dict(case.judge),
    }


def _case_from_dict(data: object) -> CaseResult:
    body = _mapping(data, "case")
    judge = body.get("judge")
    return CaseResult(
        case_id=_text(body, "case_id"),
        output=_text(body, "output", allow_empty=True),
        scores=tuple(_score_from_dict(item) for item in _list(body, "scores")),
        task=_usage_from_dict(body.get("task")),
        judge=None if judge is None else _usage_from_dict(judge),
        error=_optional_text(body.get("error"), "error"),
    )


def _score_to_dict(score: Score) -> dict[str, Any]:
    return {
        "name": score.name,
        "value": score.value,
        "passed": score.passed,
        "detail": score.detail,
        "skipped": score.skipped,
    }


def _score_from_dict(data: object) -> Score:
    body = _mapping(data, "score")
    value = body.get("value")
    passed = body.get("passed")
    return Score(
        name=_text(body, "name"),
        value=None if value is None else _number(value, "score.value"),
        passed=None if passed is None else _boolean(passed, "score.passed"),
        detail=_text(body, "detail", allow_empty=True),
        skipped=_boolean(body.get("skipped"), "score.skipped"),
    )


def _usage_to_dict(usage: Usage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_ms": usage.latency_ms,
        "cost_usd": usage.cost_usd,
        "cost_known": usage.cost_known,
    }


def _usage_from_dict(data: object) -> Usage:
    body = _mapping(data, "usage")
    return Usage(
        input_tokens=_integer(body.get("input_tokens"), "input_tokens"),
        output_tokens=_integer(body.get("output_tokens"), "output_tokens"),
        latency_ms=_number(body.get("latency_ms"), "latency_ms"),
        cost_usd=_number(body.get("cost_usd"), "cost_usd"),
        cost_known=_boolean(body.get("cost_known"), "cost_known"),
    )


def _mapping(data: object, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RunError(f"{label} must be a JSON object")
    return data


def _list(body: dict[str, Any], key: str) -> list[object]:
    value = body.get(key)
    if not isinstance(value, list):
        raise RunError(f"{key} must be a list")
    return value


def _text(body: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
    value = body.get(key)
    if not isinstance(value, str) or (not allow_empty and value == ""):
        raise RunError(f"{key} must be a string")
    return value


def _optional_text(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RunError(f"{key} must be a string or null")
    return value


def _number(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunError(f"{key} must be a number")
    return float(value)


def _integer(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunError(f"{key} must be an integer")
    return value


def _boolean(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise RunError(f"{key} must be a boolean")
    return value
