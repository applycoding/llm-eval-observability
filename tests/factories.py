"""Small constructors for gate, report, and dashboard tests."""

from pathlib import Path

from llm_eval_obs.models import Baseline, CaseResult, EvalRun, Score, Usage

ROOT = Path(__file__).resolve().parents[1]


def make_score(name: str, value: float | None, *, skipped: bool = False) -> Score:
    if skipped:
        return Score(name=name, value=None, passed=None, detail="not applicable", skipped=True)
    assert value is not None
    return Score(name=name, value=value, passed=value >= 0.5, detail="fixture", skipped=False)


def make_case(
    case_id: str,
    scores: list[Score],
    *,
    error: str | None = None,
    output: str = "output",
    cost_known: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output=output,
        scores=tuple(scores),
        task=Usage(4, 2, 0.0, 0.0, cost_known),
        judge=None,
        error=error,
    )


def make_run(
    cases: list[CaseResult],
    *,
    version: str = "2",
    prompt_id: str = "support",
    dataset: str = "datasets/support.jsonl",
    model: str = "fake-llm",
    suite_id: str = "support_regression",
) -> EvalRun:
    return EvalRun(
        prompt_id=prompt_id,
        prompt_version=version,
        prompt_description="",
        dataset=dataset,
        model=model,
        suite_id=suite_id,
        cases=tuple(cases),
    )


def make_baseline(means: dict[str, float], *, version: str = "2") -> Baseline:
    return Baseline(
        prompt_id="support",
        prompt_version=version,
        dataset="datasets/support.jsonl",
        model="fake-llm",
        suite_id="support_regression",
        scorer_means=means,
        note="fixture",
    )
