"""Score means and token rollups shared by the gate, report, and dashboard."""

from llm_eval_obs.models import EvalRun, Usage

FLOAT_PLACES = 6


def scorer_means(run: EvalRun) -> dict[str, float]:
    """Mean of applicable scores. Skipped scores are left out. Errors are not skipped.

    A case error still carries numeric scores (zeros for scorers that applied),
    so a failing call pulls the mean down and the gate can see it.
    """

    buckets: dict[str, list[float]] = {}
    for case in run.cases:
        for score in case.scores:
            if score.skipped or score.value is None:
                continue
            buckets.setdefault(score.name, []).append(score.value)
    return {
        name: round(sum(values) / len(values), FLOAT_PLACES)
        for name, values in sorted(buckets.items())
    }


def sum_usage(usages: list[Usage]) -> Usage:
    if not usages:
        return Usage.zero()
    return Usage(
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        latency_ms=round(sum(item.latency_ms for item in usages), FLOAT_PLACES),
        cost_usd=round(sum(item.cost_usd for item in usages), 10),
        cost_known=all(item.cost_known for item in usages),
    )


def task_usage(run: EvalRun) -> Usage:
    return sum_usage([case.task for case in run.cases])


def judge_usage(run: EvalRun) -> Usage:
    return sum_usage([case.judge for case in run.cases if case.judge is not None])
