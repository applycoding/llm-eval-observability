"""Shape eval runs into rows the Streamlit app can render."""

from llm_eval_obs.aggregate import judge_usage, scorer_means, task_usage
from llm_eval_obs.models import EvalRun
from llm_eval_obs.report import format_delta, format_score, mean_rows


def metric_cards(run: EvalRun) -> list[tuple[str, str]]:
    task = task_usage(run)
    judge = judge_usage(run)
    cost = task.cost_usd + judge.cost_usd
    known = task.cost_known and judge.cost_known
    cards = [("cases", str(len(run.cases)))]
    cards.extend((name, format_score(value)) for name, value in scorer_means(run).items())
    cards.append(("task tokens", str(task.input_tokens + task.output_tokens)))
    cards.append(("judge tokens", str(judge.input_tokens + judge.output_tokens)))
    cards.append(("latency ms", f"{task.latency_ms + judge.latency_ms:.1f}"))
    cost_text = f"{cost:.6f}" if known else f"{cost:.6f} (incomplete)"
    cards.append(("cost usd", cost_text))
    return cards


def case_table(run: EvalRun) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in run.cases:
        for score in case.scores:
            rows.append(
                {
                    "case": case.case_id,
                    "scorer": score.name,
                    "score": "skipped" if score.skipped else format_score(score.value),
                    "passed": "" if score.passed is None else str(score.passed),
                    "detail": score.detail,
                    "error": case.error or "",
                }
            )
    return rows


def delta_table(before: EvalRun, after: EvalRun) -> list[dict[str, str]]:
    return [
        {
            "scorer": row.scorer,
            "before": format_score(row.before),
            "after": format_score(row.after),
            "delta": format_delta(row.before, row.after),
        }
        for row in mean_rows(before, after)
    ]
