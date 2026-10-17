"""Markdown and HTML comparisons of two eval runs."""

import html
from dataclasses import dataclass

from llm_eval_obs.aggregate import judge_usage, scorer_means, task_usage
from llm_eval_obs.models import EvalRun

HONESTY_NOTE = (
    "Figures in this report are the scores from the runs being compared. "
    "They are not a published benchmark and they do not describe a hosted model "
    "unless that model produced the runs."
)


@dataclass(frozen=True)
class ScoreRow:
    case_id: str
    scorer: str
    before: float | None
    after: float | None


@dataclass(frozen=True)
class MeanRow:
    scorer: str
    before: float | None
    after: float | None


def mean_rows(before: EvalRun, after: EvalRun) -> tuple[MeanRow, ...]:
    left = scorer_means(before)
    right = scorer_means(after)
    rows = [
        MeanRow(name, left.get(name), right.get(name))
        for name in _scorer_names(before, after)
        if name in left or name in right
    ]
    return tuple(rows)


def case_score_rows(before: EvalRun, after: EvalRun) -> tuple[ScoreRow, ...]:
    left = _score_index(before)
    right = _score_index(after)
    rows: list[ScoreRow] = []
    for case_id in _case_ids(before, after):
        for name in _scorer_names(before, after):
            if (case_id, name) not in left and (case_id, name) not in right:
                continue
            rows.append(
                ScoreRow(
                    case_id=case_id,
                    scorer=name,
                    before=left.get((case_id, name)),
                    after=right.get((case_id, name)),
                )
            )
    return tuple(rows)


def render_markdown(before: EvalRun, after: EvalRun) -> str:
    lines = [
        "# Prompt comparison",
        "",
        (
            f"`{before.prompt_id}` version {before.prompt_version} "
            f"compared with version {after.prompt_version}."
        ),
        "",
        (f"Suite `{before.suite_id}`. Dataset `{before.dataset}`. Model `{before.model}`."),
        "",
        HONESTY_NOTE,
        "",
        "## Scorer means",
        "",
        _markdown_table(
            ["Scorer", f"v{before.prompt_version}", f"v{after.prompt_version}", "Delta"],
            [
                [
                    row.scorer,
                    format_score(row.before),
                    format_score(row.after),
                    format_delta(row.before, row.after),
                ]
                for row in mean_rows(before, after)
            ],
        ),
        "",
        "## Case scores",
        "",
        _markdown_table(
            ["Case", "Scorer", f"v{before.prompt_version}", f"v{after.prompt_version}", "Delta"],
            [
                [
                    row.case_id,
                    row.scorer,
                    format_score(row.before),
                    format_score(row.after),
                    format_delta(row.before, row.after),
                ]
                for row in case_score_rows(before, after)
            ],
        ),
        "",
        "## Usage",
        "",
        _usage_markdown(before, after),
        "",
        "## Outputs",
        "",
    ]
    lines.extend(_output_markdown(before, after))
    lines.append("")
    return "\n".join(lines)


def render_html(before: EvalRun, after: EvalRun) -> str:
    means = "\n".join(_mean_html_row(row) for row in mean_rows(before, after))
    cases = "\n".join(_case_html_row(row) for row in case_score_rows(before, after))
    title = html.escape(f"{before.prompt_id} v{before.prompt_version} → v{after.prompt_version}")
    meta = html.escape(f"Suite {before.suite_id}. Dataset {before.dataset}. Model {before.model}.")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{
      margin: 0;
      background: #f4f1ea;
      color: #1c1917;
      font: 16px/1.5 "Iowan Old Style", "Palatino Linotype", Palatino, serif;
    }}
    main {{ max-width: 960px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
    h1 {{ font-size: 2rem; letter-spacing: -0.03em; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
    .meta, .note {{ color: #44403c; }}
    .note {{
      background: #fff;
      border-left: 4px solid #0f4c5c;
      padding: 0.75rem 1rem;
    }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid #e7e5e4; }}
    th {{ font: 600 0.78rem/1.2 ui-sans-serif, system-ui, sans-serif; letter-spacing: 0.04em; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .up {{ color: #1b6b3a; }}
    .down {{ color: #9a3412; }}
    pre {{
      white-space: pre-wrap;
      background: #fff;
      padding: 0.75rem 1rem;
      border: 1px solid #e7e5e4;
    }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="meta">{meta}</p>
  <p class="note">{html.escape(HONESTY_NOTE)}</p>
  <h2>Scorer means</h2>
  <table>
    <thead>
      <tr>
        <th>Scorer</th>
        <th class="num">v{html.escape(before.prompt_version)}</th>
        <th class="num">v{html.escape(after.prompt_version)}</th>
        <th class="num">Delta</th>
      </tr>
    </thead>
    <tbody>
{means}
    </tbody>
  </table>
  <h2>Case scores</h2>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Scorer</th>
        <th class="num">v{html.escape(before.prompt_version)}</th>
        <th class="num">v{html.escape(after.prompt_version)}</th>
        <th class="num">Delta</th>
      </tr>
    </thead>
    <tbody>
{cases}
    </tbody>
  </table>
  <h2>Usage</h2>
  {_usage_html(before, after)}
  <h2>Outputs</h2>
  {_outputs_html(before, after)}
</main>
</body>
</html>
"""


def format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "n/a"
    delta = after - before
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    text_columns = 2 if headers[:2] == ["Case", "Scorer"] else 1
    head = "| " + " | ".join(headers) + " |"
    align = (
        "| "
        + " | ".join("---" if index < text_columns else "---:" for index in range(len(headers)))
        + " |"
    )
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, align, *body])


def _usage_markdown(before: EvalRun, after: EvalRun) -> str:
    rows = [_usage_cells(before), _usage_cells(after)]
    return _markdown_table(
        ["Version", "Task tokens", "Judge tokens", "Latency ms", "Cost USD", "Cost known"],
        rows,
    )


def _usage_cells(run: EvalRun) -> list[str]:
    task = task_usage(run)
    judge = judge_usage(run)
    return [
        run.prompt_version,
        str(task.input_tokens + task.output_tokens),
        str(judge.input_tokens + judge.output_tokens),
        f"{(task.latency_ms + judge.latency_ms):.1f}",
        f"{(task.cost_usd + judge.cost_usd):.6f}",
        "yes" if task.cost_known and judge.cost_known else "no",
    ]


def _output_markdown(before: EvalRun, after: EvalRun) -> list[str]:
    lines: list[str] = []
    left = {case.case_id: case for case in before.cases}
    right = {case.case_id: case for case in after.cases}
    for case_id in _case_ids(before, after):
        lines.append(f"### {case_id}")
        lines.append("")
        for label, found in ((before.prompt_version, left), (after.prompt_version, right)):
            case = found.get(case_id)
            lines.append(f"Version {label}")
            lines.append("")
            if case is None:
                lines.append("    (missing)")
            else:
                if case.error:
                    lines.append(f"Error: {case.error}")
                    lines.append("")
                lines.append(_indent(case.output))
            lines.append("")
    return lines


def _mean_html_row(row: MeanRow) -> str:
    return (
        "      <tr>"
        f"<td>{html.escape(row.scorer)}</td>"
        f'<td class="num">{format_score(row.before)}</td>'
        f'<td class="num">{format_score(row.after)}</td>'
        f"{_delta_cell(row.before, row.after)}"
        "</tr>"
    )


def _case_html_row(row: ScoreRow) -> str:
    return (
        "      <tr>"
        f"<td>{html.escape(row.case_id)}</td>"
        f"<td>{html.escape(row.scorer)}</td>"
        f'<td class="num">{format_score(row.before)}</td>'
        f'<td class="num">{format_score(row.after)}</td>'
        f"{_delta_cell(row.before, row.after)}"
        "</tr>"
    )


def _delta_cell(before: float | None, after: float | None) -> str:
    kind = _delta_class(before, after)
    css = "num" if not kind else f"num {kind}"
    return f'<td class="{css}">{format_delta(before, after)}</td>'


def _usage_html(before: EvalRun, after: EvalRun) -> str:
    rows = "\n".join(
        "      <tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"
        for cells in (_usage_cells(before), _usage_cells(after))
    )
    return (
        "<table><thead><tr>"
        "<th>Version</th><th>Task tokens</th><th>Judge tokens</th>"
        "<th>Latency ms</th><th>Cost USD</th><th>Cost known</th>"
        f"</tr></thead><tbody>\n{rows}\n    </tbody></table>"
    )


def _outputs_html(before: EvalRun, after: EvalRun) -> str:
    left = {case.case_id: case for case in before.cases}
    right = {case.case_id: case for case in after.cases}
    blocks: list[str] = []
    for case_id in _case_ids(before, after):
        parts = [f"<h3>{html.escape(case_id)}</h3>"]
        for version, found in ((before.prompt_version, left), (after.prompt_version, right)):
            case = found.get(case_id)
            parts.append(f"<p>Version {html.escape(version)}</p>")
            if case is None:
                parts.append("<pre>(missing)</pre>")
                continue
            if case.error:
                parts.append(f"<p>Error: {html.escape(case.error)}</p>")
            parts.append(f"<pre>{html.escape(case.output)}</pre>")
        blocks.append("\n".join(parts))
    return "\n".join(blocks)


def _score_index(run: EvalRun) -> dict[tuple[str, str], float | None]:
    indexed: dict[tuple[str, str], float | None] = {}
    for case in run.cases:
        for score in case.scores:
            indexed[(case.case_id, score.name)] = None if score.skipped else score.value
    return indexed


def _case_ids(before: EvalRun, after: EvalRun) -> list[str]:
    seen: list[str] = []
    for run in (before, after):
        for case in run.cases:
            if case.case_id not in seen:
                seen.append(case.case_id)
    return seen


def _scorer_names(before: EvalRun, after: EvalRun) -> list[str]:
    names: list[str] = []
    for run in (after, before):
        for case in run.cases:
            for score in case.scores:
                if score.name not in names:
                    names.append(score.name)
    return names


def _delta_class(before: float | None, after: float | None) -> str:
    if before is None or after is None or before == after:
        return ""
    return "up" if after > before else "down"


def _indent(text: str) -> str:
    if not text:
        return "    (empty)"
    return "\n".join(f"    {line}" for line in text.splitlines())
