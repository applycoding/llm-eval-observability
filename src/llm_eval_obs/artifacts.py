"""Rewrite the checked-in baseline and sample report from the fake model.

These files are outputs of this function. The artifact test fails if they drift,
so a score in the sample report is something the fake model just produced.
"""

from pathlib import Path

from llm_eval_obs.report import render_html, render_markdown
from llm_eval_obs.serialize import baseline_from_run, baseline_to_dict, dump_json
from llm_eval_obs.service import run_prompt_version
from llm_eval_obs.tracing import build_provider

BASELINE_NOTE = "Scores from the checked-in fake model on this dataset. Not a published benchmark."
BASELINE_PATH = Path("baselines/support_v2.json")
MARKDOWN_PATH = Path("reports/support_v1_vs_v2.md")
HTML_PATH = Path("reports/support_v1_vs_v2.html")


def write_demo_artifacts(root: Path, destination: Path | None = None) -> None:
    dest = root if destination is None else destination
    before = run_prompt_version(root=root, version="1", tracer=_tracer())
    after = run_prompt_version(root=root, version="2", tracer=_tracer())
    baseline = baseline_from_run(after, BASELINE_NOTE)
    dump_json(dest / BASELINE_PATH, baseline_to_dict(baseline))
    _write(dest / MARKDOWN_PATH, render_markdown(before, after))
    _write(dest / HTML_PATH, render_html(before, after))


def _tracer():
    return build_provider().get_tracer("llm_eval_obs")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
