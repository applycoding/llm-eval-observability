"""Small local dashboard. Install the dashboard extra, then run from the repo root:

streamlit run src/llm_eval_obs/dashboard/app.py
"""

import json
from pathlib import Path
from typing import Protocol

import streamlit as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_eval_obs.dashboard.view_model import case_table, delta_table, metric_cards
from llm_eval_obs.errors import EvalObsError
from llm_eval_obs.llm import FakeLLM
from llm_eval_obs.prompts import PromptRegistry
from llm_eval_obs.report import HONESTY_NOTE, render_html, render_markdown
from llm_eval_obs.serialize import run_from_dict
from llm_eval_obs.service import run_prompt_version
from llm_eval_obs.tracing import add_span_exporter, build_provider, span_rows

PROMPT_ID = "support"


def main() -> None:
    st.set_page_config(page_title="LLM eval desk", layout="wide")
    st.title("LLM eval desk")
    st.caption(HONESTY_NOTE)
    st.caption(
        "Latency and token counts are whatever the client reported. "
        "The fake client reports 0 ms and counts tokens as words. "
        "Cost is 0 unless the model is in the price catalog."
    )

    if "runs" not in st.session_state:
        st.session_state.runs = {}
    if "spans" not in st.session_state:
        st.session_state.spans = []

    root = Path.cwd()
    with st.sidebar:
        st.header("Fake-model run")
        versions = _versions(root)
        selected = st.multiselect("Prompt versions", versions, default=versions)
        if st.button("Run fake model", type="primary", disabled=not selected):
            _run_versions(root, selected)
        st.header("Saved run")
        upload = st.file_uploader("Eval JSON", type="json", accept_multiple_files=True)

    if upload:
        _load_uploads(upload)

    runs = st.session_state.runs
    if not runs:
        st.info("Run the fake model from the repo root, or upload a JSON eval run.")
        return

    names = list(runs)
    chosen = st.selectbox("Run", names)
    run = runs[chosen]
    st.subheader(f"{run.prompt_id} version {run.prompt_version}")
    if run.prompt_description:
        st.write(run.prompt_description)
    _metrics(metric_cards(run))
    st.dataframe(case_table(run), use_container_width=True, hide_index=True)

    if len(names) >= 2:
        st.subheader("Compare")
        left, right = st.columns(2)
        before_name = left.selectbox("Before", names, index=0)
        after_name = right.selectbox("After", names, index=min(1, len(names) - 1))
        before = runs[before_name]
        after = runs[after_name]
        st.dataframe(delta_table(before, after), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Markdown",
            render_markdown(before, after),
            file_name="comparison.md",
        )
        st.download_button(
            "Download HTML",
            render_html(before, after),
            file_name="comparison.html",
        )

    if st.session_state.spans:
        st.subheader("Spans captured in this session")
        st.dataframe(span_rows(st.session_state.spans), use_container_width=True, hide_index=True)


def _versions(root: Path) -> list[str]:
    directory = root / "prompts" / "support"
    if not directory.is_dir():
        return []
    try:
        return list(PromptRegistry.load(directory).versions(PROMPT_ID))
    except EvalObsError:
        return []


def _run_versions(root: Path, versions: list[str]) -> None:
    provider = build_provider()
    exporter = InMemorySpanExporter()
    add_span_exporter(provider, exporter)
    tracer = provider.get_tracer("llm_eval_obs")
    try:
        for version in versions:
            run = run_prompt_version(
                root=root,
                version=version,
                llm=FakeLLM(),
                tracer=tracer,
            )
            st.session_state.runs[f"{run.prompt_id}@{run.prompt_version}"] = run
    except EvalObsError as exc:
        st.error(str(exc))
        return
    finally:
        provider.force_flush()
    st.session_state.spans = list(exporter.get_finished_spans())


class _Upload(Protocol):
    name: str

    def getvalue(self) -> bytes:
        """Return the uploaded bytes."""


def _load_uploads(uploads: list[_Upload]) -> None:
    for item in uploads:
        try:
            run = run_from_dict(json.loads(item.getvalue().decode("utf-8")))
        except (EvalObsError, UnicodeError, ValueError) as exc:
            st.error(f"{item.name}: {exc}")
            continue
        st.session_state.runs[item.name] = run


def _metrics(cards: list[tuple[str, str]]) -> None:
    columns = st.columns(min(len(cards), 4) or 1)
    for index, (label, value) in enumerate(cards):
        columns[index % len(columns)].metric(label, value)


if __name__ == "__main__":
    main()
