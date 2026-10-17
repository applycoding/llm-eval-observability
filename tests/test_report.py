"""Markdown and HTML comparison reports."""

from llm_eval_obs.report import HONESTY_NOTE, render_html, render_markdown
from tests.factories import make_case, make_run, make_score


def test_report_shows_the_delta_and_the_honesty_note() -> None:
    before = make_run([make_case("refund-window", [make_score("policy_regex", 0.0)])], version="1")
    after = make_run([make_case("refund-window", [make_score("policy_regex", 1.0)])], version="2")
    markdown = render_markdown(before, after)
    html = render_html(before, after)
    assert HONESTY_NOTE in markdown
    assert HONESTY_NOTE in html
    assert "policy_regex" in markdown
    assert "+1.0000" in markdown
    assert "not a published benchmark" in markdown


def test_html_escapes_model_output() -> None:
    before = make_run(
        [make_case("c", [make_score("policy_regex", 0.0)], output="<script>alert(1)</script>")],
        version="1",
    )
    after = make_run(
        [make_case("c", [make_score("policy_regex", 1.0)], output="ok")],
        version="2",
    )
    html = render_html(before, after)
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
