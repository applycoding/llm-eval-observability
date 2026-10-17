"""The checked-in report and baseline are outputs of the fake model."""

import json

import pytest

from llm_eval_obs.artifacts import (
    BASELINE_NOTE,
    BASELINE_PATH,
    HTML_PATH,
    MARKDOWN_PATH,
    write_demo_artifacts,
)
from llm_eval_obs.report import HONESTY_NOTE
from tests.factories import ROOT


def test_sample_report_matches_a_fresh_fake_run(tmp_path) -> None:
    write_demo_artifacts(ROOT, tmp_path)
    for relative in (MARKDOWN_PATH, HTML_PATH):
        assert (tmp_path / relative).read_text(encoding="utf-8") == (ROOT / relative).read_text(
            encoding="utf-8"
        )
    fresh = json.loads((tmp_path / BASELINE_PATH).read_text(encoding="utf-8"))
    checked_in = json.loads((ROOT / BASELINE_PATH).read_text(encoding="utf-8"))
    assert fresh["note"] == BASELINE_NOTE
    assert checked_in["note"] == BASELINE_NOTE
    assert fresh["prompt_version"] == checked_in["prompt_version"] == "2"
    assert fresh["scorer_means"].keys() == checked_in["scorer_means"].keys()
    for name, value in fresh["scorer_means"].items():
        assert checked_in["scorer_means"][name] == pytest.approx(value, abs=1e-6)
    assert HONESTY_NOTE in (ROOT / MARKDOWN_PATH).read_text(encoding="utf-8")
