"""CLI exit codes for the gate and the report writer."""

import json
from pathlib import Path

from llm_eval_obs.cli import main
from tests.factories import ROOT


def test_gate_cli_passes_the_checked_in_baseline(capsys) -> None:
    code = main(["gate", "--root", str(ROOT), "--tolerance", "0", "--no-langfuse"])
    captured = capsys.readouterr()
    assert code == 0
    assert "gate passed" in captured.out


def test_gate_cli_fails_when_the_baseline_is_above_every_scorer(tmp_path, capsys) -> None:
    baseline = {
        "note": "fixture ceiling used to force a drop; not a model score",
        "suite_id": "support_regression",
        "prompt_id": "support",
        "prompt_version": "2",
        "dataset": "datasets/support.jsonl",
        "model": "fake-llm",
        "scorer_means": {
            "exact_reply": 2.0,
            "policy_regex": 2.0,
            "reference_similarity": 2.0,
            "rubric_judge": 2.0,
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    code = main(
        ["gate", "--root", str(ROOT), "--baseline", str(path), "--tolerance", "0", "--no-langfuse"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "dropped" in captured.err


def test_report_cli_writes_html(tmp_path: Path) -> None:
    out_dir = tmp_path / "runs"
    out_dir.mkdir()
    before = out_dir / "v1.json"
    after = out_dir / "v2.json"
    v1 = main(
        ["eval", "--root", str(ROOT), "--version", "1", "--out", str(before), "--no-langfuse"]
    )
    v2 = main(["eval", "--root", str(ROOT), "--version", "2", "--out", str(after), "--no-langfuse"])
    assert v1 == 0
    assert v2 == 0
    destination = tmp_path / "report.html"
    code = main(
        [
            "report",
            "--before",
            str(before),
            "--after",
            str(after),
            "--format",
            "html",
            "--out",
            str(destination),
        ]
    )
    assert code == 0
    text = destination.read_text(encoding="utf-8")
    assert "not a published benchmark" in text
    assert "<html" in text


def test_ci_workflow_lints_tests_and_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "ruff check" in workflow
    assert "pytest" in workflow
    assert "llm_eval_obs gate" in workflow
