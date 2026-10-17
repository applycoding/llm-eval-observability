"""Score gate comparisons."""

import pytest

from llm_eval_obs.aggregate import scorer_means
from llm_eval_obs.errors import GateError
from llm_eval_obs.gate import evaluate_gate
from tests.factories import make_baseline, make_case, make_run, make_score


def test_gate_fails_when_a_mean_drops() -> None:
    run = make_run([make_case("a", [make_score("policy_regex", 0.25)])])
    result = evaluate_gate(run, make_baseline({"policy_regex": 1.0}), tolerance=0)
    assert result.passed is False
    assert any("dropped" in failure for failure in result.failures)


def test_gate_passes_when_the_score_holds_or_improves() -> None:
    held = make_run([make_case("a", [make_score("policy_regex", 0.5)])])
    assert evaluate_gate(held, make_baseline({"policy_regex": 0.5}), tolerance=0).passed
    better = make_run([make_case("a", [make_score("policy_regex", 0.75)])])
    assert evaluate_gate(better, make_baseline({"policy_regex": 0.5}), tolerance=0).passed


def test_tolerance_allows_a_small_drop_and_rejects_a_large_one() -> None:
    run = make_run([make_case("a", [make_score("policy_regex", 0.75)])])
    baseline = make_baseline({"policy_regex": 0.8})
    assert evaluate_gate(run, baseline, tolerance=0.1).passed
    assert evaluate_gate(run, baseline, tolerance=0).passed is False


def test_float_slack_does_not_hide_a_real_drop() -> None:
    run = make_run([make_case("a", [make_score("policy_regex", 0.5 - 1e-9)])])
    assert evaluate_gate(run, make_baseline({"policy_regex": 0.5}), tolerance=0).passed
    dropped = make_run([make_case("a", [make_score("policy_regex", 0.49)])])
    assert evaluate_gate(dropped, make_baseline({"policy_regex": 0.5}), tolerance=0).passed is False


def test_missing_scorer_and_metadata_mismatch_fail() -> None:
    run = make_run([make_case("a", [make_score("policy_regex", 1.0)])])
    missing = evaluate_gate(
        run,
        make_baseline({"policy_regex": 1.0, "rubric_judge": 1.0}),
        tolerance=0,
    )
    assert any("missing" in failure for failure in missing.failures)
    other = make_run([make_case("a", [make_score("policy_regex", 1.0)])], version="1")
    mismatch = evaluate_gate(other, make_baseline({"policy_regex": 1.0}), tolerance=0)
    assert any("prompt_version" in failure for failure in mismatch.failures)


def test_negative_tolerance_is_rejected() -> None:
    run = make_run([make_case("a", [make_score("policy_regex", 1.0)])])
    with pytest.raises(GateError, match="tolerance"):
        evaluate_gate(run, make_baseline({"policy_regex": 1.0}), tolerance=-0.1)


def test_skipped_scores_do_not_enter_the_mean() -> None:
    run = make_run(
        [
            make_case(
                "a",
                [make_score("exact_reply", None, skipped=True), make_score("policy_regex", 1.0)],
            ),
            make_case(
                "b",
                [make_score("exact_reply", None, skipped=True), make_score("policy_regex", 0.0)],
            ),
        ]
    )
    assert scorer_means(run) == {"policy_regex": 0.5}
