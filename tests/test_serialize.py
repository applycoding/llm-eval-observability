"""JSON round trip for runs and baselines."""

import pytest

from llm_eval_obs.errors import RunError
from llm_eval_obs.serialize import dump_json, load_baseline, load_run, run_to_dict
from tests.factories import make_baseline, make_case, make_run, make_score


def test_run_round_trip(tmp_path) -> None:
    run = make_run([make_case("a", [make_score("policy_regex", 0.5)], output="hello")])
    path = tmp_path / "run.json"
    dump_json(path, run_to_dict(run))
    assert load_run(path) == run


def test_invalid_run_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RunError, match="invalid JSON"):
        load_run(path)


def test_baseline_requires_means(tmp_path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text('{"note": "x"}', encoding="utf-8")
    with pytest.raises(RunError, match="scorer_means"):
        load_baseline(path)
    baseline = make_baseline({"policy_regex": 1.0})
    assert baseline.scorer_means["policy_regex"] == 1.0
