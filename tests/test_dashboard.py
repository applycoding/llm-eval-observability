"""Dashboard view-model rows. Streamlit itself is not imported here."""

from llm_eval_obs.dashboard.view_model import case_table, delta_table, metric_cards
from tests.factories import make_case, make_run, make_score


def test_metric_cards_list_scorers_and_mark_unknown_cost() -> None:
    run = make_run(
        [make_case("a", [make_score("policy_regex", 1.0)], cost_known=False)],
    )
    cards = dict(metric_cards(run))
    assert cards["policy_regex"] == "1.0000"
    assert "incomplete" in cards["cost usd"]
    assert cards["cases"] == "1"


def test_case_and_delta_tables() -> None:
    before = make_run([make_case("a", [make_score("policy_regex", 0.0)])], version="1")
    after = make_run([make_case("a", [make_score("policy_regex", 1.0)])], version="2")
    assert case_table(after)[0]["score"] == "1.0000"
    assert delta_table(before, after)[0]["delta"] == "+1.0000"
