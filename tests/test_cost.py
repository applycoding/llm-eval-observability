"""Price catalog math. Rates in this file are fixtures, not a vendor price list."""

from llm_eval_obs.cost import ModelPrice, estimate_cost, load_catalog
from tests.factories import ROOT


def test_checked_in_fake_model_is_free_and_known() -> None:
    catalog = load_catalog(ROOT / "pricing" / "catalog.yaml")
    assert estimate_cost("fake-llm", 1200, 30, catalog) == (0.0, True)


def test_cost_uses_only_the_catalog() -> None:
    catalog = {"example-model": ModelPrice(input_per_million=3, output_per_million=15)}
    assert estimate_cost("example-model", 1000, 500, catalog) == (0.0105, True)
    assert estimate_cost("missing-model", 1000, 500, catalog) == (0.0, False)
