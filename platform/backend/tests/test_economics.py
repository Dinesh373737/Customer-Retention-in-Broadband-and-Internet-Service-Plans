import numpy as np
import pandas as pd
import pytest

from app.services.economics import campaign_scenario, economic_analysis


def test_incremental_economics_does_not_count_baseline_retention():
    result = campaign_scenario(dict(targeted=100, intervention_cost=10, incremental_impact=.1,
                                   contribution=200, fixed_cost=100, budget=None, period="12 months", currency="USD"))
    assert result["incremental_retained"] == 10
    assert result["total_expenditure"] == 1100
    assert result["incremental_contribution"] == 2000
    assert result["net_benefit"] == 900
    assert result["roi_pct"] == pytest.approx(900/1100*100)
    assert result["expected_net_value_per_customer"] == 10
    assert result["break_even_impact"] == pytest.approx(.055)


def test_budget_restricts_targets_after_fixed_cost():
    result = campaign_scenario(dict(targeted=100, intervention_cost=10, incremental_impact=.1,
                                   contribution=200, fixed_cost=100, budget=350))
    assert result["targeted"] == 25
    assert result["total_expenditure"] == 350
    assert result["budget_remaining"] == 0
    below_fixed = campaign_scenario(dict(targeted=100, intervention_cost=10, fixed_cost=100, budget=50))
    assert below_fixed["launched"] is False
    assert below_fixed["total_expenditure"] == 0


def test_zero_campaign_and_zero_cost_have_defined_outputs():
    result = campaign_scenario(dict(targeted=0, intervention_cost=0, incremental_impact=0, contribution=0, fixed_cost=0))
    assert result["roi_pct"] is None
    assert result["break_even_impact"] is None
    assert result["net_benefit"] == 0
    result = campaign_scenario(dict(targeted=100, intervention_cost=0, fixed_cost=0, budget=0))
    assert result["targeted"] == 100


@pytest.mark.parametrize("inputs", [{"incremental_impact": 2}, {"incremental_impact": -.1}, {"contribution": np.nan},
                                   {"intervention_cost": -1}, {"targeted": 1.5}, {"budget": np.inf}, {"fixed_cost": -1}])
def test_invalid_assumptions_rejected(inputs):
    with pytest.raises(ValueError):
        campaign_scenario(inputs)


def test_ranked_value_uses_unconditional_uplift_and_risk_only_for_ties():
    scored = pd.DataFrame({"row_index": [0,1], "customer_reference": ["a","b"], "churn_probability": [.2,.9]})
    result = economic_analysis(dict(targeted=2, intervention_cost=10, incremental_impact=.1, contribution=200, fixed_cost=0), scored)
    assert [row["expected_net_value"] for row in result["ranking"]] == [10,10]
    assert result["ranking"][0]["customer_reference"] == "b"
    assert len(result["sensitivity"]) == 117
    assert result["scenario"]["incremental_retained"] == pytest.approx(.2)


def test_no_profitable_ranking_when_assumed_value_negative():
    scored = pd.DataFrame({"row_index": [0], "churn_probability": [.99]})
    result = economic_analysis(dict(targeted=1, intervention_cost=100, incremental_impact=.01, contribution=10), scored)
    assert result["ranking"] == []
