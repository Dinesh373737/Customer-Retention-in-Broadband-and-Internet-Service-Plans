import numpy as np
import pytest
from retention.economics import economic_scenario, sensitivity_table
from retention.recommendations import risk_category


def test_incremental_economics_and_break_even():
    r = economic_scenario(100, 10, .1, 200, 100)
    assert r["campaign_cost"] == 1100
    assert r["retained"] == 10
    assert r["net_benefit"] == 900
    assert r["roi_pct"] == pytest.approx(900 / 1100 * 100)
    break_even = economic_scenario(100, 10, r["break_even_success_rate"], 200, 100)
    assert break_even["net_benefit"] == pytest.approx(0)


def test_zero_cost_and_zero_target_are_defined():
    r = economic_scenario(0, 0, 0, 0)
    assert r["roi_pct"] is None
    assert r["break_even_success_rate"] is None
    assert r["break_even_value"] is None
    assert r["net_benefit"] == 0


@pytest.mark.parametrize("args", [(1, -1, .1, 20), (1, 2, 1.1, 20), (1.2, 2, .1, 20), (1, 2, .1, float('nan'))])
def test_invalid_assumptions_rejected(args):
    with pytest.raises(ValueError):
        economic_scenario(*args)


def test_sensitivity_varies_cost_uplift_and_value():
    table = sensitivity_table(100, 10, .1, 200)
    assert len(table) == 27
    assert table["contact_cost"].nunique() == 3
    assert table["success_rate"].nunique() == 3
    assert table["contribution"].nunique() == 3


def test_sensitivity_rejects_invalid_base_instead_of_clipping_it():
    with pytest.raises(ValueError):
        sensitivity_table(100, 10, 2, 200)


def test_extreme_assumptions_do_not_return_infinite_financial_results():
    with pytest.raises(ValueError):
        economic_scenario(100, 10, 1, 1e308)


def test_policy_boundaries_and_invalid_probability():
    assert list(risk_category([0, .299, .3, .599, .6, 1])) == ["Low", "Low", "Medium", "Medium", "High", "High"]
    assert risk_category(.8) == "High"
    with pytest.raises(ValueError):
        risk_category(np.nan)
    with pytest.raises(ValueError):
        risk_category(.5, .6, .3)
