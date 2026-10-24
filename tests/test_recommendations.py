"""Recommendation policy checks using explicit, controlled segment outcomes."""
import numpy as np
import pandas as pd
import pytest

from retention.recommendations import customer_recommendations, risk_category, segment_recommendations


@pytest.fixture
def support_reference():
    # Ten customers without support all churn; the other thirty do not. Other
    # candidate fields are constant or outside the proposed segments, so only
    # the evidence-linked support action is justified for the selected customer.
    count = 40
    return pd.DataFrame({
        "Contract": ["Two year"] * count,
        "tenure": [24] * count,
        "TechSupport": ["No"] * 10 + ["Yes"] * 30,
        "OnlineSecurity": ["Yes"] * count,
        "MonthlyCharges": [50.] * count,
        "Churn": [1] * 10 + [0] * 30,
    })


def test_high_risk_specific_action_has_actual_evidence_and_support_caveat(support_reference):
    row = support_reference.iloc[0]
    suggestions = customer_recommendations(row, support_reference, .8)
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert "support" in suggestion["action"].lower()
    assert "10 records, 100.0% churn" in suggestion["evidence"]
    assert "all loaded records: 25.0%" in suggestion["evidence"]
    assert "not evidence of a technical problem" in suggestion["caveat"]
    assert "fewer than 30" in suggestion["caveat"]
    assert "unmeasured" in suggestion["caveat"]
    assert "causal" in suggestion["evidence"]


def test_recommendation_changes_when_observed_segment_elevation_changes(support_reference):
    row = support_reference.iloc[0].copy()
    raised = customer_recommendations(row, support_reference, .8)
    changed = support_reference.copy()
    changed["Churn"] = 1 - changed["Churn"]
    lowered = customer_recommendations(row, changed, .8)
    assert "elevated observed churn" in raised[0]["rationale"]
    assert len(lowered) == 1
    assert "no specific intervention is established" in lowered[0]["rationale"]


@pytest.mark.parametrize("probability", [0, .299, .3, .599999])
def test_specific_segment_actions_are_reserved_for_high_risk(support_reference, probability):
    suggestions = customer_recommendations(support_reference.iloc[0], support_reference, probability)
    assert len(suggestions) == 1
    assert "no specific intervention is established" in suggestions[0]["rationale"]
    assert "uncertain" in suggestions[0]["caveat"]


def test_configured_high_risk_boundary_is_inclusive(support_reference):
    row = support_reference.iloc[0]
    below = customer_recommendations(row, support_reference, .699, low=.2, high=.7)
    boundary = customer_recommendations(row, support_reference, .7, low=.2, high=.7)
    assert "medium risk" in below[0]["rationale"]
    assert "support" in boundary[0]["action"].lower()
    assert risk_category(.2, .2, .7) == "Medium"


def test_missing_customer_fields_do_not_create_segment_membership(support_reference):
    row = {"Contract": pd.NA, "tenure": None, "MonthlyCharges": np.nan}
    suggestions = customer_recommendations(row, support_reference, .95)
    assert len(suggestions) == 1
    assert "no specific intervention is established" in suggestions[0]["rationale"]
    assert "No customer complaint" in suggestions[0]["caveat"]


def test_large_segment_does_not_receive_small_sample_warning(support_reference):
    larger = pd.concat([support_reference] * 4, ignore_index=True)
    suggestions = customer_recommendations(larger.iloc[0], larger, .9)
    assert "40 records" in suggestions[0]["evidence"]
    assert "Small segment" not in suggestions[0]["caveat"]


def test_segment_table_uses_counts_and_withholds_action_for_unelevated_groups(support_reference):
    table = segment_recommendations(support_reference).set_index("Customer segment")
    support = table.loc["No technical support subscription"]
    assert support["Observed pattern"] == "100.0% churn (n=10); overall 25.0%"
    assert "support" in support["Possible action"].lower()
    assert "Fewer than 30 records" in support["Evidence and limitations"]
    # Every customer has identical charges, so this entire-population segment
    # cannot justify a pricing intervention based on elevated observed churn.
    charge_segment = table.loc["Monthly charges in upper quartile"]
    assert charge_segment["Possible action"].startswith("Monitor;")
    assert "Avoid unsupported intervention targeting" == charge_segment["Expected objective"]


def test_recommendations_require_labelled_reference(support_reference):
    with pytest.raises(ValueError, match="Labelled historical reference"):
        customer_recommendations(support_reference.iloc[0], support_reference.drop(columns="Churn"), .8)


def test_series_risk_output_preserves_record_alignment():
    probabilities = pd.Series([.1, .6, .3], index=["third", "first", "second"])
    categories = risk_category(probabilities)
    assert categories.index.equals(probabilities.index)
    assert categories.tolist() == ["Low", "High", "Medium"]


@pytest.mark.parametrize("probability", [-.01, 1.01, np.nan, np.inf])
def test_invalid_predictions_do_not_generate_recommendations(support_reference, probability):
    with pytest.raises(ValueError, match="probabilities"):
        customer_recommendations(support_reference.iloc[0], support_reference, probability)
