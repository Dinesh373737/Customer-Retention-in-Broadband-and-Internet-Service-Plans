import numpy as np
import pandas as pd
import pytest

from app.services.recommendations import generate_recommendations, catalogue


def sample():
    # Deliberately labelled synthetic fixtures; never used as benchmark results.
    df = pd.DataFrame({"Engagement": ["Low"]*100 + ["High"]*100,
                       "Age": [20,30]*100,
                       "Outcome": [1]*70+[0]*30+[1]*10+[0]*90})
    config = {"target_column": "Outcome", "positive_label": "1", "target_labels": ["0","1"],
              "features": [{"column":"Engagement","kind":"categorical"},{"column":"Age","kind":"numeric"}],
              "business_domain":"banking", "semantics": [{"column":"Engagement","concept":"engagement","meaning":"Recorded account activity level", "confirmed":True,"actionable":True,"protected":False,"estimated_cost":5}]}
    return df, config


def test_confirmed_semantics_and_real_segment_evidence_produce_domain_action():
    df, config = sample()
    result = generate_recommendations(df,config)
    row = next(row for row in result["segments"] if row["segment"] == "Engagement = Low")
    assert row["sample_size"] == 100
    assert row["observed_churn_rate"] == pytest.approx(.7)
    assert row["overall_churn_rate"] == pytest.approx(.4)
    assert row["evidence_status"] == "observational_signal"
    assert "account engagement" in row["action"].lower()
    assert row["estimated_cost"] == 5
    assert "causal" in " ".join(row["caveats"])


def test_flipping_actual_labels_changes_recommended_segment():
    df, config = sample()
    df["Outcome"] = 1-df["Outcome"]
    result = generate_recommendations(df,config)
    assert result["segments"][0]["segment"] == "Engagement = High"


def test_unknown_feature_no_mapping_invents_no_action():
    df, config = sample()
    config["semantics"] = []
    result = generate_recommendations(df,config)
    assert result["segments"] == []
    assert result["customers"] == []
    assert "gather" in " ".join(result["limitations"])


def test_unconfirmed_meaning_requires_confirmation():
    df, config = sample()
    config["semantics"][0]["confirmed"] = False
    result = generate_recommendations(df,config)
    assert result["segments"] == []
    assert result["mapping_warnings"]


def test_protected_attribute_never_drives_recommendation_even_mislabeled():
    df, config = sample()
    config["semantics"] = [{"column":"Age","concept":"pricing","meaning":"Incorrect mapping attempt", "confirmed":True,"actionable":True,"protected":False}]
    result = generate_recommendations(df,config)
    assert result["segments"] == []
    assert "protected" in result["mapping_warnings"][0]


def test_small_segment_is_investigation_not_unsupported_action():
    df, config = sample()
    df = pd.concat([df.iloc[:15], df.iloc[100:120]],ignore_index=True)
    result = generate_recommendations(df,config)
    assert result["segments"]
    assert result["segments"][0]["evidence_status"] == "insufficient"
    assert result["segments"][0]["action"].startswith("Gather context")


def test_support_subscription_is_not_turned_into_complaint():
    df, config = sample()
    config["semantics"][0].update(concept="support", meaning="Support subscription category")
    result = generate_recommendations(df,config)
    row=result["segments"][0]
    assert "support" in row["action"].lower()
    assert "not evidence of a complaint" in " ".join(row["caveats"])


def test_customer_suggestions_require_high_model_risk_and_match_evidence():
    df, config = sample()
    scored=df.copy()
    scored["row_index"]=range(len(df))
    scored["customer_reference"]=[f"r{i}" for i in range(len(df))]
    scored["churn_probability"]=.2
    scored.loc[0,"churn_probability"] = .8
    result=generate_recommendations(df,config,scored)
    assert result["customers"]
    assert {row["row_index"] for row in result["customers"]} == {0}
    assert result["customers"][0]["churn_probability"] == .8


def test_exploratory_mode_does_not_fabricate_evidence():
    df, config = sample()
    config["target_column"] = None
    result=generate_recommendations(df.drop(columns="Outcome"),config)
    assert result["evidence_status"] == "insufficient"
    assert result["segments"] == []


def test_catalogue_supports_all_required_industries():
    domains=set(catalogue()["concepts"]["engagement"]["industries"])
    assert {"telecommunications","banking","ecommerce","subscription"} <= domains


def test_catalogue_exposes_configurable_evidence_actions_and_costs():
    for rule in catalogue()["concepts"].values():
        assert {"trigger_conditions","evidence_requirements","actionability_constraints","possible_costs"} <= set(rule)
        assert rule["possible_costs"]["default_per_customer"] is None
        assert rule["trigger_conditions"]["customer_risk_threshold"]==.6


def test_concept_specific_threshold_override_changes_evidence_status(monkeypatch):
    import copy
    import app.services.recommendations as service
    df,config=sample()
    book=copy.deepcopy(catalogue())
    book["concepts"]["engagement"]["trigger_conditions"]["minimum_segment_size"]=101
    monkeypatch.setattr(service,"catalogue",lambda:book)
    result=generate_recommendations(df,config)
    assert result["segments"]
    assert all(row["evidence_status"]=="insufficient" for row in result["segments"])


def test_global_model_evidence_is_annotation_not_causal_or_required_gate():
    df,config=sample()
    result=generate_recommendations(df,config,global_drivers={"Engagement":{"importance":.35,"model":"XGBoost","method":"Raw estimator gain"}})
    assert result["segments"][0]["global_model_evidence"]["importance"]==.35
    assert any(driver["source"]=="selected-model predictive importance" for driver in result["segments"][0]["risk_drivers"])
    assert result["segments"][0]["evidence_status"]=="observational_signal"
    assert "no verified causal" in " ".join(result["segments"][0]["caveats"])


@pytest.mark.parametrize("column",["CustomerID","DateOfBirth","emailAddress"])
def test_additional_identifying_or_immutable_names_cannot_drive_actions(column):
    df,config=sample()
    df=df.rename(columns={"Engagement":column})
    config["features"][0]["column"]=column
    config["semantics"][0]["column"]=column
    result=generate_recommendations(df,config)
    assert result["segments"]==[]
    assert result["mapping_warnings"]
