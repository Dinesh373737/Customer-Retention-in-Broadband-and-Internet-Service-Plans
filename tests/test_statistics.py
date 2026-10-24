from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retention.data import load_data
from retention.eda import add_tenure_band, executive_insights, segment_summary
from retention.statistics import association_tests, logistic_odds_ratios

DATASET = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


def test_segment_rates_include_denominators_and_non_degenerate_uncertainty():
    frame = pd.DataFrame({"Contract": ["A", "A", "B", "B"], "Churn": [0, 1, 0, 0]})
    summary = segment_summary(frame, "Contract").set_index("segment")
    assert summary.loc["A", "customers"] == 2
    assert summary.loc["A", "churn_rate"] == 0.5
    assert summary.loc["B", "churn_rate"] == 0
    assert 0 < summary.loc["A", "ci_lower"] < 0.5 < summary.loc["A", "ci_upper"] < 1
    assert summary.loc["B", "ci_upper"] > 0


def test_tenure_boundaries_and_missing_are_explicit():
    values = [0, 12, 13, 24, 25, 48, 49, 60, 61, 120, np.nan]
    bands = add_tenure_band(pd.DataFrame({"tenure": values}))["TenureBand"]
    assert bands.iloc[0] == bands.iloc[1] == "0–12 months"
    assert bands.iloc[2] == bands.iloc[3] == "13–24 months"
    assert bands.iloc[4] == bands.iloc[5] == "25–48 months"
    assert bands.iloc[6] == bands.iloc[7] == "49–60 months"
    assert bands.iloc[8] == bands.iloc[9] == "61+ months"
    assert pd.isna(bands.iloc[10])


def test_sparse_tables_use_fisher_and_large_sparse_tables_withhold_chi_square():
    binary = pd.DataFrame({"Partner": ["Yes", "Yes", "No", "No"], "Churn": [0, 0, 1, 1]})
    tested = association_tests(binary).iloc[0]
    assert tested["test"] == "Fisher exact"
    assert tested["assumptions_met"]
    assert 0 <= tested["p_adjusted"] <= 1
    sparse = pd.DataFrame({"Contract": ["A", "A", "B", "B", "C", "C"], "Churn": [0, 1, 0, 1, 0, 1]})
    withheld = association_tests(sparse).iloc[0]
    assert withheld["test"] == "Not tested (sparse table)"
    assert pd.isna(withheld["p_value"])
    assert not withheld["assumptions_met"]


def test_welch_interval_has_correct_direction_and_multiple_test_adjustment():
    values = np.arange(50, dtype=float)
    frame = pd.DataFrame({
        "tenure": np.r_[values, values + 10],
        "MonthlyCharges": np.r_[values, values + 30],
        "Churn": [0] * 50 + [1] * 50,
    })
    tests = association_tests(frame).set_index("feature")
    assert tests.loc["tenure", "effect_size"] > 0
    assert tests.loc["tenure", "ci_lower"] < 10 < tests.loc["tenure", "ci_upper"]
    assert tests.loc["MonthlyCharges", "p_value"] < tests.loc["tenure", "p_value"]
    assert (tests["p_adjusted"] >= tests["p_value"]).all()
    assert tests["assumptions_met"].all()


def test_actual_statistics_and_insights_are_computed_from_data():
    frame, _ = load_data(DATASET)
    tests = association_tests(frame)
    assert len(tests) == 19
    assert tests["p_value"].notna().all()
    assert tests["effect_size"].notna().all()
    assert (tests["p_adjusted"] >= tests["p_value"]).all()
    insights = executive_insights(frame)
    assert str(int(frame["Churn"].sum())) in insights[0].replace(",", "")
    assert "historical" in insights[-1].lower()


def test_odds_ratios_have_real_model_confidence_intervals():
    pytest.importorskip("statsmodels")
    frame, _ = load_data(DATASET)
    ratios, note = logistic_odds_ratios(frame)
    assert not ratios.empty, note
    assert (ratios["ci_lower"] <= ratios["odds_ratio"]).all()
    assert (ratios["ci_upper"] >= ratios["odds_ratio"]).all()
    assert np.isfinite(ratios.select_dtypes("number").to_numpy()).all()
    assert "not causal" in note
    assert "held-out" in note


def test_odds_ratios_reject_singular_design_and_single_class():
    pytest.importorskip("statsmodels")
    count = 100
    frame = pd.DataFrame({
        "Churn": [0, 1] * (count // 2), "tenure": np.arange(count),
        "MonthlyCharges": 2 * np.arange(count), "Contract": ["Month-to-month"] * count,
        "InternetService": ["DSL"] * count, "SeniorCitizen": ["No"] * count,
    })
    ratios, note = logistic_odds_ratios(frame)
    assert ratios.empty
    assert "singular" in note
    frame["Churn"] = 0
    ratios, note = logistic_odds_ratios(frame)
    assert ratios.empty
    assert "both valid historical Churn classes" in note
