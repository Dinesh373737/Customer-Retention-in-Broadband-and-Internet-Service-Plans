import builtins
import numpy as np
import pytest
from app.services.explainability import global_explanation, customer_explanation
from app.services.modeling import train_dataset, score_frame
from test_generic_ml import synthetic_frame, synthetic_config


def test_ambiguous_encoded_names_keep_exact_source_dimensions():
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from app.services.generic_pipeline import build_preprocessor, ProbabilityModel, HybridProbabilityModel
    frame = pd.DataFrame({"A": ["A", "B_C", "Z"] * 20, "A_B": ["A", "C", "Y", "Y", "C"] * 12})
    config = {"features": [{"column": "A", "kind": "categorical"}, {"column": "A_B", "kind": "categorical"}]}
    pipeline = Pipeline([("preprocessor", build_preprocessor(config)), ("classifier", LogisticRegression(solver="liblinear"))])
    pipeline.fit(frame, [0, 1, 0, 1, 0, 1] * 10)
    member = ProbabilityModel(pipeline)
    bundle = {"models": {"Logistic Regression": member}, "configuration": config, "preferred_model": "Logistic Regression"}
    result = global_explanation(bundle)["importance"]
    ambiguous = [row for row in result if row["encoded_feature"] == "categorical__A_B_C"]
    assert {row["source_feature"] for row in ambiguous} == {"A", "A_B"}
    assert len({row["feature"] for row in result}) == 4
    bundle["models"]["Hybrid Ensemble"] = HybridProbabilityModel({"a": member, "b": member, "c": member})
    hybrid = global_explanation(bundle, "Hybrid Ensemble")["importance"]
    assert len(hybrid) == 4


@pytest.fixture(scope="module")
def explained():
    frame = synthetic_frame()
    return frame, train_dataset(frame, synthetic_config(), {"fast": True})


def test_local_explanation_refers_to_actual_calibrated_selected_model(explained):
    frame, bundle = explained
    for name in bundle["models"]:
        explanation = customer_explanation(bundle, frame.iloc[[0]], name)
        assert explanation["model"] == name
        assert explanation["probability"] == pytest.approx(score_frame(bundle, frame.iloc[[0]], name).churn_probability.iloc[0])
        assert len(explanation["drivers"]) == len(bundle["feature_columns"])
        assert all(np.isfinite(driver["contribution"]) for driver in explanation["drivers"])
        assert any("causes" in note for note in explanation["caveats"])


def test_encoded_feature_mapping_and_labeled_importances(explained):
    _, bundle = explained
    for name in bundle["models"]:
        result = global_explanation(bundle, name)
        assert result["importance"]
        assert {row["source_feature"] for row in result["importance"]} <= set(bundle["feature_columns"])
        if name == "Logistic Regression":
            assert all(row["odds_ratio"] > 0 for row in result["importance"])
        assert result["shap"] == []


def test_missing_shap_explicitly_falls_back_without_fabricating_values(explained, monkeypatch):
    _, bundle = explained
    original = builtins.__import__
    def without_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("intentional missing optional dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_shap)
    result = global_explanation(bundle, "Random Forest", include_shap=True)
    assert not result["shap_available"] and result["importance"]
    assert "fallback" in result["shap_note"]


def test_explanation_rejects_wrong_customer_dimensions(explained):
    frame, bundle = explained
    with pytest.raises(ValueError, match="exactly one"):
        customer_explanation(bundle, frame.head(2))
