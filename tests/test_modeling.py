"""Small real-data integration checks for leakage, inference and persistence."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retention.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, load_data
from retention.evaluation import evaluate_probabilities
from retention.explainability import global_importance, local_explanation
from retention.modeling import MODEL_NAMES, load_bundle, save_bundle, score_customers, train_models


@pytest.fixture(scope="module")
def records():
    data, _ = load_data(Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv")
    # A reproducible real-data subset keeps the automated checks laptop-friendly.
    return data.groupby("Churn", group_keys=False).sample(frac=0.11, random_state=9).reset_index(drop=True)


@pytest.fixture(scope="module")
def bundle(records):
    return train_models(records, random_state=17, fast=True)


def test_all_models_and_true_soft_voting(bundle, records):
    assert tuple(bundle["models"]) == MODEL_NAMES
    assert len(bundle["test_indices"]) == len(bundle["y_test"])
    assert len(bundle["metrics"]) == 4
    assert set(bundle["feature_columns"]) == set(FEATURE_COLUMNS)
    assert not {"customerID", "Churn"}.intersection(bundle["feature_columns"])
    for name in MODEL_NAMES:
        probability = score_customers(bundle, records.loc[bundle["test_indices"]], name)
        np.testing.assert_allclose(probability, bundle["probabilities"][name], rtol=1e-10)
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    expected = np.mean([bundle["probabilities"][name] for name in MODEL_NAMES[:3]], axis=0)
    np.testing.assert_allclose(expected, bundle["probabilities"]["Hybrid Ensemble"])


def test_preprocessing_fit_only_on_training_partition(bundle, records):
    metadata = bundle["metadata"]
    partitions = [set(metadata[part]["indices"]) for part in ("train", "validation", "test")]
    assert not partitions[0] & partitions[1] and not partitions[0] & partitions[2] and not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(records.index)
    training = records.loc[metadata["train"]["indices"]]
    for model in list(bundle["models"].values())[:3]:
        preprocessing = model.pipeline.named_steps["preprocessor"]
        assert list(preprocessing.feature_names_in_) == list(FEATURE_COLUMNS)
        imputer = preprocessing.named_transformers_["numeric"].named_steps["imputer"]
        np.testing.assert_allclose(imputer.statistics_, training[list(NUMERIC_COLUMNS)].median().to_numpy())
        scaler = preprocessing.named_transformers_["numeric"].named_steps["scaler"]
        imputed_training = training[list(NUMERIC_COLUMNS)].fillna(training[list(NUMERIC_COLUMNS)].median())
        np.testing.assert_allclose(scaler.mean_, imputed_training.mean().to_numpy())
        encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
        for column, categories in zip(CATEGORICAL_COLUMNS, encoder.categories_):
            assert set(categories) == set(training[column].dropna().unique())
    selected = bundle["validation_metrics"].sort_values("pr_auc", ascending=False, kind="stable").iloc[0]["model"]
    assert bundle["preferred_model"] == selected


def test_persistence_unknown_categories_and_explanations(bundle, records, tmp_path):
    path = save_bundle(bundle, tmp_path / "model.joblib")
    restored = load_bundle(path)
    customer = records.iloc[[0]].drop(columns=["Churn", "customerID"])
    before = score_customers(bundle, customer)
    np.testing.assert_allclose(before, score_customers(restored, customer), atol=1e-12, rtol=1e-12)
    customer.loc[:, "Contract"] = "Unseen future contract"
    assert np.isfinite(score_customers(restored, customer)).all()
    for name in MODEL_NAMES:
        importance = global_importance(restored, name)
        assert len(importance) and {"feature", "importance"}.issubset(importance)
        assert np.isfinite(importance["importance"]).all()
        local, description = local_explanation(restored, records.iloc[[0]], name)
        assert len(local) == len(FEATURE_COLUMNS) and "not SHAP" in description
        assert np.isfinite(local["contribution"]).all()


def test_reproducible_training(bundle, records):
    repeat = train_models(records, random_state=17, fast=True)
    assert repeat["test_indices"] == bundle["test_indices"]
    assert repeat["preferred_model"] == bundle["preferred_model"]
    for name in MODEL_NAMES:
        np.testing.assert_allclose(repeat["probabilities"][name], bundle["probabilities"][name], atol=1e-12)


def test_metric_validation_and_missing_schema(bundle, records):
    with pytest.raises(ValueError, match="Missing prediction columns"):
        score_customers(bundle, records.drop(columns="tenure"))
    with pytest.raises(ValueError, match="Probabilities"):
        evaluate_probabilities([0, 1], [0.1, 1.1])
    metric = evaluate_probabilities([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metric["roc_auc"] == metric["pr_auc"] == metric["f1"] == 1
