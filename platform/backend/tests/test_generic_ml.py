import copy
import numpy as np
import pandas as pd
import pytest

from app.services.data_engine import prepare_features, validate_configuration, frame_fingerprint
from app.services.modeling import train_dataset, score_frame, save_bundle, load_bundle, model_report, evaluate_probabilities


def synthetic_frame(rows=240):
    """Small synthetic unit fixture, never reported as an external benchmark."""
    rng = np.random.default_rng(921)
    cost = rng.uniform(5, 100, rows)
    months = rng.integers(1, 72, rows)
    churn = rng.binomial(1, 1/(1+np.exp(-(-1+cost*.018-months*.018))))
    frame = pd.DataFrame({"ref": [f"R-{i}" for i in range(rows)], "price": cost, "lifetime": months,
                          "channel": rng.choice(["mobile", "web", "store"], rows), "leave": churn})
    frame.loc[::17, "price"] = np.nan
    frame.loc[::19, "channel"] = np.nan
    return frame


def synthetic_config():
    return {"target_column": "leave", "positive_label": "1", "customer_id": "ref", "excluded_columns": [],
            "features": [{"column": "price", "kind": "numeric"}, {"column": "lifetime", "kind": "numeric"}, {"column": "channel", "kind": "categorical"}],
            "business_domain": "subscription", "prediction_horizon": "next 3 months", "semantics": []}


@pytest.fixture(scope="module")
def trained():
    frame = synthetic_frame()
    return frame, train_dataset(frame, synthetic_config(), {"fast": True})


def test_four_models_independent_partitions_and_training_only_transformers(trained):
    frame, bundle = trained
    partitions = [set(bundle[k]) for k in ["train_indices", "validation_indices", "test_indices"]]
    assert not partitions[0] & partitions[1] and not partitions[0] & partitions[2] and not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(frame.index)
    assert len(bundle["models"]) == len(bundle["metrics"]) == 4
    training = frame.loc[bundle["train_indices"]]
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        pre = bundle["models"][name].pipeline.named_steps["preprocessor"]
        assert np.allclose(pre.named_transformers_["numeric"].named_steps["imputer"].statistics_, training[["price", "lifetime"]].median())
        assert set(pre.named_transformers_["categorical"].named_steps["encoder"].categories_[0]) == set(training.channel.dropna()) | ({"(Missing)"} if training.channel.isna().any() else set())
    assert bundle["dataset_fingerprint"] == frame_fingerprint(frame)


def test_actual_soft_voting_and_test_scores_align(trained):
    frame, bundle = trained
    mean = np.mean([bundle["probabilities"][name] for name in ["Logistic Regression", "Random Forest", "XGBoost"]], axis=0)
    assert np.allclose(bundle["probabilities"]["Hybrid Ensemble"], mean)
    for name, probability in bundle["probabilities"].items():
        assert np.isfinite(probability).all() and ((probability >= 0)&(probability <= 1)).all()
        scored = score_frame(bundle, frame.loc[bundle["test_indices"]], name)
        assert np.allclose(scored.churn_probability, probability)
    assert np.array_equal(bundle["y_test"], frame.loc[bundle["test_indices"], "leave"])


def test_training_reproducible_and_validation_selects_model(trained):
    frame, original = trained
    repeated = train_dataset(frame, synthetic_config(), {"fast": True})
    assert original["train_indices"] == repeated["train_indices"]
    for name in original["models"]:
        assert np.allclose(original["probabilities"][name], repeated["probabilities"][name])
    highest = max(original["validation_metrics"], key=lambda row: row["pr_auc"])["model"]
    assert original["preferred_model"] == highest


def test_schema_strict_unknown_categories_preserve_batch_rows(trained):
    frame, bundle = trained
    batch = frame.head(3).drop(columns="leave").copy()
    batch.loc[0, "channel"] = "a new category"
    batch = pd.concat([batch, batch.iloc[[0]]], ignore_index=True)
    scored = score_frame(bundle, batch)
    assert len(scored) == 4 and "actual_churn" not in scored
    assert scored.attrs["warnings"]
    with pytest.raises(ValueError, match="missing columns"):
        score_frame(bundle, batch.rename(columns={"price": "unrelated_balance"}))
    with pytest.raises(ValueError, match="not part"):
        score_frame(bundle, batch, "Unknown model")


def test_persistence_and_configuration_fingerprints_guard_inference(trained, tmp_path):
    frame, bundle = trained
    path = tmp_path / "model.joblib"
    save_bundle(bundle, path)
    loaded = load_bundle(path)
    assert np.allclose(score_frame(loaded, frame).churn_probability, score_frame(bundle, frame).churn_probability)
    damaged = copy.deepcopy(bundle)
    damaged["configuration"]["features"].pop()
    save_bundle(damaged, path)
    with pytest.raises(ValueError, match="fingerprint"):
        load_bundle(path)


def test_reports_use_actual_outputs_and_valid_calibration(trained):
    _, bundle = trained
    report = model_report(bundle)
    assert len(report["curves"]) == 4
    for name in bundle["models"]:
        assert report["curves"][name]["calibration"]
        assert report["curves"][name]["threshold"]
        assert sum(sum(row) for row in next(m for m in report["metrics"] if m["model"] == name)["confusion_matrix"]) == len(bundle["y_test"])
    assert evaluate_probabilities([0, 0], [.1, .2])["roc_auc"] is None


def test_insufficient_or_longitudinal_data_cannot_be_misreported_as_evaluated():
    with pytest.raises(ValueError, match="60"):
        train_dataset(synthetic_frame(30), synthetic_config(), {"fast": True})
    frame = synthetic_frame()
    frame.loc[1, "ref"] = frame.loc[0, "ref"]
    with pytest.raises(ValueError, match="repeated"):
        train_dataset(frame, synthetic_config(), {"fast": True})
    config = synthetic_config()
    config["target_column"] = None
    with pytest.raises(ValueError, match="historical binary"):
        train_dataset(synthetic_frame(), config, {"fast": True})


def test_exact_duplicates_removed_before_split_and_reported():
    frame = synthetic_frame()
    duplicated = pd.concat([frame, frame.head(4)], ignore_index=True)
    bundle = train_dataset(duplicated, synthetic_config(), {"fast": True})
    assert bundle["metadata"]["duplicates_removed"] == 4
    assert len(set(bundle["train_indices"]) | set(bundle["validation_indices"]) | set(bundle["test_indices"])) == len(frame)
