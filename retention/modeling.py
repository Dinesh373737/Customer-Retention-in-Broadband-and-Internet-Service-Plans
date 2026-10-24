"""Reproducible training, validation selection, held-out evaluation and persistence."""
from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from .data import FEATURE_COLUMNS, dataset_fingerprint
from .ensemble import ProbabilityModel, SoftVotingEnsemble, probability_logit
from .evaluation import evaluate_probabilities, validation_threshold
from .preprocessing import make_preprocessor, predictor_frame, training_baselines

MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost", "Hybrid Ensemble")
ARTIFACT_VERSION = 1


def _pipeline(classifier):
    return Pipeline([("preprocessor", make_preprocessor()), ("classifier", classifier)])


def train_models(df: pd.DataFrame, random_state: int = 42, fast: bool = False) -> dict:
    """Fit on 60%, select on 20%, evaluate all four on untouched final 20%.

    RF/XGBoost hyperparameters use training-only CV. Sigmoid calibrators use
    training-only out-of-fold predictions; validation Brier determines whether
    to use each calibrator. Preferred model and F1 thresholds use validation.
    The original training pipelines remain available for raw-model explanations.
    """
    try:
        import xgboost
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required for all four models. Install it with: pip install xgboost") from exc
    if "Churn" not in df:
        raise ValueError("Training requires a cleaned, binary Churn label.")
    if not df.index.is_unique:
        raise ValueError("Training requires unique row indices; reset the dataset index first.")
    y = pd.to_numeric(df["Churn"], errors="coerce")
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Churn must contain only both integer classes 0 and 1.")
    if len(df) < 60 or y.value_counts().min() < 12:
        raise ValueError("Training needs at least 60 records and 12 examples of each churn class.")
    features = predictor_frame(df)
    if "customerID" in features or "Churn" in features:
        raise AssertionError("Identifier/target leakage in predictors.")
    y = y.astype(int)
    development_indices, test_indices = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=random_state,
    )
    train_indices, validation_indices = train_test_split(
        development_indices, test_size=0.25,
        stratify=y.iloc[development_indices], random_state=random_state + 1,
    )
    x_train, y_train = features.iloc[train_indices], y.iloc[train_indices]
    x_validation, y_validation = features.iloc[validation_indices], y.iloc[validation_indices]
    x_test, y_test = features.iloc[test_indices], y.iloc[test_indices]
    cv = StratifiedKFold(n_splits=2 if fast else 3, shuffle=True, random_state=random_state + 2)
    imbalance = float((y_train == 0).sum() / (y_train == 1).sum())
    specifications = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced", random_state=random_state), None,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=60 if fast else 160, class_weight="balanced_subsample", random_state=random_state, n_jobs=2),
            {"classifier__max_depth": [8, None], "classifier__min_samples_leaf": [3] if fast else [2, 5]},
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=70 if fast else 180, learning_rate=0.08, max_depth=3, min_child_weight=3,
                          subsample=0.85, colsample_bytree=0.85, objective="binary:logistic", eval_metric="logloss",
                          scale_pos_weight=imbalance, tree_method="hist", n_jobs=2, random_state=random_state),
            {"classifier__max_depth": [2, 4] if fast else [3, 5],
             "classifier__learning_rate": [0.08] if fast else [0.05, 0.1]},
        ),
    }
    models, tuning, calibration, validation_probabilities = {}, {}, {}, {}
    for name, (estimator, grid) in specifications.items():
        pipeline = _pipeline(estimator)
        if grid:
            search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=cv, n_jobs=1, refit=True, error_score="raise")
            search.fit(x_train, y_train)
            pipeline = search.best_estimator_
            tuning[name] = {"best_params": search.best_params_, "training_cv_pr_auc": float(search.best_score_),
                            "candidates": len(search.cv_results_["params"])}
        else:
            pipeline.fit(x_train, y_train)
            tuning[name] = {"best_params": {}, "training_cv_pr_auc": None, "candidates": 1}
        raw_validation = pipeline.predict_proba(x_validation)[:, 1]
        oof_probability = cross_val_predict(clone(pipeline), x_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        sigmoid = LogisticRegression(C=1000, solver="liblinear", max_iter=1000, random_state=random_state)
        sigmoid.fit(probability_logit(oof_probability), y_train)
        calibrated_validation = sigmoid.predict_proba(probability_logit(raw_validation))[:, 1]
        raw_brier = float(brier_score_loss(y_validation, raw_validation))
        calibrated_brier = float(brier_score_loss(y_validation, calibrated_validation))
        use_calibration = calibrated_brier < raw_brier - 1e-6
        calibration[name] = {"method": "sigmoid of training-only out-of-fold logit probabilities", "selected": bool(use_calibration),
                             "raw_validation_brier": raw_brier, "calibrated_validation_brier": calibrated_brier,
                             "selection": "lower validation Brier score"}
        models[name] = ProbabilityModel(name, pipeline, sigmoid if use_calibration else None, calibration[name])
        validation_probabilities[name] = calibrated_validation if use_calibration else raw_validation
    models["Hybrid Ensemble"] = SoftVotingEnsemble(dict(models))
    validation_probabilities["Hybrid Ensemble"] = models["Hybrid Ensemble"].predict_proba(x_validation)[:, 1]
    # All selection is complete before any prediction on the held-out test set.
    validation_metrics = pd.DataFrame([
        {"model": name, **evaluate_probabilities(y_validation, probability)}
        for name, probability in validation_probabilities.items()
    ])
    preferred_model = str(validation_metrics.sort_values("pr_auc", ascending=False, kind="stable").iloc[0]["model"])
    thresholds = {name: validation_threshold(y_validation, probability) for name, probability in validation_probabilities.items()}
    probabilities = {name: model.predict_proba(x_test)[:, 1] for name, model in models.items()}
    metrics = pd.DataFrame([
        {"model": name, **evaluate_probabilities(y_test, probability)} for name, probability in probabilities.items()
    ])
    return {
        "artifact_version": ARTIFACT_VERSION,
        "data_hash": dataset_fingerprint(df),
        "models": models,
        "metrics": metrics,
        "validation_metrics": validation_metrics,
        "preferred_model": preferred_model,
        "test_indices": df.index[test_indices].tolist(),
        "y_test": y_test.to_numpy(dtype=int),
        "probabilities": probabilities,
        "validation_probabilities": validation_probabilities,
        "y_validation": y_validation.to_numpy(dtype=int),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "training_baselines": training_baselines(x_train),
        "explanation_background": x_train.sample(n=min(100, len(x_train)), random_state=random_state),
        "metadata": {
            "random_state": random_state, "fast": fast,
            "train": {"rows": len(x_train), "indices": df.index[train_indices].tolist(), "positive_rate": float(y_train.mean())},
            "validation": {"rows": len(x_validation), "indices": df.index[validation_indices].tolist()},
            "test": {"rows": len(x_test), "indices": df.index[test_indices].tolist()},
            "split": "stratified 60% training / 20% validation / 20% held-out test",
            "tune": tuning, "cv_folds": cv.n_splits, "calibration": calibration,
            "weights": dict(zip(MODEL_NAMES[:3], [1 / 3] * 3)),
            "preferred_selection": "highest validation PR-AUC (average precision)",
            "threshold": thresholds[preferred_model], "thresholds": thresholds,
            "threshold_selection": "maximum validation F1; metric comparison uses common 0.5 threshold",
            "metric_threshold": 0.5, "pr_auc_definition": "average precision",
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        },
    }


def score_customers(bundle: dict, df: pd.DataFrame, model_name: str | None = None) -> np.ndarray:
    model_name = model_name or bundle["preferred_model"]
    if model_name not in bundle["models"]:
        raise ValueError(f"Unknown model: {model_name}")
    if len(df) == 0:
        return np.array([], dtype=float)
    return bundle["models"][model_name].predict_proba(predictor_frame(df))[:, 1]


def save_bundle(bundle: dict, path) -> Path:
    """Atomically write a joblib artifact containing inference pipelines and evidence."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, destination)
    return destination


def load_bundle(path) -> dict:
    """Load an artifact created by this application; only load trusted local files."""
    bundle = joblib.load(Path(path))
    required = {"models", "metrics", "data_hash", "feature_columns", "preferred_model"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("This file is not a compatible retention model artifact.")
    if bundle.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Model artifact version changed; retrain using the current application.")
    if list(bundle["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("Model predictor schema changed; retrain the current dataset.")
    return bundle
