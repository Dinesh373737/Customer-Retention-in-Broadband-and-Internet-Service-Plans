"""Four real models for an immutable, explicitly mapped dataset schema."""
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
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

from .data_engine import validate_configuration, target_values, prepare_features, frame_fingerprint, configuration_fingerprint, canonical_label, json_safe
from .generic_pipeline import build_preprocessor, ProbabilityModel, HybridProbabilityModel, logit_probability

MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "Hybrid Ensemble"]
ARTIFACT_VERSION = 1


def risk_category(probability, low=.3, high=.6):
    if not 0 <= low < high <= 1:
        raise ValueError("Risk boundaries must satisfy 0 <= low < high <= 1.")
    values = np.asarray(probability, dtype=float)
    if not np.isfinite(values).all() or ((values < 0)|(values > 1)).any():
        raise ValueError("Model probabilities must lie in the finite range 0 to 1.")
    result = np.where(values < low, "Low", np.where(values < high, "Medium", "High"))
    return str(result.item()) if values.ndim == 0 else result


def evaluate_probabilities(y, probability, threshold=.5):
    y, probability = np.asarray(y), np.asarray(probability)
    predicted = probability >= threshold
    both = len(np.unique(y)) == 2
    return {"accuracy": float(accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probability)) if both else None,
            "pr_auc": float(average_precision_score(y, probability)) if both else None,
            "brier": float(brier_score_loss(y, probability)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            "threshold": threshold}


def _training_data(df, config):
    clean = df.drop_duplicates().copy()
    duplicates = len(df)-len(clean)
    identifier = config.get("customer_id")
    if identifier and clean.loc[clean[identifier].notna(), identifier].duplicated().any():
        raise ValueError("Conflicting repeated customer IDs cannot use independent-customer splits. Aggregate to one pre-outcome record per customer; longitudinal/group splits are not supported in this release.")
    return clean, duplicates


def train_dataset(df, config, settings=None, progress=None):
    from xgboost import XGBClassifier
    import xgboost
    settings = settings or {}
    config = validate_configuration(df, config)
    if not config.get("target_column"):
        raise ValueError("Training needs a meaningful historical binary churn target. Use exploratory-only analysis for unlabelled data.")
    if not config["features"]:
        raise ValueError("Select at least one eligible predictor before training.")
    seed = int(settings.get("random_state", 42))
    if not 0 <= seed < 2**32-3:
        raise ValueError("random_state must be an integer from 0 to 4294967292.")
    fast = bool(settings.get("fast", False))
    selection = settings.get("selection_metric", "pr_auc")
    if selection not in {"pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy", "brier"}:
        raise ValueError("Choose a supported validation model-selection metric.")
    notify = progress or (lambda percent, message: None)
    notify(5, "Validating the confirmed schema and historical labels")
    clean, removed = _training_data(df, config)
    y = target_values(clean, config)
    if len(clean) < 60 or y.value_counts().min() < 12:
        raise ValueError("Training needs at least 60 distinct customer records and at least 12 records in each target class after deduplication.")
    positions = np.arange(len(clean))
    development, test = train_test_split(positions, test_size=.2, stratify=y, random_state=seed)
    training, validation = train_test_split(development, test_size=.25, stratify=y.iloc[development], random_state=seed+1)
    x, _ = prepare_features(clean, config)
    x_train, y_train = x.iloc[training], y.iloc[training]
    x_validation, y_validation = x.iloc[validation], y.iloc[validation]
    x_test, y_test = x.iloc[test], y.iloc[test]
    estimated_encoded = sum(min(30, x_train[f["column"]].nunique()+1) if f["kind"] == "categorical" else 1 for f in config["features"])
    if len(x_train)*estimated_encoded > 15_000_000:
        raise ValueError("The selected row/category combination would require too much dense memory. Select fewer predictors or a smaller representative dataset.")
    folds = 2 if fast else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed+2)
    imbalance = float((y_train == 0).sum() / (y_train == 1).sum())
    weighted = imbalance > 1.5 or imbalance < 2/3
    specifications = {
        MODEL_NAMES[0]: (LogisticRegression(solver="liblinear", max_iter=1500, class_weight="balanced" if weighted else None, random_state=seed), None),
        MODEL_NAMES[1]: (RandomForestClassifier(n_estimators=40 if fast else 120, min_samples_leaf=3, class_weight="balanced_subsample" if weighted else None, n_jobs=2, random_state=seed), {"classifier__max_depth": [6, None]}),
        MODEL_NAMES[2]: (XGBClassifier(n_estimators=50 if fast else 140, learning_rate=.08, subsample=.85, colsample_bytree=.85, min_child_weight=3, scale_pos_weight=imbalance if weighted else 1, objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=2, random_state=seed), {"classifier__max_depth": [2, 4]}),
    }
    models, calibration, tuning, validation_probabilities = {}, {}, {}, {}
    for index, (name, (classifier, grid)) in enumerate(specifications.items()):
        notify(10+index*22, f"Training and cross-validating {name} using training records only")
        pipeline = Pipeline([("preprocessor", build_preprocessor(config)), ("classifier", classifier)])
        if grid:
            search = GridSearchCV(pipeline, grid, cv=cv, scoring="average_precision", n_jobs=1, error_score="raise")
            search.fit(x_train, y_train)
            pipeline = search.best_estimator_
            tuning[name] = {"best_params": search.best_params_, "cv_pr_auc": float(search.best_score_)}
        else:
            pipeline.fit(x_train, y_train)
            tuning[name] = {"best_params": {}, "cv_pr_auc": None}
        raw = pipeline.predict_proba(x_validation)[:, 1]
        oof = cross_val_predict(clone(pipeline), x_train, y_train, method="predict_proba", cv=cv, n_jobs=1)[:, 1]
        calibrator = LogisticRegression(solver="liblinear", C=1000, max_iter=1000, random_state=seed).fit(logit_probability(oof), y_train)
        calibrated = calibrator.predict_proba(logit_probability(raw))[:, 1]
        raw_brier, calibrated_brier = float(brier_score_loss(y_validation, raw)), float(brier_score_loss(y_validation, calibrated))
        use = calibrated_brier < raw_brier-1e-6
        models[name] = ProbabilityModel(pipeline, calibrator if use else None)
        validation_probabilities[name] = calibrated if use else raw
        calibration[name] = {"selected": bool(use), "method": "sigmoid fitted on out-of-fold training probabilities", "raw_validation_brier": raw_brier, "calibrated_validation_brier": calibrated_brier}
    models[MODEL_NAMES[3]] = HybridProbabilityModel(dict(models))
    validation_probabilities[MODEL_NAMES[3]] = models[MODEL_NAMES[3]].predict_proba(x_validation)[:, 1]
    validation_metrics = [{"model": name, **evaluate_probabilities(y_validation, scores)} for name, scores in validation_probabilities.items()]
    preferred = sorted(validation_metrics, key=lambda row: row[selection], reverse=selection!="brier")[0]["model"]
    thresholds = {}
    for name, scores in validation_probabilities.items():
        pre, rec, cuts = precision_recall_curve(y_validation, scores)
        f1 = 2*pre[:-1]*rec[:-1]/np.maximum(pre[:-1]+rec[:-1], 1e-12)
        thresholds[name] = float(cuts[np.argmax(f1)]) if len(cuts) else .5
    notify(82, "Evaluating all four models on the same untouched held-out test customers")
    probabilities = {name: model.predict_proba(x_test)[:, 1] for name, model in models.items()}
    metrics = [{"model": name, **evaluate_probabilities(y_test, scores)} for name, scores in probabilities.items()]
    baselines = {}
    raw_training = clean.iloc[training]
    for feature in config["features"]:
        name, kind = feature["column"], feature["kind"]
        values = raw_training[name]
        if kind == "numeric":
            baselines[name] = float(pd.to_numeric(values, errors="coerce").median()) if values.notna().any() else None
        elif kind == "datetime":
            dates = pd.to_datetime(values, errors="coerce", utc=True).dropna().sort_values()
            baselines[name] = dates.iloc[len(dates)//2].isoformat() if len(dates) else None
        else:
            modes = values.dropna().map(canonical_label).mode()
            baselines[name] = modes.iloc[0] if len(modes) else None
    schema_values = {f["column"]: sorted(set(x_train[f["column"]].dropna().astype(str))) for f in config["features"] if f["kind"] in {"categorical", "ordinal"}}
    category_warnings = []
    for name, levels in schema_values.items():
        for partition, table in [("validation", x_validation), ("test", x_test)]:
            unseen = sorted(set(table[name].dropna().astype(str))-set(levels))
            if unseen:
                category_warnings.append(f"'{name}' has {len(unseen)} categories unseen in training in {partition}; saved encoder unknown-category handling is used.")
    bundle = {
        "artifact_version": ARTIFACT_VERSION, "configuration": config,
        "dataset_fingerprint": frame_fingerprint(df), "configuration_fingerprint": configuration_fingerprint(config),
        "feature_columns": [f["column"] for f in config["features"]], "models": models,
        "preferred_model": preferred, "metrics": metrics, "validation_metrics": validation_metrics,
        "train_indices": clean.index[training].tolist(), "validation_indices": clean.index[validation].tolist(), "test_indices": clean.index[test].tolist(),
        "y_test": y_test.to_numpy(dtype=int), "y_validation": y_validation.to_numpy(dtype=int),
        "probabilities": probabilities, "validation_probabilities": validation_probabilities,
        "training_baselines": baselines, "training_categories": schema_values,
        "background": raw_training.sample(n=min(80, len(raw_training)), random_state=seed),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {"random_state": seed, "fast": fast, "selection_metric": selection,
                     "split": "independent customer-level stratified 60/20/20", "train_rows": len(training), "validation_rows": len(validation), "test_rows": len(test),
                     "original_rows": len(df), "distinct_rows": len(clean), "duplicates_removed": removed,
                     "metric_threshold": .5, "suggested_thresholds": thresholds,
                     "threshold_selection": "maximum validation F1 only", "cv_folds": folds, "tuning": tuning,
                     "calibration": calibration, "ensemble_weights": {name: 1/3 for name in MODEL_NAMES[:3]},
                     "class_weighting": weighted, "training_positive_rate": float(y_train.mean()),
                     "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
                     "warnings": config.get("warnings", [])+category_warnings,
                     "limitations": ["Only independent, customer-level tabular data are supported; repeated-customer longitudinal training is blocked.",
                                     "Calibration OOF predictions use hyperparameters selected within the same training cohort, not nested CV. Validation and final test cohorts remain independent.",
                                     "Historical full-data customer scores include fitting records and are not held-out performance estimates.",
                                     "Risk bands default to 30%/60%; no treatment effect or causal churn reason is estimated."]},
    }
    notify(96, "Saving the exact fitted schema, calibration and ensemble pipelines")
    return bundle


def _selected(bundle, model_name=None):
    chosen = model_name or bundle["preferred_model"]
    if chosen not in bundle["models"]:
        raise ValueError("The requested prediction model is not part of this analysis version.")
    return chosen


def score_frame(bundle, df, model_name=None):
    chosen = _selected(bundle, model_name)
    features, _ = prepare_features(df, bundle["configuration"], scoring=True)
    scores = bundle["models"][chosen].predict_proba(features)[:, 1] if len(df) else np.array([])
    result = df.copy().reset_index(drop=True)
    result["row_index"] = np.arange(len(df))
    identifier = bundle["configuration"].get("customer_id")
    result["customer_reference"] = [str(value) if pd.notna(value) else f"Record-{index+1:06d}" for index, value in enumerate(df[identifier])] if identifier and identifier in df else [f"Record-{index+1:06d}" for index in range(len(df))]
    result["churn_probability"] = scores
    result["risk_category"] = risk_category(scores)
    target = bundle["configuration"].get("target_column")
    if target and target in df:
        labels = df[target].map(canonical_label)
        if set(labels)-set(bundle["configuration"]["target_labels"]):
            raise ValueError("The supplied target labels differ from this model's saved target mapping.")
        result["actual_outcome"] = labels.to_numpy()
        result["actual_churn"] = (labels == bundle["configuration"]["positive_label"]).astype(int).to_numpy()
    result.attrs["warnings"] = []
    for name, levels in bundle.get("training_categories", {}).items():
        unseen = set(features[name].dropna().astype(str))-set(levels)
        if unseen:
            result.attrs["warnings"].append(f"'{name}' has {len(unseen)} unseen categories; saved unknown-category handling was applied.")
    return result


def save_bundle(bundle, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    os.replace(temporary, destination)
    return destination


def load_bundle(path):
    # Artifact loading is only for locally created trusted files, never user uploads.
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict) or bundle.get("artifact_version") != ARTIFACT_VERSION or not {"models", "configuration", "dataset_fingerprint"}.issubset(bundle):
        raise ValueError("This is not a compatible generic-platform model artifact.")
    if configuration_fingerprint(bundle["configuration"]) != bundle["configuration_fingerprint"]:
        raise ValueError("The saved feature mapping does not match its model fingerprint.")
    return bundle


def model_report(bundle):
    y = np.asarray(bundle["y_test"])
    curves = {}
    for name, probability in bundle["probabilities"].items():
        probability = np.asarray(probability)
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        fraction, mean = calibration_curve(y, probability, n_bins=8, strategy="quantile")
        validation_y, validation_p = np.asarray(bundle["y_validation"]), np.asarray(bundle["validation_probabilities"][name])
        curves[name] = {"roc": [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr)],
                        "precision_recall": [{"recall": float(a), "precision": float(b)} for a, b in zip(recall, precision)],
                        "calibration": [{"predicted": float(a), "observed": float(b)} for a, b in zip(mean, fraction)],
                        "threshold": [{"threshold": float(t), "precision": float(precision_score(validation_y, validation_p >= t, zero_division=0)), "recall": float(recall_score(validation_y, validation_p >= t, zero_division=0))} for t in np.linspace(.01, .99, 50)],
                        "probability_distribution": [{"probability": float(p), "actual_churn": int(label)} for p, label in zip(probability, y)]}
    return json_safe({"preferred_model": bundle["preferred_model"], "metrics": bundle["metrics"],
                      "validation_metrics": bundle["validation_metrics"], "models": list(bundle["models"]),
                      "curves": curves, "metadata": bundle["metadata"], "generated_at": bundle["generated_at"],
                      "dataset_fingerprint": bundle["dataset_fingerprint"], "configuration_fingerprint": bundle["configuration_fingerprint"],
                      "test_prevalence": float(y.mean()), "configuration": bundle["configuration"]})
