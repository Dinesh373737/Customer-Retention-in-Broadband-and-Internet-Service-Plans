"""Honest raw-estimator importances and actual chosen-model sensitivities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter

from .data_engine import json_safe, prepare_features
from .generic_pipeline import HybridProbabilityModel
from .modeling import _selected, risk_category


CAVEAT = "Predictive associations and model sensitivities do not establish customer-specific causes or intervention effects."


def _encoded_metadata(preprocessor):
    """Map by fitted transformer positions, never ambiguous name prefixes.

    An input A with category B_C and input A_B with category C can produce
    identical sklearn names. Stable index-qualified labels keep both dimensions.
    """
    names = list(preprocessor.get_feature_names_out())
    sources = [None] * len(names)
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or isinstance(transformer, str):
            continue
        source_columns = list(columns)
        if name == "categorical":
            encoder = transformer.named_steps["encoder"]
            infrequent = getattr(encoder, "infrequent_categories_", [None] * len(source_columns))
            dropped = encoder.drop_idx_
            expanded = []
            for index, column in enumerate(source_columns):
                count = len(encoder.categories_[index])
                if infrequent[index] is not None:
                    count -= len(infrequent[index]) - 1
                if dropped is not None and dropped[index] is not None:
                    count -= 1
                expanded.extend([column] * count)
        else:
            expanded = source_columns
        positions = preprocessor.output_indices_[name]
        if positions.stop - positions.start != len(expanded):
            raise ValueError("Fitted explanation dimensions do not match their source columns.")
        sources[positions.start:positions.stop] = expanded
    if any(source is None for source in sources):
        raise ValueError("A fitted explanatory dimension has no supported source feature.")
    counts = Counter(names)
    labels = [str(name) if counts[name] == 1 else f"{name} [source {source}; encoded {index}]"
              for index, (name, source) in enumerate(zip(names, sources))]
    return names, labels, sources


def _importance(model, config):
    preprocessor = model.pipeline.named_steps["preprocessor"]
    classifier = model.pipeline.named_steps["classifier"]
    names, labels, sources = _encoded_metadata(preprocessor)
    coefficients = classifier.coef_[0] if hasattr(classifier, "coef_") else None
    values = np.abs(coefficients) if coefficients is not None else classifier.feature_importances_
    result = []
    categorical = next((columns for name, _, columns in preprocessor.transformers_ if name == "categorical"), [])
    reference_map = {}
    if len(categorical):
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        for column, categories, dropped in zip(categorical, encoder.categories_, encoder.drop_idx_):
            reference_map[column] = f"relative to training reference {categories[int(dropped)]}" if dropped is not None else "encoded category"
    feature_kinds = {f["column"]: f["kind"] for f in config["features"]}
    for index, (name, value) in enumerate(zip(names, values)):
        source = sources[index]
        kind = feature_kinds.get(source)
        reference = reference_map.get(source, "per training standard deviation" if kind in {"numeric", "datetime"} else "per confirmed ordinal rank (missing sentinel -1)")
        result.append({"feature": labels[index], "encoded_feature": str(name), "encoded_index": index,
                       "source_feature": source, "importance": float(value),
                       "coefficient": float(coefficients[index]) if coefficients is not None else None,
                       "odds_ratio": float(np.exp(np.clip(coefficients[index], -30, 30))) if coefficients is not None else None,
                       "reference": reference,
                       "method": "Regularized raw logistic coefficient before calibration" if coefficients is not None else "Raw estimator impurity/gain importance"})
    return sorted(result, key=lambda row: row["importance"], reverse=True)


def _shap_summary(bundle, model_name):
    model = bundle["models"][model_name]
    if isinstance(model, HybridProbabilityModel):
        return [], "SHAP is not computed for the calibrated probability ensemble. Individual perturbations use the actual ensemble."
    try:
        import shap
    except ImportError:
        return [], "SHAP is unavailable. Fitted importances and actual-model perturbations are the explicit fallback."
    try:
        background, _ = prepare_features(bundle["background"], bundle["configuration"])
        preprocessor = model.pipeline.named_steps["preprocessor"]
        classifier = model.pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(background)
        names, labels, sources = _encoded_metadata(preprocessor)
        if hasattr(classifier, "coef_"):
            explainer = shap.LinearExplainer(classifier, transformed)
            values = explainer.shap_values(transformed)
            units = "raw logistic log odds"
        else:
            explainer = shap.TreeExplainer(classifier)
            values = explainer.shap_values(transformed)
            units = "raw forest probability" if model_name == "Random Forest" else "raw XGBoost log odds"
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        if values.shape != transformed.shape or not np.isfinite(values).all():
            raise ValueError("The returned SHAP dimensions/values are not supported for this fitted model.")
        rows = [{"sample": int(i), "feature": labels[j], "encoded_feature": str(name), "encoded_index": j, "source_feature": sources[j],
                 "shap_value": float(values[i, j]), "value": float(transformed[i, j])}
                for i in range(len(transformed)) for j, name in enumerate(names)]
        return rows, f"Actual SHAP on {len(background)} training-background records. Scale: {units}; raw base estimator before optional probability calibration. Not causal evidence."
    except Exception as exc:
        return [], f"SHAP fallback: {type(exc).__name__}: {exc}. Fitted importances and actual-model perturbations remain available."


def global_explanation(bundle, model_name=None, include_shap=False):
    chosen = _selected(bundle, model_name)
    model = bundle["models"][chosen]
    if isinstance(model, HybridProbabilityModel):
        totals = {}
        sources = {}
        for weight, member in zip(model.weights, model.members.values()):
            member_values = _importance(member, bundle["configuration"])
            denominator = sum(row["importance"] for row in member_values) or 1
            for row in member_values:
                key = row["feature"]
                totals[key] = totals.get(key, 0) + weight*row["importance"]/denominator
                sources[key] = row["source_feature"]
        importance = [{"feature": feature, "source_feature": sources[feature], "importance": value,
                       "coefficient": None, "odds_ratio": None, "reference": "encoded predictor",
                       "method": "Equal-weight normalized member importance; heuristic, not SHAP"} for feature, value in totals.items()]
        importance.sort(key=lambda row: row["importance"], reverse=True)
    else:
        importance = _importance(model, bundle["configuration"])
    shap_rows, shap_note = _shap_summary(bundle, chosen) if include_shap else ([], "SHAP has not been requested. The importances below are not SHAP values.")
    return json_safe({"model": chosen, "importance": importance, "shap": shap_rows,
                      "shap_note": shap_note, "shap_available": bool(shap_rows),
                      "caveats": [CAVEAT, "Logistic odds ratios refer to regularized transformed-feature coefficients; they are not risk ratios or inferential confidence intervals.",
                                  "Tree importance has no direction and can be sensitive to correlated features and category counts."]})


def customer_explanation(bundle, row, model_name=None):
    chosen = _selected(bundle, model_name)
    if isinstance(row, dict):
        row = pd.DataFrame([row])
    if len(row) != 1:
        raise ValueError("Choose exactly one customer row for an individual explanation.")
    features, _ = prepare_features(row, bundle["configuration"], scoring=True)
    model = bundle["models"][chosen]
    probability = float(model.predict_proba(features)[0, 1])
    variants = pd.concat([row.copy() for _ in bundle["feature_columns"]], ignore_index=True)
    # Inputs may have integer dtype while training medians are fractional.
    # Raw variants are explicitly object typed; prepare_features restores types.
    variants = variants.astype(object)
    for index, name in enumerate(bundle["feature_columns"]):
        variants.loc[index, name] = bundle["training_baselines"].get(name)
    prepared, _ = prepare_features(variants, bundle["configuration"], scoring=True)
    alternatives = model.predict_proba(prepared)[:, 1]
    drivers = [{"feature": name, "value": row.iloc[0][name], "reference": bundle["training_baselines"].get(name),
                "contribution": probability-float(alternative)} for name, alternative in zip(bundle["feature_columns"], alternatives)]
    drivers.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return json_safe({"model": chosen, "probability": probability, "churn_probability": probability,
                      "risk_category": risk_category(probability), "drivers": drivers,
                      "method": "One-feature-at-a-time probability sensitivity of the selected model, including its calibration and actual ensemble when selected.",
                      "caveats": [CAVEAT, "Positive contribution means higher risk than replacing that input with its training median/mode reference. Effects do not sum to the score.",
                                  "Reference substitutions can be unusual combinations. They are explanations of model behavior, not feasible interventions or SHAP values."]})
