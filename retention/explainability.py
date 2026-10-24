"""Explanations for fitted pipelines with a dependable, explicit fallback."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .ensemble import SoftVotingEnsemble
from .modeling import score_customers
from .preprocessing import predictor_frame


def _base_importance(model) -> pd.DataFrame:
    preprocessing = model.pipeline.named_steps["preprocessor"]
    estimator = model.pipeline.named_steps["classifier"]
    names = preprocessing.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        coefficient = np.asarray(estimator.coef_[0], dtype=float)
        reference = ["per training standard deviation" for _ in NUMERIC_COLUMNS]
        encoder = preprocessing.named_transformers_["categorical"].named_steps["encoder"]
        for column, levels in zip(CATEGORICAL_COLUMNS, encoder.categories_):
            reference.extend([f"category versus {column}={levels[0]}" for _ in levels[1:]])
        return pd.DataFrame({"feature": names, "importance": np.abs(coefficient), "coefficient": coefficient,
                             "odds_ratio": np.exp(coefficient), "reference": reference,
                             "method": "raw logistic coefficient (before optional probability calibration)"})
    importance = np.asarray(estimator.feature_importances_, dtype=float)
    return pd.DataFrame({"feature": names, "importance": importance, "coefficient": np.nan, "odds_ratio": np.nan,
                         "reference": "encoded predictor", "method": "raw estimator impurity/gain importance (not causal)"})


def global_importance(bundle: dict, model_name: str) -> pd.DataFrame:
    """Encoded-feature importance; coefficients/odds ratios refer to raw LR only.

    Numerical LR coefficients represent one training standard deviation.
    Categorical odds ratios compare the encoded level to the dropped first level.
    Tree importance and logistic magnitudes use different units and must not be
    compared directly. Hybrid importance is a weighted normalized heuristic.
    """
    model = bundle["models"][model_name]
    if not isinstance(model, SoftVotingEnsemble):
        return _base_importance(model).sort_values("importance", ascending=False).reset_index(drop=True)
    parts = []
    for weight, member in zip(model.weights, model.members.values()):
        table = _base_importance(member)
        total = float(table["importance"].sum())
        table["importance"] = table["importance"] * weight / total if total else 0.0
        parts.append(table[["feature", "importance"]])
    combined = pd.concat(parts).groupby("feature", as_index=False)["importance"].sum()
    combined["coefficient"] = np.nan
    combined["odds_ratio"] = np.nan
    combined["reference"] = "encoded predictor"
    combined["method"] = "equal-weight normalized member importance; descriptive heuristic, not SHAP"
    return combined.sort_values("importance", ascending=False).reset_index(drop=True)


def local_explanation(bundle: dict, customer_df: pd.DataFrame, model_name: str):
    """One-feature-at-a-time probability perturbations of the actual chosen model.

    Returns (table, description). Positive contribution means this customer's
    actual feature yields higher risk than substituting its training median/mode,
    while all other inputs stay fixed. Effects need not sum to the prediction and
    a replacement may describe an unusual account. This is never causal evidence.
    """
    if len(customer_df) != 1:
        raise ValueError("Select exactly one customer for a local explanation.")
    customer = predictor_frame(customer_df)
    probability = float(score_customers(bundle, customer, model_name)[0])
    variants = pd.concat([customer.copy() for _ in bundle["feature_columns"]], ignore_index=True)
    for row, feature in enumerate(bundle["feature_columns"]):
        variants.loc[row, feature] = bundle["training_baselines"][feature]
    baseline_probabilities = score_customers(bundle, variants, model_name)
    rows = [{"feature": feature, "value": customer.iloc[0][feature],
             "baseline": bundle["training_baselines"][feature], "contribution": probability - float(baseline_probability),
             "method": "probability change from replacing one feature with training median/mode"}
            for feature, baseline_probability in zip(bundle["feature_columns"], baseline_probabilities)]
    table = pd.DataFrame(rows)
    table = table.iloc[table["contribution"].abs().argsort()[::-1]].reset_index(drop=True)
    method = ("One-feature perturbation of the selected model's actual probability, including any calibration and ensemble. "
              "Positive values indicate higher predicted risk relative to the training median/mode replacement. "
              "These are non-additive model sensitivities, not SHAP values or evidence that an intervention will prevent churn.")
    return table, method


def shap_global_summary(bundle: dict, df: pd.DataFrame, model_name: str, max_rows: int = 100):
    """Return (long SHAP table, description), or (None, explicit fallback reason).

    Table columns are sample (input index), feature (encoded name), shap_value,
    and value (transformed numeric input). SHAP describes the selected raw base
    estimator, before optional sigmoid calibration. The actual scored model is
    always used for the separate probability-perturbation local explanation.
    """
    model = bundle["models"][model_name]
    if isinstance(model, SoftVotingEnsemble):
        return None, "SHAP is not computed for the probability ensemble. Use the member importance summary and actual ensemble probability perturbations below."
    if len(df) == 0:
        return None, "No customer rows are available for SHAP."
    try:
        import shap
    except ImportError:
        return None, "SHAP is not installed. The fitted model importance and probability perturbation explanations remain available."
    try:
        subset = predictor_frame(df).sample(n=min(max_rows, len(df)), random_state=bundle["metadata"]["random_state"])
        preprocessing = model.pipeline.named_steps["preprocessor"]
        estimator = model.pipeline.named_steps["classifier"]
        transformed = preprocessing.transform(subset)
        names = preprocessing.get_feature_names_out()
        if hasattr(estimator, "coef_"):
            background = preprocessing.transform(bundle["explanation_background"])
            explainer = shap.LinearExplainer(estimator, background)
            values = explainer.shap_values(transformed)
            units = "raw logistic log odds"
        else:
            explainer = shap.TreeExplainer(estimator)
            values = explainer.shap_values(transformed)
            units = "raw forest probability" if model_name == "Random Forest" else "raw XGBoost log odds"
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        if values.shape != transformed.shape:
            raise ValueError(f"Unsupported SHAP output shape {values.shape}")
        table = pd.DataFrame({
            "sample": np.repeat(subset.index.to_numpy(), len(names)),
            "feature": np.tile(names, len(subset)),
            "shap_value": values.reshape(-1),
            "value": np.asarray(transformed).reshape(-1),
        })
        description = (f"SHAP on {len(subset)} reproducibly sampled records; units: {units}. "
                       "Categorical inputs are one-hot encoded and numeric inputs standardized. "
                       "This explains the fitted raw base model before optional calibration, not intervention effects.")
        return table, description
    except Exception as exc:
        return None, f"SHAP could not explain this estimator ({type(exc).__name__}: {exc}). Use fitted model importance and actual probability perturbations."


shap_summary = shap_global_summary
