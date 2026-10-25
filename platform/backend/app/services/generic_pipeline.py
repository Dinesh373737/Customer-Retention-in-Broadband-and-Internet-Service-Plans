"""Dataset-specific sklearn transformers and serializable probability models."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_preprocessor(config):
    numeric = [f["column"] for f in config["features"] if f["kind"] in {"numeric", "datetime"}]
    categorical = [f["column"] for f in config["features"] if f["kind"] == "categorical"]
    ordinal = [f for f in config["features"] if f["kind"] == "ordinal"]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="(Missing)", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3, max_categories=30, drop="first", sparse_output=False)),
        ]), categorical))
    if ordinal:
        transformers.append(("ordinal", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="(Missing)", keep_empty_features=True)),
            ("encoder", OrdinalEncoder(categories=[f["ordinal_order"] for f in ordinal], handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), [f["column"] for f in ordinal]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def logit_probability(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1-1e-6)
    return np.log(values/(1-values)).reshape(-1, 1)


@dataclass
class ProbabilityModel:
    pipeline: object
    calibrator: object | None = None

    def predict_proba(self, frame):
        positive = self.pipeline.predict_proba(frame)[:, 1]
        if self.calibrator is not None:
            positive = self.calibrator.predict_proba(logit_probability(positive))[:, 1]
        return np.column_stack([1-positive, positive])


@dataclass
class HybridProbabilityModel:
    members: dict
    weights: tuple = (1/3, 1/3, 1/3)

    def predict_proba(self, frame):
        weights = np.asarray(self.weights)
        if len(weights) != len(self.members) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
            raise ValueError("Hybrid weights must be finite, nonnegative and sum to one.")
        positive = sum(weight*member.predict_proba(frame)[:, 1] for weight, member in zip(weights, self.members.values()))
        return np.column_stack([1-positive, positive])
