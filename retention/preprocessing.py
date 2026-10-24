"""Predictor-only preprocessing, fitted inside each training/CV pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def predictor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the documented predictors; never expose identifiers or targets."""
    missing = sorted(set(FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Missing prediction columns: " + ", ".join(missing))
    predictors = frame.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_COLUMNS:
        predictors[column] = pd.to_numeric(predictors[column], errors="coerce")
        predictors[column] = predictors[column].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_COLUMNS:
        # Object + np.nan is compatible with sklearn's categorical imputer.
        predictors[column] = predictors[column].astype(object)
        predictors[column] = predictors[column].where(predictors[column].notna(), np.nan)
    return predictors


def make_preprocessor() -> ColumnTransformer:
    """Return an unfitted transformer. Medians/modes/categories come from training."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_COLUMNS)),
        ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
    ], remainder="drop", verbose_feature_names_out=False)


def training_baselines(predictors: pd.DataFrame) -> dict:
    """One representative value per feature for explicit perturbation explanations."""
    baselines = {}
    for column in NUMERIC_COLUMNS:
        median = predictors[column].median()
        baselines[column] = float(median) if pd.notna(median) else 0.0
    for column in CATEGORICAL_COLUMNS:
        modes = predictors[column].mode(dropna=True)
        baselines[column] = modes.iloc[0] if len(modes) else "missing"
    return baselines
