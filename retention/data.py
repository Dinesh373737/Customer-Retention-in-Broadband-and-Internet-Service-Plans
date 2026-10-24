"""Schema-aware IBM Telco CSV loading without fitting preprocessing on the data."""
from __future__ import annotations

import csv
import hashlib
import io
import os
from typing import Any, BinaryIO, TextIO

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORIES = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    **{column: ["No", "Yes", "No internet service"] for column in INTERNET_ADDONS},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """A content hash including column order and row order, used to invalidate caches."""
    payload = df.to_csv(index=False, lineterminator="\n", na_rep="<MISSING>").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_category(values: pd.Series, column: str) -> pd.Series:
    lookup = {value.casefold(): value for value in CATEGORIES[column]}
    if column == "SeniorCitizen":
        lookup.update({"0": "No", "0.0": "No", "1": "Yes", "1.0": "Yes"})
    stripped = values.astype("string").str.strip().replace("", pd.NA)
    keys = stripped.str.casefold()
    invalid = stripped.notna() & ~keys.isin(lookup)
    if invalid.any():
        examples = ", ".join(repr(value) for value in stripped[invalid].unique()[:5])
        raise ValueError(f"Unrecognised categories in {column}: {examples}. Expected: {', '.join(CATEGORIES[column])}.")
    # Object dtype with np.nan works across supported sklearn imputer versions.
    return keys.map(lookup).astype(object).where(stripped.notna(), np.nan)


def load_data(
    source: str | os.PathLike | bytes | BinaryIO | TextIO,
    require_target: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a CSV, validate its schema, and return cleaned data and an audit report.

    Numeric missing values are preserved. Imputation/encoding belong in a pipeline
    fitted on the training split. A label-free CSV is accepted in scoring mode.
    """
    try:
        if isinstance(source, bytes):
            contents = source
        elif hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            contents = source.read()
        else:
            with open(source, "rb") as csv_file:
                contents = csv_file.read()
        csv_text = contents.decode("utf-8-sig") if isinstance(contents, bytes) else contents.lstrip("\ufeff")
        # Inspect the actual header first: pandas automatically renames repeated
        # labels (e.g. tenure -> tenure.1), hiding an ambiguous input schema.
        header_reader = csv.reader(io.StringIO(csv_text, newline=""))
        for header in header_reader:
            if not header or (len(header) == 1 and not header[0].strip()):
                continue
            canonical_header = [column.strip().lstrip("\ufeff") for column in header]
            if len(canonical_header) != len(set(canonical_header)):
                raise ValueError("The CSV contains duplicate column names after trimming whitespace.")
            break
        raw = pd.read_csv(io.StringIO(csv_text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"The file could not be read as a UTF-8 CSV: {error}") from error
    raw.columns = raw.columns.astype(str).str.strip().str.lstrip("\ufeff")
    if raw.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names after trimming whitespace.")
    required = FEATURE_COLUMNS + (["Churn"] if require_target else [])
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    if raw.empty:
        raise ValueError("The CSV contains no customer records.")
    df = raw.copy()
    warnings: list[str] = []
    decisions = [
        "Trimmed whitespace in column names, identifiers and categorical values; normalised known category spelling/case.",
        "Converted SeniorCitizen to No/Yes and Churn (when supplied) to integer 0/1.",
        "Parsed tenure, MonthlyCharges and TotalCharges as numeric; left missing features for training-fitted imputation.",
        "Kept customerID only for record navigation; it must never enter the feature matrix.",
    ]
    allowed = {"customerID", "Churn", *FEATURE_COLUMNS}
    extra = [column for column in df.columns if column not in allowed]
    if extra:
        warnings.append("Extra columns excluded from the canonical analysis schema: " + ", ".join(extra))
    df = df[[column for column in ["customerID", *FEATURE_COLUMNS, "Churn"] if column in df]].copy()
    for column in CATEGORICAL_COLUMNS:
        df[column] = _normalize_category(df[column], column)
    for column in NUMERIC_COLUMNS:
        strings = df[column].astype("string").str.strip().replace("", pd.NA)
        numeric = pd.to_numeric(strings, errors="coerce").astype(float)
        invalid = strings.notna() & numeric.isna()
        if invalid.any():
            examples = ", ".join(repr(value) for value in strings[invalid].unique()[:5])
            raise ValueError(f"Invalid numeric values in {column}: {examples}.")
        finite = numeric.dropna()
        if (~np.isfinite(finite)).any() or (finite < 0).any():
            raise ValueError(f"{column} must contain finite, non-negative numeric values or blanks.")
        if column == "tenure" and ((finite % 1) != 0).any():
            raise ValueError("tenure must contain whole months or blanks.")
        df[column] = numeric
    if "Churn" in df:
        labels = df["Churn"].astype("string").str.strip().str.casefold()
        target_map = {"no": 0, "yes": 1, "0": 0, "1": 1, "0.0": 0, "1.0": 1}
        if labels.isna().any() or not labels.isin(target_map).all():
            raise ValueError("Churn labels must be non-missing Yes/No or 0/1 values.")
        df["Churn"] = labels.map(target_map).astype(int)
    if "customerID" in df:
        df["customerID"] = df["customerID"].astype("string").str.strip().replace("", pd.NA)
    duplicates = int(df.duplicated().sum())
    duplicates_removed = 0
    if duplicates and require_target:
        df = df.drop_duplicates().copy()
        duplicates_removed = duplicates
        warnings.append(f"Removed {duplicates} exact duplicate customer records before splitting data.")
        decisions.append("Removed exact duplicates to avoid repeating the same customer across train/test splits.")
    elif duplicates:
        warnings.append(f"Found {duplicates} exact duplicate records; retained every uploaded row in scoring mode.")
        decisions.append("Scoring preserves upload row count, including duplicates; no train/test split is performed.")
    if "customerID" not in df:
        df.insert(0, "customerID", pd.Series(pd.NA, index=df.index, dtype="string"))
    missing_ids = df["customerID"].isna()
    if missing_ids.any():
        existing = set(df["customerID"].dropna())
        generated = []
        for index in df.index[missing_ids]:
            identifier = f"Upload-record-{index + 1:06d}"
            while identifier in existing:
                identifier += "-generated"
            existing.add(identifier)
            generated.append(identifier)
        df.loc[missing_ids, "customerID"] = generated
        warnings.append(f"Generated navigation references for {int(missing_ids.sum())} missing customer IDs.")
    repeated_ids = int(df["customerID"].duplicated().sum())
    if repeated_ids:
        if require_target:
            raise ValueError("Conflicting duplicate customerID values found. Supply one record per customer to prevent train/test leakage.")
        warnings.append(f"Found {repeated_ids} repeated customer IDs; retained each uploaded row for scoring. Use row position to distinguish repeated references.")
    phone_inconsistent = (
        ((df["PhoneService"] == "No") & df["MultipleLines"].notna() & (df["MultipleLines"] != "No phone service"))
        | ((df["PhoneService"] == "Yes") & (df["MultipleLines"] == "No phone service"))
    )
    internet_inconsistent = pd.Series(False, index=df.index)
    for column in INTERNET_ADDONS:
        internet_inconsistent |= (
            ((df["InternetService"] == "No") & df[column].notna() & (df[column] != "No internet service"))
            | (df["InternetService"].isin(["DSL", "Fiber optic"]) & (df[column] == "No internet service"))
        )
    inconsistent_count = int((phone_inconsistent | internet_inconsistent).sum())
    if inconsistent_count:
        warnings.append(f"{inconsistent_count} records have inconsistent phone/internet service combinations; retained without inventing corrections.")
    missing_values = {column: int(df[column].isna().sum()) for column in df.columns}
    if any(missing_values.values()):
        warnings.append("Missing feature values remain; model pipelines fit imputation on training records only.")
    df = df.reset_index(drop=True)
    quality: dict[str, Any] = {
        "rows_loaded": int(len(raw)), "rows_cleaned": int(len(df)),
        "rows_before": int(len(raw)), "rows_after": int(len(df)),
        "columns": int(len(df.columns)), "missing_values": missing_values,
        "duplicate_records": duplicates, "duplicates_removed": duplicates_removed,
        "repeated_customer_ids": repeated_ids,
        "inconsistent_service_records": inconsistent_count, "extra_columns": extra,
        "target_present": "Churn" in df,
        "class_counts": {str(key): int(value) for key, value in df["Churn"].value_counts().sort_index().items()} if "Churn" in df else {},
        "warnings": warnings, "preprocessing_decisions": decisions,
        "fingerprint": dataset_fingerprint(df),
    }
    return df, quality
