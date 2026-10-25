"""Schema-confirmed tabular ingestion. This module has no legacy dependencies."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

MAX_FILE_BYTES = 30 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 200
TARGET_NAMES = {"churn", "exited", "cancelled", "canceled", "attrition", "left", "is_churn", "churned"}
KINDS = {"numeric", "categorical", "ordinal", "datetime"}
DOMAINS = {"telecommunications", "banking", "ecommerce", "subscription", "generic"}
CONCEPTS = {"pricing", "lifecycle", "support", "engagement", "contract", "satisfaction", "loyalty"}
RESERVED_OUTPUT_COLUMNS = {"row_index", "customer_reference", "churn_probability", "risk_category", "actual_outcome", "actual_churn"}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray, pd.Index)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_label(value) -> str:
    if pd.isna(value):
        raise ValueError("Target labels cannot be missing.")
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, float, np.number)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError("Target labels cannot be blank.")
    return text


def _name_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_identifier(name):
    key = _name_key(name)
    return key in {"id", "rownumber", "rowid", "recordid", "index", "customerid", "customeridentifier", "accountid", "userid", "subscriberid", "clientid", "memberid", "accountnumber", "customerreference", "customerref", "clientref"} or key.endswith("identifier")


def is_protected(name):
    key = _name_key(name)
    return key in {"age", "agegroup", "gender", "sex", "race", "ethnicity", "religion", "disability", "maritalstatus", "seniorcitizen", "geography", "nationality", "zipcode", "citytier", "estimatedsalary", "income"} or any(word in key for word in ["birthdate", "dateofbirth", "ethnicity", "religion"])


def is_sensitive(name):
    key = _name_key(name)
    return is_protected(name) or key in {"surname", "firstname", "lastname", "fullname", "name", "email", "phonenumber", "address", "ssn", "passport", "accountnumber", "creditcardnumber"}


def leakage_warning(name):
    key = _name_key(name)
    if key in {_name_key(value) for value in TARGET_NAMES} | {"status", "accountstatus", "customerstatus", "churnreason", "cancellationreason", "churndate", "cancellationdate", "terminationdate", "finalstatus", "exitedflag", "retentionoutcome"} or any(word in key for word in ["afterchurn", "postoutcome", "cancelreason", "terminationreason"]):
        return "Possible post-outcome information. Exclude unless its availability before prediction is documented and approved."
    return None


def _checked_headers(headers):
    names = [str(value).strip().lstrip("\ufeff") if value is not None else "" for value in headers]
    if not names or any(not name for name in names):
        raise ValueError("Every column needs a non-empty header.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate column headers are ambiguous; give every column a unique name.")
    if len(names) > MAX_COLUMNS:
        raise ValueError(f"At most {MAX_COLUMNS} columns are supported.")
    return names


def read_table(contents: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    if not isinstance(contents, bytes) or not contents:
        raise ValueError("The uploaded file is empty.")
    if len(contents) > MAX_FILE_BYTES:
        raise ValueError("Uploads are limited to 30 MiB.")
    extension = Path(filename).suffix.lower()
    if extension == ".csv":
        try:
            text = contents.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV files must use UTF-8 encoding. Export as UTF-8 CSV and retry.") from exc
        try:
            rows = csv.reader(io.StringIO(text), strict=True)
            header = next((row for row in rows if any(value.strip() for value in row)), None)
            if header is None:
                raise ValueError("The CSV has no header or records.")
            names = _checked_headers(header)
            expected = len(names)
            records = []
            for line, row in enumerate(rows, 2):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != expected:
                    raise ValueError(f"CSV row {line} has {len(row)} fields; expected {expected}.")
                records.append(row)
                if len(records) > MAX_ROWS:
                    raise ValueError(f"At most {MAX_ROWS:,} records are supported.")
            if not records:
                raise ValueError("The file contains headers but no customer records.")
            # Let pandas infer numeric fields while preserving string labels and blanks.
            normalized = io.StringIO()
            writer = csv.writer(normalized)
            writer.writerow(names)
            writer.writerows(records)
            frame = pd.read_csv(io.StringIO(normalized.getvalue()))
        except (csv.Error, pd.errors.ParserError) as exc:
            raise ValueError(f"The CSV is corrupt or malformed: {exc}") from exc
    elif extension == ".xlsx":
        try:
            with zipfile.ZipFile(io.BytesIO(contents)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 200 * 1024 * 1024:
                    raise ValueError("The expanded workbook exceeds the supported size limit.")
            import openpyxl
            workbook = openpyxl.load_workbook(io.BytesIO(contents), read_only=True, data_only=True)
            sheets = workbook.sheetnames
            if len(sheets) > 1 and not sheet_name:
                workbook.close()
                raise ValueError("This workbook contains multiple worksheets. Select a worksheet explicitly: " + ", ".join(sheets) + ".")
            chosen = sheet_name or sheets[0]
            if chosen not in sheets:
                raise ValueError(f"Worksheet '{chosen}' was not found. Available worksheets: {', '.join(sheets)}.")
            worksheet = workbook[chosen]
            if worksheet.max_row > MAX_ROWS + 1 or worksheet.max_column > MAX_COLUMNS:
                raise ValueError("Workbook dimensions exceed the 100,000-row / 200-column limits.")
            rows = worksheet.iter_rows(values_only=True)
            header = next(rows, None)
            if header is None:
                raise ValueError("The workbook worksheet is empty.")
            names = _checked_headers(header)
            records = [row for row in rows if any(value is not None for value in row)]
            workbook.close()
            frame = pd.DataFrame(records, columns=names)
            frame.attrs.update(worksheets=sheets, worksheet=chosen)
        except ImportError as exc:
            raise ValueError("Excel support requires openpyxl. Install the platform requirements.") from exc
        except (zipfile.BadZipFile, OSError, KeyError, SyntaxError) as exc:
            raise ValueError("The file is not a valid .xlsx workbook.") from exc
    else:
        raise ValueError("Supported uploads are UTF-8 .csv and .xlsx workbooks only.")
    if frame.empty:
        raise ValueError("The selected table has no customer records.")
    frame.columns = _checked_headers(frame.columns)
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        frame[column] = frame[column].replace("", np.nan)
    return frame.reset_index(drop=True)


def frame_fingerprint(df):
    payload = df.to_csv(index=False, na_rep="<MISSING>", lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def configuration_fingerprint(config):
    clean = {k: v for k, v in config.items() if k not in {"warnings", "decisions"}}
    return hashlib.sha256(json.dumps(json_safe(clean), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def profile_dataframe(df):
    columns, targets, warnings = [], [], []
    for name in df.columns:
        series = df[name]
        present = series.dropna()
        count = int(present.nunique())
        numeric = pd.to_numeric(present, errors="coerce")
        if not len(present):
            kind = "ambiguous"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kind = "datetime"
        elif count == 2:
            kind = "binary"
        elif numeric.notna().all():
            kind = "numeric"
        elif len(present) >= 3 and present.astype(str).str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > .9:
            kind = "datetime"
        else:
            kind = "categorical"
        target = _name_key(name) in {_name_key(item) for item in TARGET_NAMES} and 1 <= count <= 2
        if target:
            targets.append(name)
        identifier = is_identifier(name)
        sensitive = is_sensitive(name)
        leak = leakage_warning(name)
        if identifier:
            role = "identifier"
        elif target:
            role = "potential_target"
        elif sensitive or leak or count <= 1 or name in RESERVED_OUTPUT_COLUMNS:
            role = "exclude"
        else:
            role = "feature"
        columns.append({"name": str(name), "inferred_type": kind,
                        "missing": int(series.isna().sum()), "missing_rate": float(series.isna().mean()),
                        "unique": count, "sample_values": json_safe(present.drop_duplicates().head(8).tolist()),
                        "identifier": identifier, "sensitive": sensitive, "protected": is_protected(name),
                        "leakage_warning": leak, "suggested_role": role,
                        "constant": count <= 1, "high_cardinality": bool(count > 100),
                        "target_labels": sorted({canonical_label(value) for value in present}) if count <= 20 else []})
    if int(df.duplicated().sum()):
        warnings.append("Exact duplicate rows are present. Training removes exact duplicate observations before splitting; batch scoring preserves every uploaded row.")
    if len(df) < 60:
        warnings.append("Fewer than 60 rows: exploratory analysis is available, but reliable three-way model evaluation is not supported.")
    warnings.append("Confirm the churn definition, observation timing and prediction horizon; inference suggestions are not confirmed schema mappings.")
    return {"rows": int(len(df)), "columns_count": int(len(df.columns)),
            "missing_cells": int(df.isna().sum().sum()), "duplicate_rows": int(df.duplicated().sum()),
            "columns": columns, "target_candidates": targets, "warnings": warnings,
            "preview": json_safe(df.head(20).to_dict("records")),
            "worksheets": df.attrs.get("worksheets", []), "worksheet": df.attrs.get("worksheet")}


def target_values(df, config):
    target = config.get("target_column")
    if not target or target not in df:
        raise ValueError("A historical target column is required for training or outcome analysis.")
    labels = df[target].map(canonical_label)
    positive = canonical_label(config.get("positive_label"))
    known = set(config.get("target_labels") or labels.unique())
    if len(known) != 2 or positive not in known or set(labels) - known:
        raise ValueError("Choose a positive label from exactly two non-missing target classes.")
    return (labels == positive).astype(int)


def validate_configuration(df, config):
    result = copy.deepcopy(config)
    target = result.get("target_column") or None
    identifier = result.get("customer_id") or None
    excluded = result.setdefault("excluded_columns", [])
    result.setdefault("features", [])
    result.setdefault("business_domain", "generic")
    result.setdefault("prediction_horizon", None)
    result.setdefault("semantics", [])
    result.setdefault("leakage_approved", {})
    result["target_column"] = target
    result["customer_id"] = identifier
    if result["business_domain"] not in DOMAINS:
        raise ValueError("Choose one of the supported business domains.")
    if result.get("split_strategy", "stratified") != "stratified":
        raise ValueError("This release supports independent customer-level stratified splits only. Temporal/group-aware training is not implemented.")
    named = ([target] if target else []) + ([identifier] if identifier else []) + excluded
    missing = sorted(set(named) - set(df.columns))
    if missing:
        raise ValueError("Configured columns were not found: " + ", ".join(missing))
    if target and target == identifier:
        raise ValueError("The target cannot also be the customer identifier.")
    warnings, decisions = [], []
    if target:
        observed = df[target].map(canonical_label)
        if observed.nunique() != 2:
            raise ValueError("A historical target must contain exactly two non-missing classes.")
        result["target_labels"] = sorted(observed.unique().tolist())
        y = target_values(df, result)
        result["positive_label"] = canonical_label(result["positive_label"])
        result["target_labels"] = sorted({canonical_label(value) for value in df[target]})
        result["class_balance"] = {"positive": int(y.sum()), "negative": int((1-y).sum()), "positive_rate": float(y.mean())}
        if y.value_counts().min() < 12:
            warnings.append("Fewer than 12 examples in one target class: descriptive analysis works, but training is blocked.")
        if y.mean() < .05 or y.mean() > .95:
            warnings.append("Severe class imbalance: PR-AUC, class counts and calibration need careful interpretation.")
    else:
        result["positive_label"] = None
        result["target_labels"] = []
        result["class_balance"] = None
        y = None
        decisions.append("No target configured: exploratory-only analysis cannot provide predictions or evaluation without a compatible fitted model.")
    features = result["features"]
    names = [item.get("column") for item in features]
    if len(set(names)) != len(names):
        raise ValueError("Each predictor column must occur only once in the feature mapping.")
    if set(names) - set(df.columns):
        raise ValueError("A configured predictor column is absent from the dataset.")
    if len(features) > 100:
        raise ValueError("At most 100 selected predictors are supported on a standard laptop.")
    for feature in features:
        name, kind = feature["column"], feature.get("kind")
        if name in RESERVED_OUTPUT_COLUMNS or name.startswith("__"):
            raise ValueError(f"'{name}' is reserved for platform output. Rename it before using it as a predictor.")
        if name in {target, identifier} or name in excluded or is_identifier(name):
            raise ValueError(f"'{name}' is an identifier, target or excluded column and cannot be a predictor.")
        if is_sensitive(name) and not is_protected(name):
            raise ValueError(f"'{name}' contains names or direct personal identifiers. Exclude it from predictors.")
        if is_protected(name):
            warnings.append(f"'{name}' is a sensitive/protected predictor. Review fairness and segment errors; it cannot drive retention actions or what-if changes.")
        if kind not in KINDS:
            raise ValueError(f"Choose numeric, categorical, ordinal or datetime for '{name}'.")
        present = df[name].dropna()
        if present.nunique() <= 1:
            raise ValueError(f"'{name}' is constant or entirely missing. Exclude it explicitly.")
        if len(present) and present.value_counts(normalize=True).iloc[0] >= .98:
            warnings.append(f"'{name}' is near-constant; its predictive contribution may be limited.")
        leak = leakage_warning(name)
        if leak and not str(result["leakage_approved"].get(name, "")).strip():
            raise ValueError(f"'{name}' may be post-outcome information. Exclude it or document why it is known before prediction in leakage_approved.")
        if leak:
            warnings.append(f"Timing review for '{name}': {result['leakage_approved'][name]}")
        if kind == "numeric":
            numeric = pd.to_numeric(present, errors="coerce")
            if numeric.isna().any() or not np.isfinite(numeric).all():
                raise ValueError(f"'{name}' contains nonnumeric or nonfinite values; correct the type or data.")
        elif kind in {"categorical", "ordinal"}:
            if present.astype(str).str.len().gt(200).any():
                raise ValueError(f"'{name}' contains long raw text; derive a supported structured field or exclude it.")
            if kind == "categorical" and present.nunique() > 500:
                raise ValueError(f"'{name}' has over 500 categories. Exclude identifiers/raw text or supply a meaningful lower-cardinality field.")
            if kind == "categorical" and present.nunique() > 100:
                warnings.append(f"'{name}' has high cardinality; training groups infrequent categories and limits encoded columns.")
            if kind == "ordinal":
                order = feature.get("ordinal_order", [])
                order = [canonical_label(value) for value in order]
                if not order or len(set(order)) != len(order) or set(present.map(canonical_label)) - set(order):
                    raise ValueError(f"'{name}' needs a unique ordinal_order covering every observed level.")
                feature["ordinal_order"] = order
        elif kind == "datetime":
            if not feature.get("date_reference"):
                raise ValueError(f"'{name}' needs an explicit fixed date_reference known at prediction time.")
            reference = pd.to_datetime(feature["date_reference"], errors="coerce", utc=True)
            dates = pd.to_datetime(present, errors="coerce", utc=True)
            if pd.isna(reference) or dates.isna().any() or (dates > reference).any():
                raise ValueError(f"'{name}' requires valid dates not after its fixed prediction reference date.")
            decisions.append(f"'{name}' becomes elapsed days before {reference.isoformat()}; no outcome-derived or current-clock reference is used.")
        # Exact numeric equality or deterministic low-cardinality mapping reveals target proxies.
        if y is not None:
            pairs = pd.DataFrame({"x": df[name].map(lambda value: '<missing>' if pd.isna(value) else str(value)), "y": y})
            grouped = pairs.groupby("x")
            repeats_support_mapping = pairs.x.nunique() == 2 or grouped.size().min() >= 3
            if 1 < pairs.x.nunique() <= 20 and repeats_support_mapping and grouped.y.nunique().max() == 1 and pairs.y.nunique() == 2:
                raise ValueError(f"'{name}' deterministically reveals the target in this dataset; exclude this potential target proxy.")
            if kind == "numeric":
                numeric = pd.to_numeric(df[name], errors="coerce")
                valid = numeric.notna()
                if valid.sum() > 2 and (np.allclose(numeric[valid], y[valid]) or np.allclose(numeric[valid], 1-y[valid])):
                    raise ValueError(f"'{name}' copies the target or its inverse and cannot be a predictor.")
    if identifier:
        if df[identifier].isna().any():
            warnings.append("Some customer identifiers are missing; display references will be generated without using identifiers as predictors.")
        repeated = df.loc[df[identifier].notna(), identifier].duplicated().sum()
        if repeated:
            warnings.append("Repeated customer identifiers require deduplication; conflicting repeated customers are blocked from independent-customer training.")
    if not result.get("prediction_horizon"):
        warnings.append("The prediction horizon is unknown. This dataset does not establish the timing of churn or intervention opportunity.")
    for semantic in result["semantics"]:
        if semantic.get("column") not in df or semantic.get("concept") not in CONCEPTS:
            raise ValueError("Each semantic mapping needs an existing column and a supported business concept.")
        if semantic.get("confirmed") and not str(semantic.get("meaning", "")).strip():
            raise ValueError("Confirmed semantic mappings require a description of the column's business meaning.")
        if semantic.get("estimated_cost") is not None and (not np.isfinite(semantic["estimated_cost"]) or semantic["estimated_cost"] < 0):
            raise ValueError("Estimated intervention costs must be finite and nonnegative.")
        if is_protected(semantic["column"]) or is_identifier(semantic["column"]) or is_sensitive(semantic["column"]):
            semantic["protected"] = True
            semantic["actionable"] = False
            warnings.append(f"'{semantic['column']}' is sensitive/immutable and cannot drive intervention or what-if actions.")
    decisions.extend(["Targets and identifiers are explicitly excluded from predictors.",
                      "All learned imputers, encoders, scalers and category grouping are fitted on training records only.",
                      "Exact duplicate historical records are removed before splitting; batch scoring preserves row cardinality."])
    result["warnings"], result["decisions"] = list(dict.fromkeys(warnings)), list(dict.fromkeys(decisions))
    return result


def prepare_features(df, configuration, scoring=False):
    names = [item["column"] for item in configuration["features"]]
    missing = set(names) - set(df.columns)
    if missing:
        raise ValueError("The model schema requires missing columns: " + ", ".join(sorted(missing)))
    frame = pd.DataFrame(index=df.index)
    warnings = []
    for feature in configuration["features"]:
        name, kind = feature["column"], feature["kind"]
        values = df[name]
        if kind == "numeric":
            numeric = pd.to_numeric(values, errors="coerce")
            if (values.notna() & numeric.isna()).any() or not np.isfinite(numeric.dropna()).all():
                raise ValueError(f"Scoring column '{name}' must contain finite numeric values or blanks.")
            frame[name] = numeric.astype(float)
        elif kind == "datetime":
            dates = pd.to_datetime(values, errors="coerce", utc=True)
            reference = pd.to_datetime(feature["date_reference"], utc=True)
            if (values.notna() & dates.isna()).any() or (dates > reference).any():
                raise ValueError(f"Scoring dates in '{name}' must be valid and no later than the saved reference date.")
            frame[name] = (reference-dates).dt.total_seconds() / 86400
        else:
            frame[name] = values.map(lambda value: np.nan if pd.isna(value) else canonical_label(value)).astype(object)
            if kind == "ordinal":
                unsupported = set(frame[name].dropna()) - set(feature["ordinal_order"])
                if unsupported:
                    raise ValueError(f"Ordinal values in '{name}' are outside the saved order: {sorted(unsupported)}.")
    return frame, warnings
