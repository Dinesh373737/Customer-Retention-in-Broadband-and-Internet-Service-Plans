import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retention.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, dataset_fingerprint, load_data

DATASET = Path(__file__).resolve().parents[1] / "data" / "Telco-Customer-Churn.csv"


@pytest.fixture
def sample():
    return pd.read_csv(DATASET).head(10)


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def csv_source(text, kind, tmp_path):
    if kind in {"path", "string_path"}:
        destination = tmp_path / "customers.csv"
        destination.write_text(text, encoding="utf-8")
        return destination if kind == "path" else str(destination)
    if kind == "bytes":
        return text.encode("utf-8")
    stream = io.BytesIO(text.encode("utf-8")) if kind == "binary_stream" else io.StringIO(text)
    stream.seek(5)  # The loader resets seekable uploads before inspecting headers.
    return stream


@pytest.mark.parametrize("kind", ["path", "string_path", "bytes", "binary_stream", "text_stream"])
def test_duplicate_raw_headers_are_rejected_before_pandas_mangles_names(sample, tmp_path, kind):
    duplicate = pd.concat([sample, sample[["tenure"]]], axis=1)
    # Quoted headers and a BOM must not hide the exact duplicate label.
    text = "\ufeff\n" + duplicate.to_csv(index=False, quoting=csv.QUOTE_ALL)
    with pytest.raises(ValueError, match="duplicate column names"):
        load_data(csv_source(text, kind, tmp_path))


@pytest.mark.parametrize("kind", ["path", "string_path", "bytes", "binary_stream", "text_stream"])
def test_all_supported_source_types_still_load_valid_csv(sample, tmp_path, kind):
    text = "\ufeff" + sample.to_csv(index=False)
    cleaned, quality = load_data(csv_source(text, kind, tmp_path))
    assert len(cleaned) == len(sample)
    assert quality["rows_loaded"] == len(sample)


def test_whitespace_header_collision_is_rejected(sample):
    duplicate = sample.copy()
    duplicate[" tenure "] = duplicate["tenure"]
    with pytest.raises(ValueError, match="duplicate column names"):
        load_data(csv_bytes(duplicate))


def test_actual_dataset_is_valid_and_auditable():
    cleaned, quality = load_data(DATASET)
    assert len(cleaned) > 100
    assert set(cleaned["Churn"].unique()) == {0, 1}
    assert set(cleaned["SeniorCitizen"].unique()) == {"No", "Yes"}
    assert set(FEATURE_COLUMNS) == set(NUMERIC_COLUMNS) | set(CATEGORICAL_COLUMNS)
    assert len(FEATURE_COLUMNS) == 19
    assert "customerID" not in FEATURE_COLUMNS and "Churn" not in FEATURE_COLUMNS
    assert quality["rows_cleaned"] == len(cleaned)
    assert quality["missing_values"]["TotalCharges"] == cleaned["TotalCharges"].isna().sum()
    json.dumps(quality, allow_nan=False)


def test_scoring_accepts_label_free_and_identifier_free_upload(sample):
    source = sample.drop(columns=["Churn", "customerID"])
    cleaned, quality = load_data(io.BytesIO(csv_bytes(source)), require_target=False)
    assert "Churn" not in cleaned
    assert cleaned["customerID"].is_unique
    assert quality["target_present"] is False
    with pytest.raises(ValueError, match="Missing required columns: Churn"):
        load_data(csv_bytes(source))


def test_missing_schema_and_unknown_category_are_actionable(sample):
    with pytest.raises(ValueError, match="MonthlyCharges"):
        load_data(csv_bytes(sample.drop(columns="MonthlyCharges")))
    sample.loc[0, "Contract"] = "Lifetime contract"
    with pytest.raises(ValueError, match="Unrecognised categories in Contract"):
        load_data(csv_bytes(sample))


def test_whitespace_categories_and_blank_numeric_are_normalised(sample):
    sample["SeniorCitizen"] = sample["SeniorCitizen"].astype(object)
    sample.loc[0, "gender"] = " female "
    sample.loc[0, "SeniorCitizen"] = " YES "
    sample.loc[0, "Churn"] = " yes "
    sample.loc[0, "TotalCharges"] = "   "
    cleaned, quality = load_data(csv_bytes(sample))
    assert cleaned.loc[0, "gender"] == "Female"
    assert cleaned.loc[0, "SeniorCitizen"] == "Yes"
    assert cleaned.loc[0, "Churn"] == 1
    assert np.isnan(cleaned.loc[0, "TotalCharges"])
    assert quality["missing_values"]["TotalCharges"] == 1


@pytest.mark.parametrize("column,value", [("MonthlyCharges", -1), ("tenure", 2.5), ("TotalCharges", "unparseable"), ("TotalCharges", np.inf), ("Churn", "Maybe"), ("Churn", np.nan)])
def test_invalid_numbers_and_labels_rejected(sample, column, value):
    sample[column] = sample[column].astype(object)
    sample.loc[0, column] = value
    with pytest.raises(ValueError):
        load_data(csv_bytes(sample))


def test_duplicate_removal_and_conflicting_ids_prevent_repeated_customers(sample):
    duplicate = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    cleaned, quality = load_data(csv_bytes(duplicate))
    assert len(cleaned) == len(sample)
    assert quality["duplicates_removed"] == 1
    conflicting = sample.copy()
    conflicting.loc[1, "customerID"] = conflicting.loc[0, "customerID"]
    with pytest.raises(ValueError, match="Conflicting duplicate customerID"):
        load_data(csv_bytes(conflicting))


def test_service_inconsistencies_reported_without_inventing_values(sample):
    sample.loc[0, ["PhoneService", "MultipleLines"]] = ["No", "Yes"]
    cleaned, quality = load_data(csv_bytes(sample))
    assert quality["inconsistent_service_records"] == 1
    assert cleaned.loc[0, "MultipleLines"] == "Yes"


def test_batch_scoring_preserves_all_rows_even_with_repeated_ids(sample):
    source = pd.concat([sample, sample.iloc[[0]]], ignore_index=True).drop(columns="Churn")
    source.loc[1, "customerID"] = source.loc[0, "customerID"]
    cleaned, quality = load_data(csv_bytes(source), require_target=False)
    assert len(cleaned) == len(source)
    assert quality["duplicate_records"] == 1
    assert quality["duplicates_removed"] == 0
    assert quality["repeated_customer_ids"] == 2
    assert any("scoring" in message for message in quality["warnings"])


def test_fingerprint_tracks_real_content_changes(sample):
    cleaned, _ = load_data(csv_bytes(sample))
    unchanged = cleaned.copy()
    modified = cleaned.copy()
    modified.loc[0, "MonthlyCharges"] += 1
    assert dataset_fingerprint(cleaned) == dataset_fingerprint(unchanged)
    assert dataset_fingerprint(cleaned) != dataset_fingerprint(modified)
