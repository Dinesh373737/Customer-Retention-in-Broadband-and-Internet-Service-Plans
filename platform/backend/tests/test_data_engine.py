"""Synthetic fixtures test validation behavior; they are not benchmark data."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.services.data_engine import read_table, profile_dataframe, validate_configuration, prepare_features, target_values, frame_fingerprint, configuration_fingerprint


def fixture_frame(rows=180):
    rng = np.random.default_rng(41)
    return pd.DataFrame({"CustomerID": [f"C{i}" for i in range(rows)], "fee": rng.uniform(10, 100, rows),
                         "months": rng.integers(1, 60, rows), "plan": rng.choice(["Monthly", "Annual", "None"], rows),
                         "Exited": rng.choice(["Stayed", "Left"], rows), "rating": rng.choice(["Low", "Mid", "High"], rows)})


def fixture_config():
    return {"target_column": "Exited", "positive_label": "Left", "customer_id": "CustomerID", "business_domain": "subscription",
            "excluded_columns": ["rating"], "features": [{"column": "fee", "kind": "numeric"}, {"column": "months", "kind": "numeric"}, {"column": "plan", "kind": "categorical"}], "semantics": []}


def test_malformed_excel_xml_is_a_validation_error():
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types><malformed")
    with pytest.raises(ValueError, match="valid .xlsx"):
        read_table(payload.getvalue(), "broken.xlsx")


@pytest.mark.parametrize("name", ["row_index", "churn_probability", "risk_category", "actual_churn", "__pair"])
def test_internal_output_columns_cannot_silently_replace_predictors(name):
    frame = fixture_frame()
    frame[name] = np.linspace(1, 25, len(frame))
    config = fixture_config()
    config["features"].append({"column": name, "kind": "numeric"})
    with pytest.raises(ValueError, match="reserved"):
        validate_configuration(frame, config)


@pytest.mark.parametrize("name", ["Surname", "Email", "PhoneNumber", "FullName"])
def test_direct_personal_identifiers_are_not_predictors(name):
    frame = fixture_frame()
    frame[name] = [f"person-{i % 8}" for i in range(len(frame))]
    config = fixture_config()
    config["features"].append({"column": name, "kind": "categorical"})
    with pytest.raises(ValueError, match="personal identifiers"):
        validate_configuration(frame, config)


def test_small_distinct_numeric_sample_is_not_mistaken_for_a_target_proxy():
    frame = pd.DataFrame({"fee": [11, 23, 38, 42, 56, 69], "cancelled": [0, 1, 0, 0, 1, 0]})
    config = {"target_column": "cancelled", "positive_label": "1", "features": [{"column": "fee", "kind": "numeric"}]}
    actual = validate_configuration(frame, config)
    assert actual["features"][0]["column"] == "fee"
    assert any("Fewer than 12" in warning for warning in actual["warnings"])


def test_csv_profile_and_schema_do_not_assume_ibm_names_or_last_column():
    original = fixture_frame()
    loaded = read_table(original.to_csv(index=False).encode(), "customers.csv")
    profile = profile_dataframe(loaded)
    assert profile["rows"] == 180 and profile["columns_count"] == 6
    assert profile["target_candidates"] == ["Exited"]
    assert next(c for c in profile["columns"] if c["name"] == "Exited")["target_labels"] == ["Left", "Stayed"]
    validated = validate_configuration(loaded, fixture_config())
    assert validated["positive_label"] == "Left"
    assert validated["class_balance"]["positive"] == int(original.Exited.eq("Left").sum())
    json.dumps(profile, allow_nan=False)


@pytest.mark.parametrize("content,name,message", [(b"", "x.csv", "empty"), (b"a,a\n1,2", "x.csv", "Duplicate"),
    (b"a,b\n1,2,3", "x.csv", "fields"), (b"a,b\n", "x.csv", "no customer"), (b"hello", "x.exe", "Supported"),
    (b"\xff\xfe", "x.csv", "UTF-8"), (b"notzip", "x.xlsx", "valid")])
def test_bad_uploads_have_actionable_errors(content, name, message):
    with pytest.raises(ValueError, match=message):
        read_table(content, name)


def test_csv_quoted_and_whitespace_duplicate_headers_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        read_table(b'" fee ",fee\n1,2\n', "x.csv")


def test_xlsx_selects_explicit_sheet_and_validates_content():
    pytest.importorskip("openpyxl")
    stream = io.BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        pd.DataFrame({"help": ["dictionary"]}).to_excel(writer, sheet_name="Dictionary", index=False)
        fixture_frame().to_excel(writer, sheet_name="Customers", index=False)
    with pytest.raises(ValueError, match="Select a worksheet"):
        read_table(stream.getvalue(), "x.xlsx")
    frame = read_table(stream.getvalue(), "x.xlsx", sheet_name="Customers")
    assert frame.shape == (180, 6)
    assert profile_dataframe(frame)["worksheets"] == ["Dictionary", "Customers"]


@pytest.mark.parametrize("labels,positive", [([0, 1], "1"), ([False, True], "true"), (["active", "cancelled"], "cancelled"), ([1.0, 2.0], "2")])
def test_arbitrary_binary_positive_encodings(labels, positive):
    frame = fixture_frame()
    frame["Exited"] = [labels[index % 2] for index in range(len(frame))]
    config = fixture_config()
    config["positive_label"] = positive
    validated = validate_configuration(frame, config)
    assert target_values(frame, validated).sum() == 90
    assert target_values(frame.iloc[[1]], validated).iloc[0] == 1


@pytest.mark.parametrize("bad", [None, "unsupported", " "])
def test_missing_or_invalid_target_rows_rejected(bad):
    frame, config = fixture_frame(), fixture_config()
    frame.loc[0, "Exited"] = bad
    with pytest.raises(ValueError):
        validate_configuration(frame, config)


def test_single_class_or_three_class_target_cannot_train():
    frame, config = fixture_frame(), fixture_config()
    frame["Exited"] = "Left"
    with pytest.raises(ValueError, match="exactly two"):
        validate_configuration(frame, config)


def test_exploratory_unlabelled_configuration_is_explicit():
    frame = fixture_frame().drop(columns="Exited")
    config = fixture_config()
    config.update(target_column=None, positive_label=None)
    validated = validate_configuration(frame, config)
    assert validated["class_balance"] is None
    assert any("exploratory-only" in decision for decision in validated["decisions"])


@pytest.mark.parametrize("column", ["CustomerID", "Exited"])
def test_identifiers_and_target_never_enter_predictors(column):
    config = fixture_config()
    config["features"].append({"column": column, "kind": "categorical"})
    with pytest.raises(ValueError, match="cannot be a predictor"):
        validate_configuration(fixture_frame(), config)


def test_post_outcome_feature_needs_timing_review_and_exact_proxy_is_blocked():
    frame, config = fixture_frame(), fixture_config()
    frame["Status"] = np.arange(len(frame)) % 3
    config["features"].append({"column": "Status", "kind": "categorical"})
    with pytest.raises(ValueError, match="post-outcome"):
        validate_configuration(frame, config)
    config["leakage_approved"] = {"Status": "Recorded in the observation window, before churn labels."}
    assert validate_configuration(frame, config)["leakage_approved"]["Status"]
    frame["copy"] = frame.Exited.eq("Left").astype(int)
    config["features"].append({"column": "copy", "kind": "numeric"})
    with pytest.raises(ValueError, match="target"):
        validate_configuration(frame, config)


def test_ordinal_order_and_fixed_reference_dates_are_explicit():
    frame, config = fixture_frame(), fixture_config()
    config["excluded_columns"] = []
    config["features"].append({"column": "rating", "kind": "ordinal"})
    with pytest.raises(ValueError, match="ordinal_order"):
        validate_configuration(frame, config)
    config["features"][-1]["ordinal_order"] = ["Low", "Mid", "High"]
    frame["opened"] = pd.date_range("2020-01-01", periods=len(frame)).astype(str)
    config["features"].append({"column": "opened", "kind": "datetime", "date_reference": "2021-01-01"})
    validated = validate_configuration(frame, config)
    prepared, _ = prepare_features(frame, validated)
    assert prepared.opened.iloc[0] == 366
    newer = frame.head(1).copy()
    newer["opened"] = "2022-01-01"
    with pytest.raises(ValueError, match="reference"):
        prepare_features(newer, validated, scoring=True)


def test_missing_values_preserved_and_bad_numeric_does_not_silently_become_missing():
    frame, config = fixture_frame(), fixture_config()
    frame.loc[0, "fee"] = np.nan
    frame.loc[1, "plan"] = np.nan
    validated = validate_configuration(frame, config)
    prepared, _ = prepare_features(frame, validated)
    assert prepared.fee.isna().sum() == 1 and prepared.plan.isna().sum() == 1
    invalid = frame.copy()
    invalid["fee"] = invalid.fee.astype(object)
    invalid.loc[1, "fee"] = "bad"
    with pytest.raises(ValueError, match="nonnumeric"):
        validate_configuration(invalid, config)
    with pytest.raises(ValueError, match="finite numeric"):
        prepare_features(invalid, validated, scoring=True)


def test_constant_high_cardinality_sensitive_and_no_horizon_audits():
    frame, config = fixture_frame(600), fixture_config()
    frame["same"] = "constant"
    config["features"].append({"column": "same", "kind": "categorical"})
    with pytest.raises(ValueError, match="constant"):
        validate_configuration(frame, config)
    config["features"][-1] = {"column": "textkey", "kind": "categorical"}
    frame["textkey"] = [f"unique-{i}" for i in range(len(frame))]
    with pytest.raises(ValueError, match="500 categories"):
        validate_configuration(frame, config)
    config["features"].pop()
    frame["Age"] = np.arange(len(frame)) % 80
    config["semantics"] = [{"column": "Age", "concept": "lifecycle", "meaning": "Customer age", "confirmed": True, "actionable": True}]
    checked = validate_configuration(frame, config)
    assert checked["semantics"][0]["protected"] and not checked["semantics"][0]["actionable"]
    assert any("horizon" in warning for warning in checked["warnings"])


def test_fingerprints_change_with_values_and_schema_not_notes():
    frame, config = fixture_frame(), fixture_config()
    checked = validate_configuration(frame, config)
    initial = configuration_fingerprint(checked)
    checked["warnings"].append("new display note")
    assert configuration_fingerprint(checked) == initial
    checked["positive_label"] = "Stayed"
    assert configuration_fingerprint(checked) != initial
    copy = frame.copy()
    copy.loc[0, "fee"] += 1
    assert frame_fingerprint(copy) != frame_fingerprint(frame)
