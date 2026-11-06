"""Automatic schema inference and the one-call quickstart workflow."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.autoconfig import suggest_configuration, suggest_positive_label
from app.services.data_engine import profile_dataframe


def bank_frame(rows=240):
    rng = np.random.default_rng(7)
    active = rng.integers(0, 2, rows)
    exited = (rng.random(rows) < np.where(active == 1, .15, .45)).astype(int)
    return pd.DataFrame({"RowNumber": range(rows), "CustomerId": rng.integers(10**7, 2 * 10**7, rows),
                         "Surname": rng.choice(["Li", "Smith", "Khan"], rows), "Age": rng.integers(18, 80, rows),
                         "MonthlyFee": rng.uniform(10, 90, rows), "IsActiveMember": active,
                         "Complain": rng.integers(0, 2, rows), "AccountStatus": np.where(exited == 1, "closed", "open"),
                         "Exited": exited})


def test_suggestion_detects_target_label_identifier_and_excludes_unsafe_columns():
    frame = bank_frame()
    result = suggest_configuration(frame, profile_dataframe(frame))
    config = result["configuration"]
    assert config["target_column"] == "Exited"
    assert config["positive_label"] == "1"
    assert config["customer_id"] == "CustomerId"
    features = {item["column"] for item in config["features"]}
    assert {"Age", "MonthlyFee", "IsActiveMember", "Complain"} <= features
    assert not features & {"RowNumber", "CustomerId", "Surname", "AccountStatus", "Exited"}
    excluded = {item["column"] for item in result["excluded"]}
    assert {"Surname", "AccountStatus"} <= excluded
    semantics = {item["column"]: item for item in config["semantics"]}
    assert semantics["MonthlyFee"]["concept"] == "pricing" and semantics["MonthlyFee"]["actionable"]
    assert "Age" not in semantics or not semantics["Age"]["actionable"]


def test_positive_label_prefers_churn_words_then_minority_class():
    assert suggest_positive_label(pd.DataFrame({"y": ["Yes", "No", "No"]}), "y") == "Yes"
    assert suggest_positive_label(pd.DataFrame({"y": ["stay", "go", "stay", "stay"]}), "y") == "go"


def test_unlabelled_data_falls_back_to_exploratory_configuration():
    frame = bank_frame().drop(columns=["Exited", "AccountStatus"])
    config = suggest_configuration(frame, profile_dataframe(frame))["configuration"]
    assert config["target_column"] is None and config["positive_label"] is None
    assert config["features"]


def test_quickstart_trains_and_derives_economic_defaults(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        frame = bank_frame()
        dataset = client.post("/api/v1/datasets/upload", files={"file": ("bank.csv", frame.to_csv(index=False).encode(), "text/csv")}).json()
        started = client.post(f"/api/v1/datasets/{dataset['id']}/quickstart", json={})
        assert started.status_code == 202, started.text
        body = started.json()
        assert body["job_id"] and body["configuration"]["target_column"] == "Exited"
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{body['job_id']}").json()
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(.1)
        assert job["status"] == "completed", job
        defaults = client.get(f"/api/v1/analyses/{body['analysis_id']}/economics/defaults").json()
        assert defaults["values"]["targeted"] >= 1
        assert any("MonthlyFee" in line for line in defaults["basis"])
        saved = client.get(f"/api/v1/datasets/{dataset['id']}/profile").json()
        assert saved["configuration"]["auto_configured"] is True


def test_quickstart_without_target_opens_descriptive_analysis(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        frame = bank_frame().drop(columns=["Exited", "AccountStatus"])
        dataset = client.post("/api/v1/datasets/upload", files={"file": ("u.csv", frame.to_csv(index=False).encode(), "text/csv")}).json()
        body = client.post(f"/api/v1/datasets/{dataset['id']}/quickstart").json()
        assert body["job_id"] is None and body["mode"] == "exploratory"
        assert client.get(f"/api/v1/analyses/{body['analysis_id']}/overview").status_code == 200


def test_bundled_sample_dataset_loads_once(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        first = client.post("/api/v1/datasets/sample")
        assert first.status_code == 201, first.text
        second = client.post("/api/v1/datasets/sample").json()
        assert first.json()["id"] == second["id"]
        assert first.json()["profile"]["rows"] == 7043
