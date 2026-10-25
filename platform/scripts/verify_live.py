"""Final live-API regression checks, including real exploratory-only records."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from configurations import configuration, DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    args = parser.parse_args()
    checks = []
    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
        print("PASS", label, flush=True)
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        check("actual API health", client.get("/health").status_code == 200)
        benchmarks = json.loads((ROOT / "benchmarks/results.json").read_text())
        for record in benchmarks:
            check(f"{record['dataset']}: actual benchmark passed", record["status"] == "passed")
            base = "/analyses/" + record["analysis_id"]
            models = client.get(base + "/models").json()
            check(f"{record['dataset']}: persisted model/version identity", models["analysis_id"] == record["analysis_id"] and models["dataset_id"] == record["dataset_id"])
            check(f"{record['dataset']}: evaluation survived server restart", models["metrics"] == record["all_metrics"])
            response = client.get(base + "/recommendations", params={"filters": json.dumps({"not_a_column": ["x"]})})
            check(f"{record['dataset']}: invalid filters are errors", response.status_code == 422)
        ibm = next(record for record in benchmarks if record["dataset"] == "ibm")
        base = "/analyses/" + ibm["analysis_id"]
        explanation = client.get(base + "/explanations", params={"model": "XGBoost", "include_shap": "true"})
        check("actual generic SHAP or explicit supported fallback", explanation.status_code == 200 and (explanation.json()["shap_available"] or "fallback" in explanation.json()["shap_note"].lower() or "unavailable" in explanation.json()["shap_note"].lower()))
        check("SHAP/model association", explanation.json()["model"] == "XGBoost")
        empty = client.get(base + "/customers", params={"search": "reference-never-exists-987654321"}).json()
        check("empty customer search is an honest empty state", empty["total"] == 0 and empty["rows"] == [])
        frame = pd.read_csv(DATASETS["banking"]["path"])
        config = configuration("banking", frame)
        exploratory = frame.head(200).drop(columns=["Exited"])
        upload = client.post("/datasets/upload", files={"file": ("bank-exploratory.csv", exploratory.to_csv(index=False).encode())})
        check("real unlabelled upload", upload.status_code == 201)
        identifier = upload.json()["id"]
        config.update(target_column=None, positive_label=None)
        check("exploratory schema confirmation", client.post(f"/datasets/{identifier}/configuration", json=config).status_code == 200)
        created = client.post("/analyses", json={"dataset_id": identifier})
        check("Mode C creation", created.status_code == 201 and created.json()["operating_mode"] == "C")
        exploratory_id = created.json()["analysis_id"]
        exp_base = "/analyses/" + exploratory_id
        overview = client.get(exp_base + "/overview").json()
        check("Mode C has descriptions without invented outcomes", overview["label_available"] is False and overview["churn_rate"] is None)
        customers = client.get(exp_base + "/customers").json()
        check("Mode C has no invented probabilities", all(row["churn_probability"] is None and row["risk_category"] == "Unscored" for row in customers["rows"]))
        check("Mode C model request explicitly unavailable", client.get(exp_base + "/models").status_code == 409)
        check("Mode C cannot train without labels", client.post("/analyses/train", json={"dataset_id": identifier}).status_code == 409)
        report = client.get(exp_base + "/export", params={"kind": "report"}).json()
        check("Mode C report contains no fabricated model metrics", report["models"] is None)
        recommendations = client.get(exp_base + "/recommendations").json()
        check("Mode C recommendation evidence unavailable", recommendations["evidence_status"] == "insufficient" and recommendations["segments"] == [])
        small = frame.groupby("Exited", group_keys=False).head(4)[["CreditScore", "Exited"]]
        uploaded = client.post("/datasets/upload", files={"file": ("too-small.csv", small.to_csv(index=False).encode())}).json()
        small_config = {"target_column": "Exited", "positive_label": "1", "features": [{"column": "CreditScore", "kind": "numeric"}]}
        check("insufficient-data schema still supports descriptive analysis", client.post(f"/datasets/{uploaded['id']}/configuration", json=small_config).status_code == 200)
        job = client.post("/analyses/train", json={"dataset_id": uploaded["id"], "fast": True})
        check("insufficient-data training is a real asynchronous job", job.status_code == 202)
        deadline = time.monotonic() + 30
        while True:
            status = client.get("/jobs/" + job.json()["job_id"]).json()
            if status["status"] in {"completed", "failed"}:
                break
            if time.monotonic() > deadline:
                raise AssertionError("Expected failed training job did not finish")
            time.sleep(.1)
        check("training failures retain genuine diagnostics", status["status"] == "failed" and "at least 60" in status["error"])
        check("failed analysis never returns invented metrics", client.get("/analyses/" + job.json()["analysis_id"] + "/models").status_code == 409)
        check("empty files rejected", client.post("/datasets/upload", files={"file": ("empty.csv", b"")}).status_code == 422)
        check("unsupported uploads rejected", client.post("/datasets/upload", files={"file": ("malicious.py", b"print('never executed')")}).status_code == 422)
        check("unknown dataset IDs rejected", client.get("/datasets/not-an-id/profile").status_code == 404)
    output = {"all_passed": True, "checks": checks, "check_count": len(checks), "exploratory_analysis_id": exploratory_id}
    (ROOT / "benchmarks/live_verification.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"PASS {len(checks)} live API checks", flush=True)


if __name__ == "__main__":
    main()
