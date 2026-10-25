"""Real-data HTTP end-to-end benchmarks. Run the platform API first.

Download the named datasets with download_benchmarks.py. Every model trains on
its own confirmed schema. Unavailable sources and failed checks are reported.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from app.services.data_engine import read_table
from configurations import DATASETS, configuration


def checked(response, status=200):
    if response.status_code != status:
        raise AssertionError(f"HTTP {response.status_code} from {response.request.url.path}: {response.text[:1200]}")
    return response.json()


def save(records):
    path = ROOT / "benchmarks/results.json"
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Actual dataset compatibility benchmarks", "",
             "Each dataset was uploaded through the HTTP API, explicitly configured and trained separately. These are held-out historical results, not cross-provider transfer tests or causal intervention evidence.", "",
             "| Dataset | Rows | Features | Target | Selected model | ROC-AUC | PR-AUC | F1 | Recommendation status | Result |",
             "|---|---:|---:|---|---|---:|---:|---:|---|---|"]
    for record in records:
        metrics = record.get("selected_test_metrics", {})
        values = [record["dataset"], str(record.get("rows", "—")), str(record.get("features", "—")), record.get("target", "—"),
                  record.get("selected_model", "—"), *[f"{metrics[key]:.4f}" if metrics.get(key) is not None else "—" for key in ["roc_auc", "pr_auc", "f1"]],
                  record.get("recommendation_status", "Not executed"), record["status"]]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Configurations and limitations", "",
              "- Full standard tuning (unless an individual row records fast=true), fixed seed 42 and validation PR-AUC selection; equal-weight hybrid is compared without test-set selection.",
              "- Explicit mappings are in configurations.py. Training-only preprocessing and calibration use independent stratified training/validation/test cohorts. Exact duplicate rows are removed and reported by each model's metadata.",
              "- Iranian Status is excluded conservatively after timing review. UCI documents Charge Amount bands 0–9, but the file includes code 10: it is treated as an unordered category and its pricing mapping remains unconfirmed. Age and Age Group are excluded.",
              "- Banking names/identifiers and age, gender, geography and salary are excluded; six business predictors remain. Activity is an engagement indicator, not evidence of dissatisfaction or financial difficulty.",
              "- E-commerce uses the actual E Comm worksheet, not Data Dict. Gender, marital status and city tier are excluded. SatisfactionScore is explicitly ordered 1–5; missing numeric values are imputed on training only.",
              "- The generic IBM benchmark uses a separate 15-feature configuration and separate new artifacts; it does not alter the original 19-feature case study or its saved results.",
              "- Banking, e-commerce and IBM prediction horizons are unknown. Their results are retrospective sample evaluations, not verified prospective forecasts.",
              "- External raw files are excluded from Git. Source links, hashes and license metadata remain in the local download provenance. No unavailable benchmark is replaced by synthetic data.", ""]
    for record in records:
        lines += [f"### {record['dataset']}", "", record.get("error", "All recorded HTTP checks passed."), "",
                  "Checks: " + ", ".join(record.get("checks", [])), ""]
    (ROOT / "benchmarks/results.md").write_text("\n".join(lines), encoding="utf-8")


def benchmark(client, name, spec, fast):
    record = {"dataset": name, "status": "failed", "fast": fast, "checks": []}
    path = spec["path"]
    if not path.is_file():
        record.update(status="unavailable", error="Actual source file is missing. Follow the dataset download guide; no substitute was used.")
        return record
    try:
        frame = read_table(path.read_bytes(), path.name, spec["sheet"])
        config = configuration(name, frame)
        record.update(rows=len(frame), columns=len(frame.columns), features=len(config["features"]), target=config["target_column"], excluded_columns=config["excluded_columns"], prediction_horizon=config["prediction_horizon"])
        form = {"sheet_name": spec["sheet"]} if spec["sheet"] else {}
        dataset = checked(client.post("/datasets/upload", files={"file": (path.name, path.read_bytes())}, data=form), 201)
        dataset_id = dataset["id"]
        record["dataset_id"] = dataset_id
        assert dataset["profile"]["rows"] == len(frame)
        checked(client.post(f"/datasets/{dataset_id}/configuration", json=config))
        record["checks"].extend(["real file upload and schema profile", "explicit target/positive label/features/semantics configuration"])
        started = time.perf_counter()
        job = checked(client.post("/analyses/train", json={"dataset_id": dataset_id, "fast": fast, "random_state": 42}), 202)
        record["request_seconds"] = time.perf_counter() - started
        analysis_id = job["analysis_id"]
        record["analysis_id"] = analysis_id
        deadline = time.monotonic() + 900
        seen_progress = []
        while True:
            status = checked(client.get(f"/jobs/{job['job_id']}"))
            if not seen_progress or seen_progress[-1] != status["progress"]:
                seen_progress.append(status["progress"])
                print(f"{name}: {status['status']} {status['progress']}% {status['message']}", flush=True)
            if status["status"] == "failed":
                raise AssertionError(status["error"])
            if status["status"] == "completed":
                break
            if time.monotonic() > deadline:
                raise AssertionError("Real background training did not finish within 15 minutes.")
            time.sleep(.5)
        record["training_seconds"] = time.perf_counter() - started
        record["progress_observed"] = seen_progress
        base = f"/analyses/{analysis_id}"
        models = checked(client.get(base + "/models"))
        assert len(models["metrics"]) == 4 and len(models["curves"]) == 4
        selected = models["preferred_model"]
        record.update(selected_model=selected, selected_test_metrics=next(row for row in models["metrics"] if row["model"] == selected), all_metrics=models["metrics"], metadata=models["metadata"])
        report_dir = ROOT / "benchmarks" / "reports"
        report_dir.mkdir(exist_ok=True)
        (report_dir / f"{name}_models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
        record["checks"].extend(["true background training and progress", "four held-out models and calibration curves"])
        overview = checked(client.get(base + "/overview"))
        checked(client.get(base + "/eda"))
        explanation = checked(client.get(base + "/explanations", params={"model": selected}))
        assert explanation["importance"]
        customers = checked(client.get(base + "/customers", params={"limit": 7, "model": selected}))
        assert customers["total"] == len(frame) and len(customers["rows"]) == 7
        assert all(0 <= row["churn_probability"] <= 1 for row in customers["rows"])
        row_index = customers["rows"][0]["row_index"]
        detail = checked(client.get(base + f"/customers/{row_index}", params={"model": selected}))
        assert len(detail["explanation"]["drivers"]) == len(config["features"])
        recommendation = checked(client.get(base + "/recommendations", params={"row_index": row_index, "model": selected}))
        qualified = sum(row["evidence_status"] == "observational_signal" for row in recommendation["segments"])
        limited = len(recommendation["segments"]) - qualified
        record["recommendation_status"] = f"{qualified} evidence-qualified; {limited} limited-evidence segments"
        (report_dir / f"{name}_recommendations.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
        record["checks"].extend(["dynamic overview and EDA", "global and individual explanation", "customer probability and historical outcome separation", "evidence-linked recommendations"])
        batch = frame.head(12).drop(columns=[config["target_column"]]).copy()
        batch = __import__('pandas').concat([batch, batch.iloc[[0]]], ignore_index=True)
        scored = checked(client.post(base + "/batch-score", files={"file": ("unlabelled.csv", batch.to_csv(index=False).encode())}, data={"model": selected}))
        assert scored["total"] == 13
        incompatible = batch.drop(columns=[config["features"][0]["column"]])
        assert client.post(base + "/batch-score", files={"file": ("bad.csv", incompatible.to_csv(index=False).encode())}).status_code == 422
        assert client.get(base + "/explanations", params={"model": "Nonexistent model"}).status_code == 422
        record["checks"].extend(["unlabelled scoring with duplicate-row preservation", "incompatible schema and unknown model rejection"])
        inputs = {"budget": 1000, "targeted": 100, "intervention_cost": 15, "incremental_impact": .1, "contribution": 240, "fixed_cost": 100, "period": "12 months", "currency": "USD"}
        economics = checked(client.post(base + "/economics", json=inputs))
        record["economics"] = {"scenario": economics["scenario"], "sensitivity_rows": len(economics["sensitivity"]), "ranked_records": len(economics["ranking"])}
        record["checks"].append("editable hypothetical budget/economics scenario")
        if detail["actionable_features"]:
            feature = detail["actionable_features"][0]
            value = (feature["minimum"] + feature["maximum"]) / 2 if feature["kind"] == "numeric" else feature["categories"][0]
            scenario = checked(client.post(base + "/what-if", json={"row_index": row_index, "changes": {feature["column"]: value}, "model": selected}))
            assert 0 <= scenario["scenario_probability"] <= 1
            record["checks"].append("confirmed actionable what-if sensitivity")
        assert client.post(base + "/what-if", json={"row_index": row_index, "changes": {config["target_column"]: 0}}).status_code == 422
        for kind in ["report", "metrics", "customers", "recommendations"]:
            exported = client.get(base + "/export", params={"kind": kind, "model": selected})
            assert exported.status_code == 200 and exported.content
            if kind == "report":
                assert exported.json()["dataset"]["id"] == dataset_id
            else:
                assert list(csv.reader(io.StringIO(exported.text)))
        assert checked(client.get(base + "/models"))["metrics"] == models["metrics"]
        record["checks"].extend(["immutable/outcome what-if rejection", "valid report and CSV exports", "test metrics unchanged by exploration"])
        record["status"] = "passed"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        print(f"FAILED {name}: {record['error']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/api/v1")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASETS))
    args = parser.parse_args()
    records = []
    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        checked(client.get("/health"))
        for name, spec in DATASETS.items():
            if args.dataset and name != args.dataset:
                continue
            print(f"START {name}", flush=True)
            records.append(benchmark(client, name, spec, args.fast))
            save(records)
    print(json.dumps([{k: r.get(k) for k in ["dataset", "status", "selected_model", "error"]} for r in records]), flush=True)
    if any(record["status"] != "passed" for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
