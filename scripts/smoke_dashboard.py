"""Execute all dashboard pages and important controls with real model artifacts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from retention.data import load_data, dataset_fingerprint
from retention.modeling import load_bundle

ROOT = Path(__file__).resolve().parents[1]


def _healthy(app, label):
    errors = [error.message for error in app.exception]
    if errors:
        raise AssertionError(f"{label}: " + " | ".join(errors))
    # Explicit business-logic warnings and ordinary validation messages are valid.


def _number(app, label):
    return next(element for element in app.number_input if element.label == label)


def main():
    df, _ = load_data(ROOT / "data/Telco-Customer-Churn.csv")
    bundle = load_bundle(ROOT / "artifacts/default_bundle.joblib")
    assert bundle["data_hash"] == dataset_fingerprint(df), "Run python -m scripts.train first."
    assert len(bundle["models"]) == 4
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    checks = []
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        _healthy(app, page)
        checks.append({"page": page, "passed": True,
                       "plotly_charts": len(app.get("plotly_chart")), "tables": len(app.dataframe)})
        print(f"PASS {page}", flush=True)
    app.sidebar.radio[0].set_value("Executive Overview").run()
    app.sidebar.multiselect[0].set_value(["Two year"]).run()
    _healthy(app, "business filter")
    expected = int(df["Contract"].eq("Two year").sum())
    observed = next(metric.value for metric in app.metric if metric.label == "Customers")
    assert int(observed.replace(",", "")) == expected
    checks.append({"control": "contract filter changes KPI to actual group count", "passed": True})
    app.sidebar.multiselect[0].set_value([]).run()
    app.sidebar.radio[0].set_value("Economic Impact").run()
    _number(app, "Customers targeted").set_value(0)
    _number(app, "Additional campaign costs").set_value(0.0)
    app.run()
    _healthy(app, "zero-customer economic scenario")
    assert next(metric.value for metric in app.metric if metric.label == "Estimated ROI") == "Undefined"
    checks.append({"control": "zero campaign denominators handled", "passed": True})
    app.sidebar.radio[0].set_value("Model Comparison").run()
    next(element for element in app.selectbox if element.label == "Rank the test-results table by").set_value("recall").run()
    _healthy(app, "comparison ranking")
    checks.append({"control": "model table metric ranking", "passed": True})
    app.sidebar.radio[0].set_value("Customer Risk Explorer").run()
    next(element for element in app.multiselect if element.label == "Risk category").set_value([]).run()
    _healthy(app, "batch scoring with empty historical view")
    assert any("Score a CSV without a churn label" in element.value for element in app.subheader)
    checks.append({"control": "batch scoring remains available with no historical risk rows", "passed": True})
    app.sidebar.radio[0].set_value("Explainability").run()
    next(element for element in app.selectbox if element.label == "Model to explain with SHAP").set_value("XGBoost").run()
    next(element for element in app.button if element.key == "run_shap").click().run()
    _healthy(app, "optional SHAP calculation and fallback")
    checks.append({"control": "SHAP summary calculation or explicit fallback", "passed": True})
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "data_hash": bundle["data_hash"],
              "models": list(bundle["models"]), "all_passed": True, "checks": checks}
    (ROOT / "artifacts/verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS {len(checks)} dashboard checks; saved artifacts/verification.json", flush=True)


if __name__ == "__main__":
    main()
