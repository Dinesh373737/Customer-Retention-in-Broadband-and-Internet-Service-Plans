"""Train actual models and export reproducible historical results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from retention.data import load_data
from retention.eda import executive_insights
from retention.modeling import train_models, save_bundle
from retention.recommendations import segment_recommendations


def render_report(df, bundle, source):
    lines = ["# Generated historical analysis", "", f"Dataset: {source}. Records: {len(df):,}.",
             f"Generated: {bundle['generated_at']}. Fingerprint: {bundle['data_hash']}.", "",
             "## Observed findings", ""] + [f"- {line}" for line in executive_insights(df)]
    lines += ["", "## Held-out model comparison", "", "```text", bundle["metrics"].to_string(index=False), "```", "",
              f"Preferred model selected on validation PR-AUC: **{bundle['preferred_model']}**.",
              "Test results are final descriptive comparisons, not a basis for changing the fitted model or ensemble weights.",
              "", "See training_metadata.json for actual splits, tuning, calibration and ensemble policy.",
              "No retention interventions or financial outcomes were observed. Predictions are associations in historical sample data."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("TELCO_DATA_PATH", "data/Telco-Customer-Churn.csv"))
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--fast", action="store_true", help="Small model grid for verification; still uses real data")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    df, quality = load_data(args.data)
    print(f"Validated {len(df):,} historical records. Training models...", flush=True)
    bundle = train_models(df, fast=args.fast)
    save_bundle(bundle, output / "default_bundle.joblib")
    bundle["metrics"].to_csv(output / "model_comparison.csv", index=False)
    bundle["validation_metrics"].to_csv(output / "validation_metrics.csv", index=False)
    with (output / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, indent=2, default=str)
    test_records = pd.DataFrame({"row_index": bundle["test_indices"], "actual_churn": bundle["y_test"]})
    for name, scores in bundle["probabilities"].items():
        test_records[name] = scores
    test_records.to_csv(output / "held_out_predictions.csv", index=False)
    segment_recommendations(df).to_csv(output / "segment_recommendations.csv", index=False)
    metadata = {key: bundle[key] for key in ["data_hash", "generated_at", "preferred_model", "metadata"]}
    with (output / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
    (output / "results.md").write_text(render_report(df, bundle, args.data), encoding="utf-8")
    print(bundle["metrics"].to_string(index=False), flush=True)
    print(f"Saved models, metadata and results to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
