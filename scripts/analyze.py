"""Export actual data-quality, exploratory and statistical results without training."""
import argparse
import json
import os
from pathlib import Path

from retention.data import load_data
from retention.eda import descriptive_stats, segment_summary, add_tenure_band
from retention.statistics import association_tests, logistic_odds_ratios
from retention.recommendations import segment_recommendations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("TELCO_DATA_PATH", "data/Telco-Customer-Churn.csv"))
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    df, quality = load_data(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    descriptive_stats(df).to_csv(output / "descriptive_statistics.csv", index=False)
    association_tests(df).to_csv(output / "statistical_associations.csv", index=False)
    odds, explanation = logistic_odds_ratios(df)
    odds.to_csv(output / "logistic_odds_ratios.csv", index=False)
    (output / "statistical_method.txt").write_text(explanation + "\n", encoding="utf-8")
    for column in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "TenureBand"]:
        segment_summary(add_tenure_band(df), column).to_csv(output / f"segments_{column}.csv", index=False)
    segment_recommendations(df).to_csv(output / "segment_recommendations.csv", index=False)
    print(f"Exported actual descriptive and statistical results for {len(df):,} records to {output.resolve()}")
    print(explanation)


if __name__ == "__main__":
    main()
