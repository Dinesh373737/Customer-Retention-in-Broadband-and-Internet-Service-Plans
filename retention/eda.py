"""Computed descriptive summaries for the historical IBM customer sample."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from retention.data import NUMERIC_COLUMNS

TENURE_BANDS = ["0–12 months", "13–24 months", "25–48 months", "49–60 months", "61+ months"]


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """Use (−1,12], (12,24], (24,48], (48,60], (60,∞] whole-month bands."""
    result = df.copy()
    result["TenureBand"] = pd.cut(
        result["tenure"], bins=[-1, 12, 24, 48, 60, np.inf], labels=TENURE_BANDS,
        right=True,
    )
    return result


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    proportion = successes / count
    denominator = 1 + z * z / count
    centre = (proportion + z * z / (2 * count)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def segment_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Observed churn rates and 95% Wilson binomial intervals, including missing groups."""
    if column not in df or "Churn" not in df:
        raise ValueError(f"Segment summaries require {column} and historical Churn labels.")
    data = df[[column, "Churn"]].copy()
    data[column] = data[column].astype(object).where(data[column].notna(), "Missing")
    grouped = data.groupby(column, observed=True, sort=False)["Churn"].agg(customers="count", churned="sum").reset_index()
    grouped = grouped.rename(columns={column: "segment"})
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    intervals = [_wilson_interval(int(row.churned), int(row.customers)) for row in grouped.itertuples()]
    grouped["ci_lower"] = [interval[0] for interval in intervals]
    grouped["ci_upper"] = [interval[1] for interval in intervals]
    if isinstance(df[column].dtype, pd.CategoricalDtype):
        order = {str(value): index for index, value in enumerate(df[column].cat.categories)}
        grouped = grouped.iloc[sorted(range(len(grouped)), key=lambda index: order.get(str(grouped.iloc[index]["segment"]), len(order)))]
    return grouped.reset_index(drop=True)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Counts, spread, quartiles and missing counts; charges remain source currency units."""
    summary = df[NUMERIC_COLUMNS].describe().T.rename_axis("feature").reset_index()
    summary["missing"] = [int(df[column].isna().sum()) for column in summary["feature"]]
    return summary


def executive_insights(df: pd.DataFrame) -> list[str]:
    """Generate plain-language observations from the current (possibly filtered) sample."""
    if df.empty or "Churn" not in df:
        return ["Historical Churn labels are needed to describe observed churn patterns."]
    churn = int(df["Churn"].sum())
    observations = [f"In this selected IBM sample, {churn:,} of {len(df):,} customers ({df['Churn'].mean():.1%}) are labelled as churned."]
    for column, label in [("Contract", "contract"), ("InternetService", "internet-service"), ("TenureBand", "tenure")]:
        source = add_tenure_band(df) if column == "TenureBand" else df
        summary = segment_summary(source, column)
        eligible = summary[summary["customers"] >= 30].sort_values("churn_rate", ascending=False)
        if len(eligible) >= 2:
            highest, lowest = eligible.iloc[0], eligible.iloc[-1]
            observations.append(
                f"Among {label} groups with at least 30 records, {highest['segment']} has the highest observed churn rate "
                f"({highest['churn_rate']:.1%}; n={int(highest['customers']):,}), compared with {lowest['segment']} "
                f"({lowest['churn_rate']:.1%}; n={int(lowest['customers']):,})."
            )
    if df["Churn"].nunique() == 2:
        means = df.groupby("Churn")["MonthlyCharges"].mean()
        observations.append(f"Average monthly charges are {means.loc[1]:.2f} for churned customers and {means.loc[0]:.2f} for non-churned customers, in the dataset's source currency units.")
    observations.append("These are historical associations within the selected sample; they do not establish why an individual left or whether an offer would change their decision.")
    return observations
