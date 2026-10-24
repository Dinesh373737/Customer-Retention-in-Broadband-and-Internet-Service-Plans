"""Transparent, evidence-linked suggestions; no automatic customer contact."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_category(probabilities, low: float = .3, high: float = .6):
    """Illustrative policy boundaries, not statistically established cutoffs."""
    if not 0 <= low < high <= 1:
        raise ValueError("Risk thresholds must satisfy 0 <= low < high <= 1.")
    values = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Churn probabilities must be finite values from 0 to 1.")
    result = np.where(values < low, "Low", np.where(values < high, "Medium", "High"))
    if values.ndim == 0:
        return str(result.item())
    if isinstance(probabilities, pd.Series):
        return pd.Series(result, index=probabilities.index, name="Risk category")
    return result


def _candidates(df: pd.DataFrame):
    return [
        ("Month-to-month contract", df["Contract"].eq("Month-to-month"), "Contract", "Month-to-month",
         "Review whether a suitable longer-term contract is appropriate",
         "Assess contract suitability without assuming customer preferences", "Offer a choice only after checking suitability and consent."),
        ("First 12 months", df["tenure"].le(12), "tenure", 12,
         "Offer an optional onboarding or account review",
         "Help new subscribers understand their subscribed services", "Short tenure does not establish an onboarding problem."),
        ("No technical support subscription", df["TechSupport"].eq("No"), "TechSupport", "No",
         "Review whether technical-support information would be useful",
         "Invite a conversation about available support options", "No support subscription is not evidence of a technical problem."),
        ("No online security subscription", df["OnlineSecurity"].eq("No"), "OnlineSecurity", "No",
         "Discuss available security services if relevant to the customer",
         "Check service suitability through further engagement", "A missing subscription does not imply a security issue or unmet need."),
        ("Monthly charges in upper quartile", df["MonthlyCharges"].ge(df["MonthlyCharges"].quantile(.75)),
         "MonthlyCharges", float(df["MonthlyCharges"].quantile(.75)),
         "Review pricing and service-plan suitability",
         "Assess affordability and value without automatically discounting", "Charges do not measure satisfaction; contribution margins are unavailable."),
    ]


def _evidence(df, mask):
    group = df.loc[mask, "Churn"].dropna()
    baseline = df["Churn"].mean()
    count = len(group)
    rate = group.mean() if count else float("nan")
    return count, rate, baseline, (
        f"Historical loaded-data segment: {count:,} records, {rate:.1%} churn; "
        f"all loaded records: {baseline:.1%}. Observational association, not causal evidence."
    )


def customer_recommendations(row, df: pd.DataFrame, probability: float,
                             low: float = .3, high: float = .6) -> list[dict]:
    risk = risk_category(probability, low, high)
    if "Churn" not in df:
        raise ValueError("Labelled historical reference data is required for recommendations.")
    suggestions = []
    if risk == "High":
        for segment, mask, feature, value, action, objective, caveat in _candidates(df):
            raw = row.get(feature)
            if pd.isna(raw):
                continue
            match = raw <= value if feature == "tenure" else (
                raw >= value if feature == "MonthlyCharges" else raw == value)
            n, rate, baseline, evidence = _evidence(df, mask)
            if match and n and rate > baseline:
                suggestions.append({"action": action,
                    "rationale": f"Customer belongs to {segment.lower()}, a group with elevated observed churn in this dataset.",
                    "evidence": evidence, "objective": objective,
                    "caveat": caveat + (" Small segment (fewer than 30 records); estimates are unstable." if n < 30 else "")
                    + " Intervention effectiveness is unmeasured; validate in a controlled pilot."})
    if not suggestions:
        suggestions.append({"action": "Review the account before deciding whether engagement is appropriate",
            "rationale": f"The model assigns {risk.lower()} risk; no specific intervention is established by these data.",
            "evidence": f"Predicted churn probability {probability:.1%}; policy boundaries {low:.0%}/{high:.0%}.",
            "objective": "Gather context before allocating retention resources",
            "caveat": "A prediction is uncertain. No customer complaint, preference, contact or response is observed."})
    return suggestions


def segment_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    items = []
    for segment, mask, _, _, action, objective, caveat in _candidates(df):
        n, rate, baseline, evidence = _evidence(df, mask)
        if not n:
            continue
        elevated = rate > baseline
        items.append({"Customer segment": segment,
            "Observed pattern": f"{rate:.1%} churn (n={n:,}); overall {baseline:.1%}",
            "Possible action": action if elevated else "Monitor; this segment does not show elevated churn",
            "Expected objective": objective if elevated else "Avoid unsupported intervention targeting",
            "Evidence and limitations": evidence + " " + caveat +
             (" Fewer than 30 records; interpret cautiously." if n < 30 else "")})
    return pd.DataFrame(items)
