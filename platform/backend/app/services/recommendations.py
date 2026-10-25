"""Deterministic suggestions grounded in confirmed semantics and computed evidence."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .data_engine import target_values, is_sensitive, is_identifier


LIMITATIONS = [
    "Historical associations and model sensitivities are not verified causes or intervention effects.",
    "No customer contact, discount, complaint, preference or retention outcome is invented.",
    "Incremental intervention effectiveness requires suitable experimental or causal evidence.",
    "Protected or immutable personal characteristics cannot be the basis of a retention action.",
]


def catalogue():
    return json.loads(Path(__file__).with_name("recommendations_catalog.json").read_text(encoding="utf-8"))


def protected_feature(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", column.lower())
    protected = {"age", "agegroup", "gender", "sex", "race", "ethnicity", "religion", "disability", "nationality", "geography", "country", "city", "citytier", "postcode", "zipcode", "maritalstatus", "income", "estimatedsalary", "surname", "firstname", "lastname", "name", "email", "phone", "ssn", "seniorcitizen"}
    return (is_sensitive(column) or is_identifier(column) or normalized in protected
            or normalized.endswith(("emailaddress", "phonenumber")))


def _wilson(successes: int, count: int):
    if not count:
        return (None, None)
    z = 1.959963984540054
    p = successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return center - margin, center + margin


def _groups(series: pd.Series, kind: str):
    if kind in {"numeric", "datetime"} and pd.to_numeric(series, errors="coerce").nunique() > 6:
        values = pd.to_numeric(series, errors="coerce")
        q1, q3 = values.quantile([.25, .75])
        if q1 < q3:
            return [(f"{series.name} ≤ {q1:.4g}", values.le(q1)),
                    (f"{series.name} > {q3:.4g}", values.gt(q3))]
    counts = series.astype("string").fillna("(missing)").value_counts()
    return [(f"{series.name} = {value}", series.astype("string").fillna("(missing)").eq(value)) for value in counts.head(12).index]


def _adjust(values):
    if not values:
        return []
    arr = np.asarray(values)
    order = np.argsort(arr)
    ordered = arr[order] * len(arr) / np.arange(1, len(arr) + 1)
    result = np.empty(len(arr))
    result[order] = np.minimum(1, np.minimum.accumulate(ordered[::-1])[::-1])
    return result.tolist()


def generate_recommendations(df: pd.DataFrame, config: dict, scored: pd.DataFrame | None = None,
                             row_index: int | None = None, drivers: list | None = None,
                             global_drivers: dict | None = None):
    book = catalogue()
    result = {"customers": [], "segments": [], "catalog": book, "limitations": LIMITATIONS,
              "evidence_status": "insufficient", "mapping_warnings": []}
    if not config.get("target_column"):
        result["limitations"] = ["No historical churn labels are available; segment intervention evidence cannot be estimated.", *LIMITATIONS]
        return result
    y = target_values(df, config)
    baseline = float(y.mean())
    mappings = []
    available_features = {item["column"]: item["kind"] for item in config.get("features", [])}
    for semantic in config.get("semantics", []):
        column = semantic.get("column")
        if column not in available_features or column not in df:
            continue
        if not semantic.get("confirmed") or not str(semantic.get("meaning", "")).strip():
            result["mapping_warnings"].append(f"{column}: confirm the business meaning before generating an action.")
            continue
        if semantic.get("protected") or protected_feature(column):
            result["mapping_warnings"].append(f"{column}: protected/personal attribute is excluded from recommendation triggers.")
            continue
        concept = semantic.get("concept")
        rule = book["concepts"].get(concept)
        if not rule or config.get("business_domain", "generic") not in rule["industries"]:
            continue
        mappings.append((semantic, rule))
    candidates = []
    for semantic, rule in mappings:
        column = semantic["column"]
        for segment, mask in _groups(df[column], available_features[column]):
            mask = mask.fillna(False)
            n = int(mask.sum())
            rest = int((~mask).sum())
            if not n or not rest:
                continue
            positives = int(y.loc[mask].sum())
            rest_positive = int(y.loc[~mask].sum())
            p_value = float(fisher_exact([[positives, n - positives], [rest_positive, rest - rest_positive]]).pvalue)
            lower, upper = _wilson(positives, n)
            candidates.append({"semantic": semantic, "rule": rule, "mask": mask, "segment": segment,
                               "sample_size": n, "comparison_size": rest, "observed_churn_rate": positives / n,
                               "ci_lower": lower, "ci_upper": upper, "p_value": p_value})
    for item, adjusted in zip(candidates, _adjust([item["p_value"] for item in candidates])):
        item["p_adjusted"] = adjusted
        trigger = item["rule"].get("trigger_conditions", {})
        strong = (item["sample_size"] >= trigger.get("minimum_segment_size", book["minimum_segment_size"])
                  and item["comparison_size"] >= trigger.get("minimum_comparison_size", book["minimum_segment_size"])
                  and item["observed_churn_rate"] - baseline >= trigger.get("minimum_rate_difference", book["minimum_rate_difference"])
                  and (not trigger.get("wilson_lower_must_exceed_baseline", True) or item["ci_lower"] > baseline)
                  and adjusted <= trigger.get("maximum_adjusted_p", book["maximum_adjusted_p"]))
        if item["observed_churn_rate"] <= baseline:
            continue
        semantic, rule = item["semantic"], item["rule"]
        domain = config.get("business_domain", "generic")
        action = book.get("industry_actions", {}).get(domain, {}).get(semantic["concept"], rule["action"])
        if not strong:
            action = "Gather context before proposing a specific intervention"
        rate = item["observed_churn_rate"]
        evidence = f"Observed churn {rate:.1%} in n={item['sample_size']:,}; overall {baseline:.1%}. 95% Wilson interval {item['ci_lower']:.1%}–{item['ci_upper']:.1%}; BH-adjusted Fisher p={adjusted:.3g}."
        record = {"customer_reference": None, "churn_probability": None, "feature": semantic["column"],
                  "business_concept": semantic["concept"], "business_meaning": semantic["meaning"],
                  "segment": item["segment"], "action": action,
                  "rationale": "Confirmed business meaning and an elevated historical segment association support considering a review." if strong else "This observed segment estimate does not satisfy the catalogue's sample-size and uncertainty requirements.",
                  "evidence": evidence, "sample_size": item["sample_size"], "comparison_size": item["comparison_size"],
                  "observed_churn_rate": rate, "overall_churn_rate": baseline, "ci_lower": item["ci_lower"], "ci_upper": item["ci_upper"],
                  "p_value": item["p_value"], "p_adjusted": adjusted,
                  "objective": rule["objective"] if strong else "Understand the account and gather evidence before acting",
                  "caveats": [rule["caveat"], "Observational association only; no verified causal intervention effect."],
                  "estimated_cost": semantic.get("estimated_cost", rule.get("possible_costs", {}).get("default_per_customer")),
                  "actionability_constraints": rule.get("actionability_constraints", []),
                  "evidence_requirements": rule.get("evidence_requirements", []),
                  "evidence_status": "observational_signal" if strong else "insufficient",
                  "risk_drivers": [{"feature": semantic["column"], "source": "historical segment association"}]}
        global_evidence = (global_drivers or {}).get(semantic["column"])
        if global_evidence is not None:
            if not isinstance(global_evidence, dict):
                global_evidence = {"importance": float(global_evidence), "method": "Selected-model predictive importance; not causal evidence"}
            record["global_model_evidence"] = global_evidence
            record["risk_drivers"].append({"feature": semantic["column"], "source": "selected-model predictive importance", **global_evidence})
        if scored is not None and "churn_probability" in scored:
            probabilities = pd.to_numeric(scored.loc[item["mask"].to_numpy(), "churn_probability"], errors="coerce").dropna()
            record["segment_mean_probability"] = float(probabilities.mean()) if len(probabilities) else None
            record["churn_probability"] = record["segment_mean_probability"]
        result["segments"].append(record)
        if scored is not None:
            eligible = scored.loc[item["mask"].to_numpy()].copy()
            if row_index is not None:
                eligible = eligible.loc[eligible["row_index"] == row_index]
            if "churn_probability" in eligible:
                eligible = eligible.loc[pd.to_numeric(eligible["churn_probability"], errors="coerce") >= trigger.get("customer_risk_threshold", .6)]
            for _, customer in eligible.head(30).iterrows():
                recommendation = dict(record)
                recommendation.update(customer_reference=str(customer.get("customer_reference", customer.get("row_index"))),
                                      row_index=int(customer["row_index"]), churn_probability=float(customer["churn_probability"]))
                if drivers:
                    recommendation["risk_drivers"] = [*drivers, *record["risk_drivers"]]
                result["customers"].append(recommendation)
    result["segments"].sort(key=lambda item: (item["evidence_status"] == "observational_signal", item["observed_churn_rate"]), reverse=True)
    result["customers"] = sorted(result["customers"], key=lambda item: item["churn_probability"], reverse=True)[:200]
    if any(item["evidence_status"] == "observational_signal" for item in result["segments"]):
        result["evidence_status"] = "observational_signal"
    if not result["segments"]:
        result["limitations"] = ["No confirmed, non-protected business mapping has an elevated supported segment. Investigate or gather voluntary feedback before selecting an intervention.", *LIMITATIONS]
    return result
