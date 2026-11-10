"""Automatic schema suggestions so a dataset can be analysed without manual column mapping.

Every suggestion is passed through ``validate_configuration``: columns the validator rejects
(target proxies, possible post-outcome fields, constant or free-text columns) are excluded and
the reason is reported back, so the automatic path never bypasses the platform's safeguards.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .data_engine import (TARGET_NAMES, RESERVED_OUTPUT_COLUMNS, canonical_label, is_identifier,
                          is_protected, is_sensitive, leakage_warning, validate_configuration)

POSITIVE_LABELS = ["yes", "true", "1", "churn", "churned", "left", "exited", "cancelled", "canceled",
                   "attrited", "attritedcustomer", "inactive", "lost", "closed", "y"]
TARGET_HINTS = ["churn", "exited", "attrition", "attrited", "cancel", "left", "lapsed", "defect"]

DOMAIN_HINTS = {
    "telecommunications": ["tenure", "contract", "internet", "phone", "streaming", "multiplelines", "techsupport",
                            "fiber", "dsl", "minutes", "calls", "roaming", "dataplan", "voicemail", "broadband", "sim"],
    "banking": ["balance", "creditscore", "numofproducts", "hascrcard", "creditcard", "loan", "salary", "deposit",
                "transaction", "overdraft", "card", "bank", "cardcategory"],
    "ecommerce": ["order", "cashback", "coupon", "cart", "purchase", "delivery", "warehouse", "prefered", "preferred",
                  "category", "hourspendonapp", "daysincelastorder", "orderamount"],
    "subscription": ["subscription", "renewal", "membership", "watch", "viewing", "premium", "trial", "plan",
                     "autorenew", "seats", "license"],
}

# Ordered: the first matching concept wins.
CONCEPT_HINTS = [
    ("support", ["support", "complain", "complaint", "ticket", "servicecall", "customerservice", "helpdesk", "callcenter"]),
    ("satisfaction", ["satisfaction", "csat", "nps", "rating", "review", "feedback", "sentiment"]),
    ("contract", ["contract", "term", "commitment", "renewal", "autorenew", "plantype", "subscriptiontype", "billingcycle"]),
    ("pricing", ["charge", "price", "fee", "bill", "cost", "payment", "spend", "amount", "discount", "cashback",
                 "coupon", "revenue", "arpu", "paperless"]),
    ("loyalty", ["loyalty", "points", "reward", "activemember", "membership", "tier"]),
    ("lifecycle", ["tenure", "monthswith", "dayssince", "since", "lastorder", "lastlogin", "accountage",
                   "customerage", "yearswith"]),
    ("engagement", ["login", "usage", "usage", "active", "streaming", "hours", "visit", "session", "orders",
                    "ordercount", "numofproducts", "products", "app", "device", "online", "backup", "security",
                    "minutes", "calls", "data", "download", "engage", "frequency", "count"]),
]
ACTIONABLE_CONCEPTS = {"pricing", "contract", "support", "engagement", "satisfaction", "loyalty"}
CONCEPT_MEANING = {
    "pricing": "price, charges or billing",
    "contract": "contract or plan terms",
    "support": "service and support contact",
    "satisfaction": "customer satisfaction feedback",
    "loyalty": "loyalty or membership status",
    "lifecycle": "customer lifecycle or account age",
    "engagement": "product usage and engagement",
}


def _key(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _quoted_column(message: str, names: set[str]) -> str | None:
    for match in re.findall(r"'([^']+)'", message):
        if match in names:
            return match
    return None


def suggest_target(df: pd.DataFrame, profile: dict) -> str | None:
    """Pick the most plausible binary churn outcome column."""
    def binary(name):
        return df[name].dropna().nunique() == 2

    for name in profile.get("target_candidates", []):
        if name in df and binary(name):
            return name
    targets = {_key(item) for item in TARGET_NAMES}
    scored = []
    for name in df.columns:
        key = _key(name)
        if not binary(name) or is_identifier(name):
            continue
        if key in targets:
            scored.append((0, name))
        elif any(hint in key for hint in TARGET_HINTS):
            scored.append((1, name))
    return sorted(scored)[0][1] if scored else None


def suggest_positive_label(df: pd.DataFrame, target: str) -> str:
    labels = df[target].dropna().map(canonical_label)
    unique = sorted(labels.unique().tolist())
    for preferred in POSITIVE_LABELS:
        for label in unique:
            if _key(label) == preferred:
                return label
    # Fall back to the minority class: churners are almost always the smaller group.
    return labels.value_counts().sort_values(kind="stable").index[0]


def suggest_identifier(df: pd.DataFrame, target: str | None) -> str | None:
    named = [name for name in df.columns if name != target and is_identifier(name)]
    if named:
        # Prefer a business key (customer/account ID) over a positional row number.
        people = ["customer", "client", "account", "subscriber", "member", "user"]
        return sorted(named, key=lambda name: (not any(word in _key(name) for word in people), list(df.columns).index(name)))[0]
    for name in df.columns:
        key = _key(name)
        present = df[name].dropna()
        if name == target or not len(present):
            continue
        if (key.endswith("id") or key.endswith("number") or key.endswith("ref")) and present.nunique() == len(df) and not pd.api.types.is_float_dtype(present):
            return name
    return None


def suggest_domain(columns) -> str:
    keys = [_key(name) for name in columns]
    scores = {domain: sum(any(hint in key for hint in hints) for key in keys) for domain, hints in DOMAIN_HINTS.items()}
    domain, score = max(scores.items(), key=lambda item: item[1])
    return domain if score >= 2 else "generic"


def suggest_concept(name: str) -> str | None:
    key = _key(name)
    if "creditscore" in key:
        return None
    for concept, hints in CONCEPT_HINTS:
        if any(hint in key for hint in hints):
            return concept
    return None


def _feature_kind(df: pd.DataFrame, column: dict):
    name, inferred = column["name"], column.get("inferred_type")
    present = df[name].dropna()
    if inferred == "datetime":
        dates = pd.to_datetime(present, errors="coerce", utc=True)
        if dates.isna().any() or not len(dates):
            return None, "dates could not be parsed consistently"
        return {"column": name, "kind": "datetime", "date_reference": dates.max().date().isoformat()}, None
    numeric = pd.to_numeric(present, errors="coerce")
    if inferred in {"numeric", "binary"} and numeric.notna().all() and np.isfinite(numeric).all():
        return {"column": name, "kind": "numeric" if inferred == "numeric" else "categorical"}, None
    if present.astype(str).str.len().gt(200).any():
        return None, "contains long free text"
    if present.nunique() > 500:
        return None, "too many distinct values to be a useful category"
    return {"column": name, "kind": "categorical"}, None


def suggest_configuration(df: pd.DataFrame, profile: dict) -> dict:
    """Return ``{"configuration": validated config, "notes": [...], "excluded": [...]}``."""
    target = suggest_target(df, profile)
    positive = suggest_positive_label(df, target) if target else None
    identifier = suggest_identifier(df, target)
    domain = suggest_domain(df.columns)
    notes, excluded = [], []

    def exclude(name, reason):
        excluded.append({"column": name, "reason": reason})

    features = []
    for column in profile["columns"]:
        name = column["name"]
        if name in {target, identifier}:
            continue
        if is_identifier(name):
            exclude(name, "looks like an identifier")
            continue
        if is_sensitive(name) and not is_protected(name):
            exclude(name, "contains personal identifying information")
            continue
        if leakage_warning(name):
            exclude(name, "may only be known after the customer churned")
            continue
        if column.get("unique", 0) <= 1:
            exclude(name, "has a single value")
            continue
        if name in RESERVED_OUTPUT_COLUMNS or name.startswith("__"):
            exclude(name, "uses a reserved platform column name")
            continue
        feature, reason = _feature_kind(df, column)
        if feature is None:
            exclude(name, reason)
            continue
        features.append(feature)

    semantics = []
    for feature in features:
        concept = suggest_concept(feature["column"])
        if not concept:
            continue
        protected = is_protected(feature["column"])
        semantics.append({"column": feature["column"], "concept": concept, "confirmed": True,
                          "meaning": f"Auto-mapped from the column name: '{feature['column']}' appears to describe {CONCEPT_MEANING[concept]}.",
                          "actionable": concept in ACTIONABLE_CONCEPTS and not protected and "total" not in _key(feature["column"]),
                          "protected": protected})

    config = {"target_column": target, "positive_label": positive, "customer_id": identifier,
              "features": features, "business_domain": domain, "prediction_horizon": None,
              "semantics": semantics, "leakage_approved": {}, "excluded_columns": []}

    # Let the validator have the final word; drop any column it rejects and retry.
    names = {item["column"] for item in features}
    for _ in range(len(features) + 1):
        config["excluded_columns"] = [c for c in df.columns if c not in {target, identifier} and c not in names]
        try:
            validated = validate_configuration(df, config)
            break
        except ValueError as error:
            column = _quoted_column(str(error), names)
            if column is None:
                raise
            exclude(column, str(error).replace(f"'{column}' ", "").rstrip(".").lower())
            names.discard(column)
            config["features"] = [item for item in config["features"] if item["column"] != column]
            config["semantics"] = [item for item in config["semantics"] if item["column"] != column]
    else:  # pragma: no cover - loop always breaks or raises
        raise ValueError("Automatic configuration could not find a valid predictor set.")

    if target:
        notes.append(f"Outcome '{target}' detected; '{positive}' is treated as churn.")
    else:
        notes.append("No churn outcome column was detected, so the dataset opens in exploratory mode.")
    if identifier:
        notes.append(f"'{identifier}' is used as the customer reference and kept out of the model.")
    notes.append(f"{len(validated['features'])} predictors selected automatically; {len(excluded)} columns left out.")
    if semantics:
        notes.append(f"{len(validated['semantics'])} columns mapped to business concepts from their names. Review them if a name is misleading.")
    validated["auto_configured"] = True
    return {"configuration": validated, "notes": notes, "excluded": excluded}


def suggest_economics(frame: pd.DataFrame, configuration: dict, scored: pd.DataFrame | None) -> dict:
    """Derive campaign assumptions from the data where possible; fall back to neutral defaults."""
    values = {"budget": None, "targeted": 100, "intervention_cost": 15.0, "incremental_impact": .10,
              "contribution": 240.0, "fixed_cost": 0.0, "period": "12 months", "currency": "USD"}
    basis = []
    if scored is not None and "risk_category" in scored:
        high = int((scored["risk_category"] == "High").sum())
        if high:
            values["targeted"] = high
            basis.append(f"Customers targeted: all {high:,} customers in the High risk band.")
    elif configuration.get("target_column"):
        rate = float(configuration.get("class_balance", {}).get("positive_rate") or 0)
        if rate:
            values["targeted"] = max(1, int(round(len(frame) * rate)))
            basis.append(f"Customers targeted: dataset size × historical churn rate ({rate:.1%}).")

    features = [item["column"] for item in configuration.get("features", []) if item.get("kind") == "numeric"]
    pricing = [item["column"] for item in configuration.get("semantics", []) if item.get("concept") == "pricing" and item["column"] in features]
    candidates = pricing + [name for name in features if suggest_concept(name) == "pricing" and name not in pricing]
    monthly = [name for name in candidates if "month" in _key(name)]
    others = [name for name in candidates if "total" not in _key(name) and name not in monthly]
    column = (monthly or others or [None])[0]
    if column:
        charge = float(pd.to_numeric(frame[column], errors="coerce").median())
        if np.isfinite(charge) and charge > 0:
            values["contribution"] = round(charge * 12 * .35, 2)
            values["intervention_cost"] = round(charge * .5, 2)
            basis.append(f"Contribution per retained customer: 12 × median '{column}' ({charge:,.2f}) × 35% assumed margin.")
            basis.append(f"Cost per intervention: half of one median '{column}' payment, e.g. a one-off credit.")
    if not basis:
        basis.append("No pricing column was found, so neutral default assumptions are used.")
    basis.append("Incremental impact of 10% is a planning assumption, not an observed effect.")
    return {"values": values, "basis": basis}
