"""Computed descriptive and observational analytics for confirmed schemas."""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

from .data_engine import json_safe, target_values, prepare_features


def apply_filters(df, filters=None):
    frame = df
    for name, rule in (filters or {}).items():
        if name not in df:
            raise ValueError(f"Filter column '{name}' is not in this dataset.")
        if isinstance(rule, list):
            frame = frame[frame[name].astype(str).isin([str(value) for value in rule])]
        elif isinstance(rule, dict):
            numeric = pd.to_numeric(frame[name], errors="coerce")
            valid = numeric.notna()
            if rule.get("min") is not None:
                valid &= numeric >= float(rule["min"])
            if rule.get("max") is not None:
                valid &= numeric <= float(rule["max"])
            if rule.get("include_missing"):
                valid |= frame[name].isna()
            frame = frame[valid]
        else:
            raise ValueError("Filters must be a category list or numeric min/max object.")
    return frame.copy()


def wilson_interval(churned, count):
    if not count:
        return None, None
    z = 1.959963984540054
    p = churned / count
    denominator = 1 + z*z/count
    centre = (p + z*z/(2*count)) / denominator
    margin = z*np.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return float(max(0, centre-margin)), float(min(1, centre+margin))


def segment_summary(df, column, config, limit=20):
    if column not in df:
        raise ValueError(f"Unknown segment column: {column}")
    frame = pd.DataFrame({"category": df[column].fillna("(Missing)").astype(str)})
    has_target = bool(config.get("target_column") and config["target_column"] in df)
    if has_target:
        frame["outcome"] = target_values(df, config)
    if frame.category.nunique() > limit:
        keep = frame.category.value_counts().head(limit-1).index
        frame["category"] = frame.category.where(frame.category.isin(keep), "(Other categories)")
    records = []
    for value, group in frame.groupby("category", sort=False):
        count = len(group)
        positive = int(group.outcome.sum()) if has_target else None
        lower, upper = wilson_interval(positive, count) if has_target else (None, None)
        records.append({"category": value, "segment": value, "count": count,
                        "customers": count, "churned": positive,
                        "churn_rate": positive/count if has_target else None,
                        "ci_lower": lower, "ci_upper": upper,
                        "small_segment": count < 30})
    return sorted(records, key=lambda row: row["count"], reverse=True)


def overview(df, config, filters=None):
    frame = apply_filters(df, filters)
    has_target = bool(config.get("target_column"))
    y = target_values(frame, config) if has_target and len(frame) else None
    # Empty filtered frames retain a known target definition without inventing a rate.
    churned = int(y.sum()) if y is not None else (0 if has_target else None)
    rate = float(y.mean()) if y is not None else None
    numeric_features = [f["column"] for f in config.get("features", []) if f["kind"] == "numeric"]
    categorical_features = [f["column"] for f in config.get("features", []) if f["kind"] in {"categorical", "ordinal"}]
    summaries = []
    for column in numeric_features:
        values = pd.to_numeric(frame[column], errors="coerce")
        summaries.append({"feature": column, "mean": values.mean(), "median": values.median(),
                          "min": values.min(), "max": values.max(), "missing": int(values.isna().sum())})
    segments = [{"feature": column, "groups": segment_summary(frame, column, config)}
                for column in categorical_features[:3]]
    findings = []
    if len(frame):
        findings.append(f"The selected view contains {len(frame):,} of {len(df):,} customer records.")
        if has_target:
            findings.append(f"{churned:,} historical records ({rate:.1%}) have the configured positive outcome '{config['positive_label']}'.")
            for item in segments:
                eligible = [r for r in item["groups"] if r["count"] >= 30]
                if len(eligible) > 1:
                    high = max(eligible, key=lambda row: row["churn_rate"])
                    findings.append(f"For {item['feature']}, '{high['category']}' has the highest observed churn among groups with at least 30 records: {high['churn_rate']:.1%} (n={high['count']:,}).")
        else:
            findings.append("No historical churn target is configured. This view is descriptive only; no churn probabilities or model scores are inferred.")
    else:
        findings.append("No customer records match these filters. Clear filters to restore the full view.")
    findings.append("Observed associations do not establish a customer's reason for leaving or an intervention's effect.")
    return json_safe({"rows": len(frame), "total_rows": len(df), "columns_count": len(df.columns),
                      "label_available": has_target, "churned": churned, "retained": len(frame)-churned if has_target else None,
                      "churn_rate": rate, "missing_cells": int(frame.isna().sum().sum()),
                      "duplicate_rows": int(frame.duplicated().sum()), "numeric_summaries": summaries,
                      "class_distribution": [{"name": "Churned", "count": churned}, {"name": "Stayed", "count": len(frame)-churned}] if has_target else [],
                      "segments": segments, "findings": findings, "warnings": config.get("warnings", [])})


def _adjust_pvalues(rows):
    valid = [(i, r["p_value"]) for i, r in enumerate(rows) if r.get("p_value") is not None and np.isfinite(r["p_value"])]
    valid.sort(key=lambda item: item[1])
    running = 1.0
    for rank in range(len(valid)-1, -1, -1):
        index, probability = valid[rank]
        running = min(running, probability * len(valid) / (rank+1))
        rows[index]["p_adjusted"] = float(min(1, running))
    return rows


def association_tests(df, config):
    if not config.get("target_column") or not len(df):
        return []
    y = target_values(df, config)
    rows = []
    for feature in config.get("features", []):
        column = feature["column"]
        item = {"feature": column, "test": "Not estimable", "p_value": None, "p_adjusted": None,
                "effect_size": None, "effect_name": None, "ci_lower": None, "ci_upper": None,
                "n": int(df[column].notna().sum()), "notes": "Observational association; no causal interpretation."}
        if feature["kind"] in {"numeric", "datetime"}:
            if feature["kind"] == "datetime":
                values = prepare_features(df, {"features": [feature]})[0][column]
            else:
                values = pd.to_numeric(df[column], errors="coerce")
            a, b = values[y.eq(0)].dropna(), values[y.eq(1)].dropna()
            if min(len(a), len(b)) >= 2 and a.var()/len(a)+b.var()/len(b) > 0:
                statistic, probability = stats.ttest_ind(b, a, equal_var=False)
                variance = a.var()/len(a) + b.var()/len(b)
                degrees = variance**2 / ((a.var()/len(a))**2/(len(a)-1) + (b.var()/len(b))**2/(len(b)-1))
                difference = b.mean()-a.mean()
                margin = stats.t.ppf(.975, degrees)*np.sqrt(variance)
                pooled = np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
                item.update(test="Welch t-test", statistic=float(statistic), p_value=float(probability),
                            effect_size=float(difference/pooled) if pooled else None, effect_name="Standardised mean difference",
                            ci_lower=float(difference-margin), ci_upper=float(difference+margin),
                            notes="CI is the raw mean difference (churned minus stayed), not its standardized effect. Independent observations and approximate sampling normality assumed.")
        else:
            table = pd.crosstab(df[column], y)
            if min(table.shape, default=0) >= 2 and table.shape[0] <= 50:
                chi2, probability, _, expected = stats.chi2_contingency(table, correction=False)
                n = table.to_numpy().sum()
                item.update(effect_size=float(np.sqrt(chi2/(n*min(table.shape[0]-1, table.shape[1]-1)))), effect_name="Cramer's V")
                if expected.min() >= 1 and (expected < 5).mean() <= .2:
                    item.update(test="Chi-square", statistic=float(chi2), p_value=float(probability), notes="Expected-cell approximation checked; observations assumed independent.")
                elif table.shape == (2, 2):
                    statistic, probability = stats.fisher_exact(table.to_numpy())
                    item.update(test="Fisher exact", statistic=float(statistic), p_value=float(probability), notes="Two-sided exact test for sparse 2x2 table; statistic is unadjusted odds ratio.")
                else:
                    item.update(notes="Sparse multi-category table: chi-square p-value withheld because expected-count assumptions fail.")
        rows.append(item)
    return json_safe(_adjust_pvalues(rows))


def eda(df, config, numeric=None, categorical=None, second=None, filters=None):
    frame = apply_filters(df, filters)
    numeric_columns = [f["column"] for f in config.get("features", []) if f["kind"] == "numeric"]
    categorical_columns = [f["column"] for f in config.get("features", []) if f["kind"] in {"categorical", "ordinal"}]
    if numeric is not None and numeric not in numeric_columns:
        raise ValueError("Choose a configured numeric feature for its distribution.")
    if categorical is not None and categorical not in categorical_columns:
        raise ValueError("Choose a configured categorical/ordinal feature for segment comparisons.")
    if second is not None and second not in [f["column"] for f in config.get("features", [])]:
        raise ValueError("The selected comparison feature is not a configured predictor.")
    numeric = numeric or (numeric_columns[0] if numeric_columns else None)
    categorical = categorical or (categorical_columns[0] if categorical_columns else None)
    has_target = bool(config.get("target_column"))
    y = target_values(frame, config) if has_target and len(frame) else None
    histogram, boxplot, scatter, pair_comparison = [], [], [], []
    if numeric and len(frame):
        values = pd.to_numeric(frame[numeric], errors="coerce")
        finite = values.dropna()
        if len(finite):
            edges = np.histogram_bin_edges(finite, bins=min(25, max(1, int(np.sqrt(len(finite))))))
            for label, mask in [("Stayed", y.eq(0)), ("Churned", y.eq(1))] if y is not None else [("All records", pd.Series(True, index=frame.index))]:
                group = values[mask].dropna()
                counts, _ = np.histogram(group, bins=edges)
                histogram.extend({"bin": f"{edges[i]:.3g}–{edges[i+1]:.3g}", "lower": float(edges[i]), "upper": float(edges[i+1]), "count": int(count), "outcome": label} for i, count in enumerate(counts))
                if len(group):
                    boxplot.append({"outcome": label, "min": group.min(), "q1": group.quantile(.25), "median": group.median(), "q3": group.quantile(.75), "max": group.max(), "count": len(group)})
    categories = segment_summary(frame, categorical, config) if categorical else []
    numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce") if numeric_columns else pd.DataFrame()
    if y is not None:
        numeric_frame = numeric_frame.assign(__historical_churn=y)
    corr = numeric_frame.corr() if len(numeric_frame.columns) else pd.DataFrame()
    if numeric and second in numeric_columns and second != numeric:
        subset = frame.sample(n=min(1000, len(frame)), random_state=42)
        scatter = [{"x": row[numeric], "y": row[second], "outcome": "Churned" if has_target and canonical_outcome(row, config) else "Stayed" if has_target else "All records"} for _, row in subset.iterrows()]
    elif categorical and second in categorical_columns and second != categorical:
        combined = frame.copy()
        combined["__pair"] = frame[categorical].fillna("(Missing)").astype(str) + " / " + frame[second].fillna("(Missing)").astype(str)
        pair_comparison = segment_summary(combined, "__pair", config, limit=20)
    descriptions = []
    for name in numeric_columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        descriptions.append({"feature": name, "count": int(values.count()), "missing": int(values.isna().sum()),
                             "mean": values.mean(), "std": values.std(), "min": values.min(), "q1": values.quantile(.25),
                             "median": values.median(), "q3": values.quantile(.75), "max": values.max()})
    return json_safe({"numeric_columns": numeric_columns, "categorical_columns": categorical_columns,
                      "selected_numeric": numeric, "selected_categorical": categorical, "selected_second": second,
                      "histogram": histogram, "boxplot": boxplot, "categories": categories,
                      "correlation": {"columns": list(corr.columns), "matrix": corr.to_numpy().tolist()},
                      "scatter": scatter, "pair_comparison": pair_comparison,
                      "missing": [{"column": column, "count": int(frame[column].isna().sum()), "rate": float(frame[column].isna().mean()) if len(frame) else None} for column in frame],
                      "descriptive": descriptions, "associations": association_tests(frame, config),
                      "findings": overview(df, config, filters)["findings"],
                      "notes": ["Category displays combine excess levels into Other; uncertainty intervals are unadjusted 95% Wilson intervals.",
                                "Feature-test p-values use Benjamini-Hochberg adjustment across estimable tests. Confidence intervals are not multiplicity-adjusted.",
                                "All descriptive filters leave model evaluation partitions unchanged."]})


def canonical_outcome(row, config):
    from .data_engine import canonical_label
    return canonical_label(row[config["target_column"]]) == canonical_label(config["positive_label"])
