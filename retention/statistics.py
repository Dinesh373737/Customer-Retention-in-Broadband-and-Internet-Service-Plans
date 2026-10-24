"""Exploratory association tests; observational results do not identify causal effects."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from retention.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

ASSOCIATION_COLUMNS = [
    "feature", "test", "statistic", "p_value", "p_adjusted", "effect_size",
    "effect_name", "ci_lower", "ci_upper", "n", "assumptions_met", "notes",
]
ODDS_RATIO_COLUMNS = ["term", "coefficient", "standard_error", "odds_ratio", "ci_lower", "ci_upper", "p_value"]


def _benjamini_hochberg(values: pd.Series) -> np.ndarray:
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values.to_numpy(dtype=float)))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(values.iloc[valid].to_numpy())]
    ranked = values.iloc[order].to_numpy() * len(order) / np.arange(1, len(order) + 1)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def association_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-square/Fisher associations and Welch comparisons, with BH FDR adjustment.

    Categorical effect size is bias-corrected Cramer's V; numerical effect size is
    Hedges' g (churned minus non-churned). Numerical CI columns describe the raw
    mean difference, not the standardised effect. All intervals are unadjusted 95%.
    """
    if "Churn" not in df or not df["Churn"].isin([0, 1]).all():
        raise ValueError("Association tests require non-missing binary historical Churn labels.")
    rows: list[dict] = []
    for column in CATEGORICAL_COLUMNS:
        if column not in df:
            continue
        subset = df[[column, "Churn"]].dropna()
        table = pd.crosstab(subset[column], subset["Churn"])
        row = {
            "feature": column, "test": "Not estimable", "statistic": np.nan,
            "p_value": np.nan, "effect_size": np.nan, "effect_name": "Bias-corrected Cramer's V",
            "ci_lower": np.nan, "ci_upper": np.nan, "n": len(subset),
            "assumptions_met": False, "notes": "At least two observed feature categories and both churn classes are required.",
        }
        if min(table.shape, default=0) >= 2:
            chi2, p_value, _, expected = stats.chi2_contingency(table, correction=False)
            sufficient = bool(expected.min() >= 1 and np.mean(expected < 5) <= 0.2)
            n = table.to_numpy().sum()
            row_count, col_count = table.shape
            phi2_corrected = max(0.0, chi2 / n - ((col_count - 1) * (row_count - 1)) / (n - 1)) if n > 1 else np.nan
            corrected_rows = row_count - (row_count - 1) ** 2 / (n - 1) if n > 1 else np.nan
            corrected_cols = col_count - (col_count - 1) ** 2 / (n - 1) if n > 1 else np.nan
            denominator = min(corrected_rows - 1, corrected_cols - 1)
            row["effect_size"] = np.sqrt(phi2_corrected / denominator) if denominator > 0 else np.nan
            row.update({"test": "Chi-square", "statistic": float(chi2), "p_value": float(p_value), "assumptions_met": sufficient})
            row["notes"] = f"Minimum expected cell count {expected.min():.2f}; {np.mean(expected < 5):.0%} of expected counts below 5. Complete cases; independent-customer assumption."
            if not sufficient:
                if table.shape == (2, 2):
                    fisher = stats.fisher_exact(table.to_numpy())
                    row.update({"test": "Fisher exact", "statistic": float(fisher.statistic), "p_value": float(fisher.pvalue), "assumptions_met": True})
                    row["notes"] += " Sparse 2×2 table: two-sided Fisher test used; statistic is the unadjusted odds ratio."
                else:
                    row.update({"test": "Not tested (sparse table)", "p_value": np.nan})
                    row["notes"] += " Chi-square approximation withheld because expected-cell assumptions fail."
        rows.append(row)
    for column in NUMERIC_COLUMNS:
        if column not in df:
            continue
        negative = df.loc[df["Churn"] == 0, column].dropna().to_numpy(dtype=float)
        positive = df.loc[df["Churn"] == 1, column].dropna().to_numpy(dtype=float)
        row = {
            "feature": column, "test": "Welch t-test", "statistic": np.nan, "p_value": np.nan,
            "effect_size": np.nan, "effect_name": "Hedges' g (churned − non-churned)",
            "ci_lower": np.nan, "ci_upper": np.nan, "n": len(negative) + len(positive),
            "assumptions_met": False, "notes": "At least two non-missing records in each churn class are required.",
        }
        if len(negative) >= 2 and len(positive) >= 2:
            n0, n1 = len(negative), len(positive)
            v0, v1 = np.var(negative, ddof=1), np.var(positive, ddof=1)
            variance = v0 / n0 + v1 / n1
            difference = float(np.mean(positive) - np.mean(negative))
            if variance > 0:
                test = stats.ttest_ind(positive, negative, equal_var=False)
                degrees = variance ** 2 / ((v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
                margin = stats.t.ppf(0.975, degrees) * np.sqrt(variance)
                pooled_sd = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
                correction = 1 - 3 / (4 * (n0 + n1) - 9)
                row.update({
                    "statistic": float(test.statistic), "p_value": float(test.pvalue),
                    "effect_size": float(correction * difference / pooled_sd),
                    "ci_lower": float(difference - margin), "ci_upper": float(difference + margin),
                    "assumptions_met": bool(min(n0, n1) >= 30),
                    "notes": f"Raw mean difference = {difference:.3f} (churned minus non-churned); CI is for this raw difference. n(non-churned)={n0}, n(churned)={n1}. Unequal variances allowed; independent observations and approximate sampling normality assumed."
                    + (" Small groups: inspect distributions before inference." if min(n0, n1) < 30 else " Large groups support the mean's normal approximation."),
                })
            else:
                row["notes"] = "Both groups have zero variance; Welch test is not estimable."
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=ASSOCIATION_COLUMNS)
    result["p_adjusted"] = _benjamini_hochberg(result["p_value"])
    result["notes"] += " P-values adjusted together across estimable feature tests using Benjamini–Hochberg FDR; confidence intervals are unadjusted."
    return result[ASSOCIATION_COLUMNS]


def logistic_odds_ratios(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Fit a prespecified, complete-case, low-dimensional inferential logit model.

    This full-sample descriptive model is separate from the held-out predictive
    pipeline. TotalCharges is omitted because it closely tracks tenure × charges.
    No stepwise selection or fabricated fallback confidence intervals are used.
    """
    empty = pd.DataFrame(columns=ODDS_RATIO_COLUMNS)
    try:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationError, PerfectSeparationWarning
    except ImportError:
        return empty, "Odds-ratio analysis is unavailable because statsmodels is not installed. Install the project's requirements and retry."
    required = ["Churn", "tenure", "MonthlyCharges", "Contract", "InternetService", "SeniorCitizen"]
    missing = [column for column in required if column not in df]
    if missing:
        return empty, "Odds-ratio analysis requires columns: " + ", ".join(missing)
    complete = df[required].dropna().copy()
    if not complete["Churn"].isin([0, 1]).all() or complete["Churn"].nunique() != 2:
        return empty, "Odds ratios cannot be estimated: both valid historical Churn classes are required."
    design = pd.DataFrame({
        "tenure (per 12 months)": complete["tenure"] / 12,
        "MonthlyCharges (per 10 units)": complete["MonthlyCharges"] / 10,
    }, index=complete.index)
    references = []
    for column, levels in {
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "SeniorCitizen": ["No", "Yes"],
    }.items():
        observed_levels = [level for level in levels if level in set(complete[column])]
        if not observed_levels:
            return empty, f"Odds-ratio analysis cannot interpret the categories in {column}; validate the dataset first."
        references.append(f"{column}: {observed_levels[0]}")
        for level in observed_levels[1:]:
            design[f"{column}: {level}"] = (complete[column] == level).astype(float)
    # Constant input terms cannot have estimated slopes and are reported explicitly.
    constant_columns = [column for column in design if design[column].nunique() <= 1]
    design = design.drop(columns=constant_columns)
    design = sm.add_constant(design.astype(float), has_constant="add")
    if len(complete) < max(30, 10 * design.shape[1]):
        return empty, f"Odds ratios are not reported: {len(complete)} complete records are insufficient for {design.shape[1]} parameters under the minimum 10-records-per-parameter guardrail."
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        return empty, "Odds ratios are not reported: the prespecified design matrix is singular (perfectly collinear predictors). Consider a larger, more varied sample."
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = sm.Logit(complete["Churn"].astype(float), design).fit(disp=False, maxiter=150)
        separation = any(isinstance(item.message, PerfectSeparationWarning) for item in caught)
        if separation or not model.mle_retvals.get("converged", False):
            return empty, "Odds ratios are not reported: the logistic model did not converge or perfect separation was detected; confidence intervals would be unreliable."
        confidence = model.conf_int()
        if not np.isfinite(np.r_[model.params.to_numpy(), confidence.to_numpy().ravel()]).all():
            return empty, "Odds ratios are not reported: coefficient estimates or confidence intervals are non-finite, indicating an unstable fit."
        with np.errstate(over="ignore"):
            ratios = np.exp(model.params)
            lower = np.exp(confidence[0])
            upper = np.exp(confidence[1])
        if not np.isfinite(np.r_[ratios.to_numpy(), lower.to_numpy(), upper.to_numpy()]).all():
            return empty, "Odds ratios are not reported: exponentiating unstable estimates produced non-finite intervals."
        result = pd.DataFrame({
            "term": model.params.index, "coefficient": model.params.to_numpy(),
            "standard_error": model.bse.to_numpy(), "odds_ratio": ratios.to_numpy(),
            "ci_lower": lower.to_numpy(), "ci_upper": upper.to_numpy(), "p_value": model.pvalues.to_numpy(),
        })
        # The intercept describes baseline odds rather than a feature association.
        result = result[result["term"] != "const"].reset_index(drop=True)
        note = (
            f"Full-sample descriptive logistic model, n={len(complete):,} complete records ({len(df) - len(complete):,} omitted for missing model fields). "
            "Predictors were prespecified: tenure per 12 months, monthly charges per 10 source-currency units, contract, internet service and senior-citizen status. "
            "Reference categories: " + "; ".join(references) + ". "
            "TotalCharges is excluded to limit collinearity. Odds ratios above 1 indicate higher adjusted churn odds; below 1 indicate lower adjusted odds. "
            "These are adjusted associations, not causal effects, risk ratios or predictions from the held-out ML model. "
            "95% Wald confidence intervals and unadjusted coefficient p-values assume independent observations, a correctly specified logit, linear numeric effects on log-odds, and no important omitted confounding."
        )
        if constant_columns:
            note += " Constant terms omitted: " + ", ".join(constant_columns) + "."
        return result, note
    except (np.linalg.LinAlgError, PerfectSeparationError, ValueError, FloatingPointError) as error:
        return empty, f"Odds ratios could not be estimated reliably ({type(error).__name__}: {error}). The analysis does not substitute coefficients or confidence intervals."
