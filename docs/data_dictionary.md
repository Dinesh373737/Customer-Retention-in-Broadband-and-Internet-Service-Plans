# Dataset and data dictionary

This application analyses the historical IBM Telco Customer Churn sample. The supplied CSV has 7,043 customer records and 21 columns. Counts and summaries in the application are recomputed after validation and filtering; other valid uploads need not contain the same number of rows. It contains account and subscription attributes, not live network measurements, complaints, satisfaction scores or retention-campaign outcomes.

Source: [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d). See `data/README.md` and `data/provenance.json` for the bundled file's provenance.

| Field | Meaning and accepted values | Treatment |
| --- | --- | --- |
| customerID | Customer reference | Navigation only; excluded from all predictors. Missing references are generated and labelled in the quality report. |
| gender | Female, Male | Categorical; dataset's original binary coding is retained. |
| SeniorCitizen | Source 0/1; cleaned No/Yes | Categorical status, not a continuous age measure. |
| Partner | No, Yes | Categorical. |
| Dependents | No, Yes | Categorical. |
| tenure | Whole nonnegative months with the provider | Numeric. |
| PhoneService | No, Yes | Categorical. |
| MultipleLines | No, Yes, No phone service | The service-unavailable category is preserved. |
| InternetService | DSL, Fiber optic, No | Categorical; no connection-speed measurements are available. |
| OnlineSecurity | No, Yes, No internet service | Subscription attribute. |
| OnlineBackup | No, Yes, No internet service | Subscription attribute. |
| DeviceProtection | No, Yes, No internet service | Subscription attribute. |
| TechSupport | No, Yes, No internet service | Subscription attribute; absence does not imply a technical problem. |
| StreamingTV | No, Yes, No internet service | Subscription attribute. |
| StreamingMovies | No, Yes, No internet service | Subscription attribute. |
| Contract | Month-to-month, One year, Two year | Categorical. |
| PaperlessBilling | No, Yes | Categorical. |
| PaymentMethod | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Categorical. |
| MonthlyCharges | Nonnegative charge per month | Numeric, in dataset source-currency units; no currency conversion is implied. |
| TotalCharges | Nonnegative total charge | Numeric; whitespace values become missing, not zero. |
| Churn | Source No/Yes; cleaned 0/1 | Historical target, never a predictor. Required for analysis/training; optional for batch scoring. |

## Validation and preprocessing

`retention.data.load_data` accepts a local path, UTF-8 CSV bytes or a file-like object. It requires all 19 predictor columns, trims surrounding whitespace, normalises known categories without changing their meaning, and rejects unknown category names or invalid/missing labels. All three numeric fields must be finite and nonnegative when present; tenure must be in whole months. Invalid text in a numeric field is an actionable error, not silently imputed. Missing predictor values remain missing so the model pipeline can learn imputation from its training partition only.

The loader keeps a canonical schema. It reports and excludes extra columns. In historical analysis/training mode it removes exact duplicate cleaned records and rejects conflicting repeated customer IDs to reduce the risk of the same customer entering both training and test partitions. In scoring mode it preserves every uploaded row, including duplicates and repeated IDs, and warns about repeated references. Identical features need not mean that two customers are the same person. It reports inconsistent phone/internet service combinations and retains the supplied values; there is no reliable basis to invent corrections. The quality report records row counts, per-field missing values, class counts, duplicates, service inconsistencies, warnings, processing decisions and a content hash. An upload without IDs receives generated record references used only for navigation.

The bundled data contains 11 whitespace-only `TotalCharges` cells. These are converted to missing values. No other missing predictor values, exact duplicates or service inconsistencies were found when the bundled file was validated. These findings are specific to the supplied file, not hardcoded validation assumptions.

`FEATURE_COLUMNS` explicitly excludes `customerID` and `Churn`. Categorical imputation and one-hot encoding, numeric median imputation and scaling (where needed) are fitted inside model pipelines after the stratified train/test split. A training-derived transform is reused for validation, testing and scoring. A changed dataset content hash invalidates cached results.

## Derived summaries and inference

Tenure bands use right-inclusive intervals: **0–12**, **13–24**, **25–48**, **49–60**, and **61+** whole months. Missing tenure stays missing. Segment tables include customer counts and observed churn rates, plus 95% Wilson binomial confidence intervals; very small groups should be interpreted cautiously. Dashboard filtering describes the selected historical sample and does not redefine the held-out model test set.

Categorical association tables use a chi-square independence test when every expected cell count is at least 1 and no more than 20% are below 5. Sparse 2×2 tables use a two-sided Fisher exact test. For larger sparse tables, a chi-square p-value is withheld and the reason is shown. Effect sizes are bias-corrected Cramer's V. Numeric comparisons use a Welch t-test and Hedges' g, with a 95% interval for the **raw mean difference, churned minus non-churned**. Welch tests assume independent customer records and approximately normal sampling of means; small groups receive a caution. Benjamini–Hochberg adjustment controls false discovery rate across the estimable feature tests. Intervals remain unadjusted.

A separate, prespecified descriptive logistic model estimates adjusted odds ratios and 95% Wald confidence intervals. Predictors are tenure per 12 months, monthly charges per 10 units, contract, internet service and senior-citizen status. Reference categories are stated with the actual fit. `TotalCharges` is excluded to limit collinearity. Complete cases are used; the output reports omitted records and explicitly declines to report unstable fits, singular matrices, insufficient samples or perfect separation. This full-sample inferential fit is separate from the train/test predictive model and does not supply held-out performance claims.

All results are observational associations in the historical sample. Statistical significance, a model coefficient or a feature importance value cannot show that changing a subscription will prevent churn. Offer effectiveness and incremental customer retention must be measured through suitable interventions; this dataset does not supply those outcomes.
