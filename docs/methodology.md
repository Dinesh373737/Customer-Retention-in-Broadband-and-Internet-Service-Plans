# Methodology and limitations

This is a local academic decision-support prototype for **Customer Retention in Broadband and Internet Service Plans**, using historical IBM Telco sample records. It has no live provider connection, telemetry, complaint history, customer satisfaction measures, customer contact capability or observed retention-offer outcomes.

## Data and validation

The full supplied CSV is available in `data/Telco-Customer-Churn.csv`. Its original source is the [IBM Telco project](https://github.com/IBM/telco-customer-churn-on-icp4d). The application verifies each file rather than assuming 7,043 rows or an identical schema. It checks required predictor columns, categories, numeric conversion, negative values, missingness, exact duplicates, conflicting identifiers and target labels. Whitespace and case are normalized to canonical categories. Extra fields never become model inputs. Customer identifiers are for navigation only.

The 11 blank `TotalCharges` values in the supplied full dataset become missing numeric values. Numerical medians, categorical imputations, scaling and encoding are learned inside model pipelines from training records. Missing values are not filled from the complete dataset. See the data dictionary and generated quality report for exact decisions. Service combinations are checked; unsupported inconsistencies are disclosed rather than silently repaired.

Tenure groups use reproducible boundaries documented in the dashboard. All group comparisons include record counts and uncertainty where appropriate. Dashboard filters change descriptive analysis, recommendations and customer tables. They never redefine the held-out evaluation sample.

## Predictive evaluation

The target is historical `Churn` (Yes=1, No=0). Inputs exclude both `Churn` and `customerID`, and explicitly select the 19 supported predictor columns. A reproducible stratified split separates training, validation and final test data. Training-only cross-validation selects a small hyperparameter grid for Random Forest and XGBoost. Logistic Regression supplies an interpretable baseline. Class handling uses the training partition's distribution.

The additional hybrid really averages the three models' probabilities with non-negative equal weights summing to one. Calibration and the preferred model are assessed without test labels. Exact partitions, hyperparameters, calibration policy and package versions are recorded with saved artifacts. The test comparison can show that the hybrid performs worse than a component. Changing a chart ranking metric does not retrain or select a deployment model. No test metric is optimized.

Accuracy is the fraction of correct historical classifications; precision is the fraction of flagged records that actually churned; recall is the fraction of historical churners flagged; F1 balances precision and recall. ROC-AUC describes ranking across thresholds. PR-AUC (average precision) is especially useful with unequal class frequencies. Brier score and reliability plots examine probability quality. Neither ranking scores nor calibration establish intervention profitability.

The initial illustrative risk policy is low below 30%, medium from 30% to below 60%, and high at least 60%. The default binary evaluation decision boundary is 50%. These serve different purposes. Both are editable for exploration; they are not optimized on the final test sample. Threshold-versus-precision-and-recall exploration uses validation predictions. Probability is uncertainty, not a forecast with certainty. Historical full-data risk scores include training records and must not be reported as held-out accuracy.

## Explanations and statistics

Regularized Logistic Regression coefficients show direction in encoded, standardized feature space. Exponentiated coefficients describe odds ratios under that model's coding and scale, not percentage-point probability changes. Tree importance reflects predictive split contributions and can favor certain features. Optional SHAP explanations use the fitted tree and transformed columns. When unavailable, the app explicitly labels its coefficient and feature-perturbation fallback, which is not SHAP. Individual feature changes may produce unrealistic combinations and do not identify a causal reason for leaving.

A separate prespecified, lower-dimensional Statsmodels logistic model supplies odds ratios and confidence intervals. It is an exploratory association analysis of historical records, not the predictive model's held-out score. Descriptive numerical comparisons use Welch tests and confidence intervals; categorical comparisons use chi-square when expected counts allow it, with appropriate sparse-table treatment. Effect sizes and Benjamini-Hochberg adjusted p-values help avoid treating a collection of small p-values as proof of business importance. Confidence intervals rely on observational sampling and model assumptions; they do not ensure provider-wide generalizability.

## Retention decisions and economics

Transparent rules link selected account features to actual segment churn rates and counts. An elevated segment rate can motivate a review, optional onboarding conversation or support information. No subscription to support does not establish a support incident. Customers may decline engagement; no contact is made. Response is a conceptual workflow stage until actual intervention results exist.

Financial scenarios are hypothetical and editable. All monetary values use the chosen currency and contribution period. Contribution means margin after ongoing service costs and before campaign costs, not gross billing revenue. For N targeted customers, cost c each, additional campaign cost F, incremental retention success u and contribution margin v per additional retained customer:

- Campaign cost = N × c + F.
- Additional retained customers = N × u (an expectation, possibly fractional).
- Incremental contribution = N × u × v.
- Net benefit = N × u × v − (N × c + F).
- ROI = net benefit / campaign cost × 100%, undefined if cost is zero.
- Break-even success = campaign cost / (N × v), undefined with zero denominator; values above 100% mean the assumptions cannot break even.
- Break-even value = campaign cost / (N × u), undefined with zero denominator.

The calculator never counts customers who would have stayed anyway as retained because of a campaign. Predicted churn is not incremental treatment success. Sensitivity analysis varies cost, success and contribution. A controlled pilot with a comparable untreated group is needed to estimate incremental uplift; the IBM sample does not provide these outcomes.

## Monitoring after a future deployment

This prototype allows rerunning analysis and training on new uploaded labelled files. In a future deployment, track feature drift, calibration, recall/precision, intervention uplift and contribution over time; establish label availability and governance; validate retrained versions on a later independent cohort. There is no automatic live learning in this application. Saved artifacts are loaded only when their dataset fingerprint matches; changing data triggers an explicit new analysis.
