# Actual dataset compatibility benchmarks

Each dataset was uploaded through the HTTP API, explicitly configured and trained separately. These are held-out historical results, not cross-provider transfer tests or causal intervention evidence.

| Dataset | Rows | Features | Target | Selected model | ROC-AUC | PR-AUC | F1 | Recommendation status | Result |
|---|---:|---:|---|---|---:|---:|---:|---|---|
| iranian | 3150 | 10 | Churn | XGBoost | 0.9665 | 0.8717 | 0.7429 | 2 evidence-qualified; 1 limited-evidence segments | passed |
| banking | 10000 | 6 | Exited | XGBoost | 0.7656 | 0.5153 | 0.2567 | 1 evidence-qualified; 2 limited-evidence segments | passed |
| ecommerce | 5630 | 15 | Churn | Random Forest | 0.9674 | 0.8734 | 0.7817 | 5 evidence-qualified; 3 limited-evidence segments | passed |
| ibm | 7043 | 15 | Churn | XGBoost | 0.8487 | 0.6643 | 0.5867 | 4 evidence-qualified; 0 limited-evidence segments | passed |

## Configurations and limitations

- Full standard tuning (unless an individual row records fast=true), fixed seed 42 and validation PR-AUC selection; equal-weight hybrid is compared without test-set selection.
- Explicit mappings are in configurations.py. Training-only preprocessing and calibration use independent stratified training/validation/test cohorts. Exact duplicate rows are removed and reported by each model's metadata.
- Iranian Status is excluded conservatively after timing review. UCI documents Charge Amount bands 0–9, but the file includes code 10: it is treated as an unordered category and its pricing mapping remains unconfirmed. Age and Age Group are excluded.
- Banking names/identifiers and age, gender, geography and salary are excluded; six business predictors remain. Activity is an engagement indicator, not evidence of dissatisfaction or financial difficulty.
- E-commerce uses the actual E Comm worksheet, not Data Dict. Gender, marital status and city tier are excluded. SatisfactionScore is explicitly ordered 1–5; missing numeric values are imputed on training only.
- The generic IBM benchmark uses a separate 15-feature configuration and separate new artifacts; it does not alter the original 19-feature case study or its saved results.
- Banking, e-commerce and IBM prediction horizons are unknown. Their results are retrospective sample evaluations, not verified prospective forecasts.
- External raw files are excluded from Git. Source links, hashes and license metadata remain in the local download provenance. No unavailable benchmark is replaced by synthetic data.

### iranian

All recorded HTTP checks passed.

Checks: real file upload and schema profile, explicit target/positive label/features/semantics configuration, true background training and progress, four held-out models and calibration curves, dynamic overview and EDA, global and individual explanation, customer probability and historical outcome separation, evidence-linked recommendations, unlabelled scoring with duplicate-row preservation, incompatible schema and unknown model rejection, editable hypothetical budget/economics scenario, immutable/outcome what-if rejection, valid report and CSV exports, test metrics unchanged by exploration

### banking

All recorded HTTP checks passed.

Checks: real file upload and schema profile, explicit target/positive label/features/semantics configuration, true background training and progress, four held-out models and calibration curves, dynamic overview and EDA, global and individual explanation, customer probability and historical outcome separation, evidence-linked recommendations, unlabelled scoring with duplicate-row preservation, incompatible schema and unknown model rejection, editable hypothetical budget/economics scenario, immutable/outcome what-if rejection, valid report and CSV exports, test metrics unchanged by exploration

### ecommerce

All recorded HTTP checks passed.

Checks: real file upload and schema profile, explicit target/positive label/features/semantics configuration, true background training and progress, four held-out models and calibration curves, dynamic overview and EDA, global and individual explanation, customer probability and historical outcome separation, evidence-linked recommendations, unlabelled scoring with duplicate-row preservation, incompatible schema and unknown model rejection, editable hypothetical budget/economics scenario, immutable/outcome what-if rejection, valid report and CSV exports, test metrics unchanged by exploration

### ibm

All recorded HTTP checks passed.

Checks: real file upload and schema profile, explicit target/positive label/features/semantics configuration, true background training and progress, four held-out models and calibration curves, dynamic overview and EDA, global and individual explanation, customer probability and historical outcome separation, evidence-linked recommendations, unlabelled scoring with duplicate-row preservation, incompatible schema and unknown model rejection, editable hypothetical budget/economics scenario, confirmed actionable what-if sensitivity, immutable/outcome what-if rejection, valid report and CSV exports, test metrics unchanged by exploration
