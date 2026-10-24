# Completed verification

Verified on 23 October 2026 using Python 3.12 on Windows.

- **63 automated tests passed** (41.60 seconds). The single warning was the intentional unseen-category inference test, which verifies the encoder's supported fallback. Schema cleaning, malformed headers, duplicate policies, training-only preprocessing, all four real models, reproducibility, persistence, explanation sensitivity, statistical inference, recommendation evidence and economics were checked.
- **13 dashboard checks passed**, including all eight pages, a contract filter changing the KPI to its actual group size, zero-cost economic denominators, model-table ranking, batch scoring availability with empty historical results, and the SHAP calculation button. Machine-readable evidence: `artifacts/verification.json`.
- Full training used **7,043 actual records**: 4,225 training, 1,409 validation and 1,409 held-out test records. Preprocessing was fitted only on training records. Exact indices and settings are in `artifacts/training_metadata.json`.
- XGBoost was selected using validation average precision. The untouched test comparison is in `artifacts/model_comparison.csv`. The hybrid leads ROC-AUC and Brier score on that test set; XGBoost leads average precision. No overall ensemble superiority is assumed.
- Actual SHAP summaries succeeded for Logistic Regression, Random Forest and XGBoost using 10 held-out records each. The corresponding CSVs contain finite, model-derived attributions and explicitly identify their raw-estimator scales. See `artifacts/explanation_checks.json`.
- Descriptive statistics, association tests, adjusted logistic odds ratios and confidence intervals were generated from the full historical dataset using `python -m scripts.analyze`.
- The local dashboard was opened and visually inspected in the browser. The final server binds only to `127.0.0.1:8501`.

The Codex execution sandbox blocks Python's Windows local socket initialization, which stalled the first Streamlit test attempt before application execution. Server and dashboard verification subsequently ran successfully with normal local socket access. This is an execution-environment restriction, not a requirement to change system firewall or security settings. Ordinary local execution follows the README.

Re-run the README's test and smoke commands when changing model logic or dashboard behavior. Test metrics describe this held-out historical sample; financial assumptions remain hypothetical and no retention outcome has been observed.
