# Retention Lab

**Customer Retention in Broadband and Internet Service Plans** — a local Principles of Management and Economics project built with Streamlit, Pandas, Plotly and real fitted machine-learning pipelines.

The dashboard analyzes the historical IBM Telco sample, compares Logistic Regression, Random Forest, XGBoost and a genuine equal-weight probability ensemble, and connects customer scores to transparent retention reviews and editable incremental campaign economics. Historical labels, model estimates and hypothetical scenarios are explicitly distinguished. No provider connection, customer contact or live learning is claimed.

## Run on Windows

Use Python 3.11 or 3.12. Open PowerShell in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open **http://localhost:8501**. On this workspace the environment and full-data trained artifact have already been prepared. You can launch directly with the third command, or run `launch.ps1` if your PowerShell script policy allows it. Installation needs an internet connection; analysis runs locally afterward.

For a fresh clone, trained binary artifacts are excluded from Git. Click **Train / retrain models** in the sidebar, or run `python -m scripts.train` with the project environment before launching, to reproduce the saved pipelines and model comparison.

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Optional SHAP global plots:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

Without SHAP, actual fitted-model importances and individual probability perturbations remain available and are explicitly labelled. A SHAP failure displays its reason instead of inventing an explanation.

## Data, training and reproduction

The full actual CSV is included at `data/Telco-Customer-Churn.csv`. The supplied file contains **7,043 rows and 21 columns**. Source documentation: [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d). See `data/provenance.json` for the local file hash and `data/README.md` for provenance limits. The application validates uploaded files independently of these counts.

Choose another CSV in **Dataset & data quality**, enter a local path, or set an environment override before launching:

```powershell
$env:TELCO_DATA_PATH = 'C:\path\to\Telco-Customer-Churn.csv'
```

A training file needs the 19 documented predictors and the `Churn` label. `customerID` is optional for navigation and excluded from predictors. The **Customer Risk Explorer → Batch scoring** accepts the same predictor columns without a target label. Download the scored CSV from that view. Extra columns never become predictors. Malformed categories, labels and numeric values produce actionable validation errors.

When the configured full file is unavailable, the application visibly labels its fallback of 240 **real sampled records**. No synthetic data are presented as actual data. A model artifact is used for evaluation only if its dataset fingerprint matches the loaded file. Uploading a new dataset unlocks analysis immediately; choose **Train / retrain models** to generate matching predictions and evaluation. Training failure is shown with diagnostics, never replaced with random scores.

Regenerate full-data artifacts and downloadable result files:

```powershell
.\.venv\Scripts\python.exe -m scripts.train
# A different historical dataset:
.\.venv\Scripts\python.exe -m scripts.train --data 'C:\path\to\customers.csv' --output artifacts\other_analysis
# Smaller tuning search for a demonstration, still real training:
.\.venv\Scripts\python.exe -m scripts.train --fast --output artifacts\quick_check
```

To validate and export exploratory/statistical results separately, without training models:

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze
```

This saves descriptive statistics, association tests, actual inferential odds ratios with intervals, segment rates and the data-quality audit in `artifacts/`.

Full training uses a modest grid and two CPU threads per tree model. Runtime depends on the laptop. The saved bundle contains fitted preprocessing, component models, calibration, ensemble, exact partition indices, training baselines and final probabilities. Only load joblib files created by this application from a trusted local source. Scikit-learn and XGBoost versions are pinned for reproducible artifacts; retrain when changing versions.

## Dashboard

| Page | Working features |
|---|---|
| Executive Overview | Six computed KPIs, churn distribution, contract/internet/tenure comparisons, charge distributions and computed insights |
| Exploratory Analysis | All requested categorical comparisons, numeric distributions/scatter/correlation, segment counts, confidence intervals, descriptive statistics, association tests and inferential odds ratios |
| Model Comparison | Four real models on the same held-out records, all metrics, chart ranking control, ROC/PR curves, matrices, actual/predicted summary, probability distributions, validation threshold trade-offs and calibration |
| Explainability | Logistic coefficients and odds ratios, forest/XGBoost importances, optional global SHAP and summary plots, actual selected-model individual sensitivities |
| Customer Risk Explorer | Business/risk filters, sortable scores, historical labels clearly separate, selected-customer details, CSV export, unlabelled batch scoring and contract/charges what-if scenarios |
| Retention Centre | Evidence-linked customer suggestions, segment rates and sample sizes, proposed objectives and caveats, downloadable segment recommendations |
| Economic Impact | Editable target count/cost/uplift/contribution/period/currency assumptions, incremental net benefit/ROI/break-even calculations and cost/success/value sensitivity |
| Methodology & Journey | Source/schema/quality, full methodology and limitations, all seven customer stages and downloadable analysis report |

Sidebar business filters affect descriptive charts, recommendations and customer exploration consistently. They do not change the test cohort or reported model scores. Empty filtered selections produce explanatory states. Risk boundaries are adjustable, initially 30%/60%. The initial binary classification boundary is 50%; the validation-derived F1 boundary can be inspected separately. Choosing a metric to sort the final model table is an exploratory display preference, not automatic model selection.

## Method and leakage prevention

- The reproducible seed is 42. The target is normalized to Yes=1/No=0. Identifiers and labels never enter the feature matrix.
- A stratified **60% training / 20% validation / 20% test** split occurs before any learned preprocessing. Exact duplicate training records are removed, and conflicting repeated customer IDs are rejected before splitting.
- Numeric medians, categorical imputations, numeric scaling and dropped-reference one-hot encoding are fitted within training pipelines. The 11 whitespace-only `TotalCharges` values in the supplied file remain missing until that training-only imputation. Unknown schema categories are rejected clearly.
- Random Forest and XGBoost use small training-only cross-validation searches, ranked by average precision. Class weighting uses training information. Logistic Regression is the interpretable baseline.
- Sigmoid calibration is fitted using **out-of-fold training predictions** and enabled for a component only if its validation Brier score improves. The validation cohort also selects the preferred model by PR-AUC and computes a suggested F1 boundary. Models remain fitted to the training cohort so validation and test remain independent.
- The hybrid combines the three actual calibrated-or-raw component probabilities with fixed equal weights. Weights sum to one and are never tuned to the test set. No improvement is assumed.
- All final metrics use identical untouched test records. PR-AUC is reported as average precision; default classification metrics use a common 0.5 boundary. Final test results do not choose weights, calibration or the preferred model.
- Full historical customer scoring contains training and validation records and is **not** a new held-out performance result. What-if changes are model scenarios, not evidence that changing a contract or price will prevent churn.

The separate Statsmodels inferential logit is prespecified and reports adjusted odds ratios and 95% intervals. It is descriptive full-sample association analysis, distinct from the predictive test evaluation. Numerical tests use Welch comparisons and effect sizes; categorical tests check expected-cell assumptions, use Fisher for sparse 2×2 tables, and withhold unsupported larger sparse tests. Benjamini-Hochberg correction covers the feature-test family. Statistical significance and predictive importance are not causation or campaign profitability.

## Economics

Campaign cost = targeted × contact/offer cost + additional costs. Additional retained customers = targeted × **assumed incremental intervention success**, excluding people who would have stayed anyway. Incremental contribution = additional retained × contribution margin over the stated period. Net benefit subtracts campaign cost; ROI divides net benefit by campaign cost when positive. Undefined ROI or break-even denominators display as unavailable, never infinity.

All monetary values and uplift assumptions are hypothetical and editable. Charges in the IBM data do not establish retention cost, causal success or contribution margin. Recommendations suggest review without applying discounts or contacting customers. See `docs/methodology.md` for formulas and assumptions.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.smoke_dashboard
```

Tests cover schema cleaning and invalid inputs, missing values, independent partitions and predictor leakage, all four fitted models, reproducible probabilities, exact ensemble arithmetic, saved inference, explanation fallback, statistical edge cases, risk boundaries and economic formulas. The dashboard smoke script opens every page with Streamlit's application test harness and exercises key controls. See `artifacts/verification.json` after a verification run.

## Structure and deliverables

```text
app.py                         Streamlit entry point
launch.ps1                     Local Windows launch helper
requirements*.txt              Core and optional dependencies
retention/
  data.py                      Schema validation, quality audit, content hashing
  preprocessing.py             Training-fitted imputers, encoders and scaling
  eda.py                       Group rates, uncertainty, computed insights
  statistics.py                Tests, effect sizes, odds ratios and intervals
  modeling.py                  Reproducible tuning, training and persistence
  ensemble.py                  Actual probability-model wrappers and soft voting
  evaluation.py                Metrics and validation threshold selection
  explainability.py            Fitted importances, perturbations, optional SHAP
  recommendations.py           Transparent evidence-linked retention rules
  economics.py                 Incremental scenarios and sensitivity
  dashboard/                   Layout, navigation, controls and eight pages
scripts/                       CLI training and dashboard verification
tests/                         Automated unit/integration checks
data/                          Full historical CSV, real fallback, provenance
artifacts/                     Trained joblib, actual metrics, metadata, test predictions
docs/                          Requirements, dictionary, method, workflow, report outline
```

Generated outputs include `artifacts/model_comparison.csv`, `validation_metrics.csv`, `held_out_predictions.csv`, `training_metadata.json`, `data_quality.json`, `segment_recommendations.csv` and `results.md`. The dashboard also exports filtered risk records, comparison metrics, recommendations, scenario results and its report.

Read `docs/data_dictionary.md`, `docs/methodology.md`, `docs/customer_journey.md` and `docs/project_report_outline.md` for submission material. The attached requirements are preserved in `docs/project_requirements.md`.

## Scope and limits

These are findings about a historical IBM sample, not all broadband providers. The dataset has no complaints, network speed, real-time usage, intervention outcomes or true contribution margins. Feature associations cannot prove cancellation reasons or causal treatment effects. Customer response is a conceptual journey stage. Retraining is explicitly initiated on uploaded labelled data; continuous monitoring and automatic learning are future work. Validate interventions in a controlled pilot before treating hypothetical uplift or financial outputs as observed results.
