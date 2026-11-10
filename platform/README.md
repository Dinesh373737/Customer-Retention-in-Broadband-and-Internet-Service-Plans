# Retention Lab 2.0

A local, cross-industry customer retention platform using **React / TypeScript / Vite** and **FastAPI / SQLite / Scikit-learn / XGBoost**. It coexists with the preserved IBM Telco Streamlit case study in the repository root. The root README and launcher point to React by default; legacy source, datasets, tests and analysis artifacts are preserved.

## Install and run on Windows

Prerequisites: Python **3.11 or 3.12**, Node.js **22+**, and pnpm **11**. Node's standard installer includes npm; if needed, install pnpm with `npm install -g pnpm@11`.

From the repository root in PowerShell:

```powershell
python -m venv platform\.venv
.\platform\.venv\Scripts\python.exe -m pip install -r platform\backend\requirements.txt
# Optional real SHAP summaries; fitted-model importances/sensitivities also work without it:
.\platform\.venv\Scripts\python.exe -m pip install -r platform\backend\requirements-optional.txt

Set-Location platform\frontend
pnpm install --frozen-lockfile
pnpm build
Set-Location ..\..

.\platform\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir platform/backend --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The backend serves the built React application. API documentation is at **http://127.0.0.1:8000/docs**. Installing packages needs internet; subsequent analysis stays local. `platform\launch.ps1 -Install` automates first setup; `platform\launch.ps1` launches a prepared application. PowerShell execution policies may require using the explicit commands above.

For frontend development, leave the API running in one terminal, then run `pnpm dev` from `platform/frontend` in another. Open **http://127.0.0.1:5173**; Vite proxies `/api` to port 8000. Restart the backend after rebuilding the frontend if its asset directory was absent when the server started.

On macOS/Linux use `platform/.venv/bin/python` instead of the Windows Python path. No Docker, Redis, paid service, external LLM or cloud deployment is needed.

### Original IBM case study

Use the [original IBM instructions](../README.md#original-ibm-streamlit-case-study) and its separate environment. The root launcher also accepts `.\launch.ps1 -Legacy`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

The new Workspace screen links to that independent application at port **8501**. The generic platform does not import the legacy feature allowlist, reuse the IBM artifact for other datasets, or overwrite root artifacts. Root tests passed before and after the upgrade; see [verification](docs/verification.md).

## Workflow and supported modes

### Automatic setup (default)

Drop a CSV/`.xlsx` on **Home** (or click **Use IBM Telco sample**) and nothing else is required. The backend:

- detects the churn outcome (`Churn`, `Exited`, `Attrition`, …), its positive label (`Yes`, `1`, `True`, `Churned`, otherwise the minority class) and the customer ID (preferring business keys over row numbers);
- selects predictors with inferred types, and leaves out identifiers, names/contact fields, possible post-outcome columns, constants and free text. Every suggestion still passes the same validator, so target proxies are dropped with the reason shown in **Data setup**;
- infers the business domain and maps columns to business concepts from their names (pricing, contract, support, engagement, …) so recommendations and what-if scenarios work immediately. Protected attributes are never actionable;
- trains the four models in quick mode, then fills Insights, Models, Explainability, Customers and the Retention plan;
- seeds the campaign economics from the data (High-risk customer count, median monthly charge) and calculates the scenario on page load.

Explainability and the Retention plan pick customers through a searchable, risk-ordered picker (defaulting to the highest-risk customer) instead of row indexes. **Data setup → Adjust manually** opens the original three-step editor, pre-filled with the automatic choices, for anything that was misread.

### Manual steps (optional)

1. **Upload:** upload UTF-8 CSV or `.xlsx`. For a workbook with multiple sheets, choose a worksheet explicitly; errors list available names. Inspect actual row/column counts, types, missingness, duplicates and the preview. Public uploads are limited to 25 MiB, 100,000 rows and 200 columns.
2. **Confirm the schema:** choose an outcome and its positive label, optional ID, excluded columns, feature types, domain and known prediction timing. Inferred suggestions are editable. Targets and IDs cannot become predictors; names/contact information and reserved output fields must be excluded. Constants, invalid values and target proxies produce useful errors. At most 100 predictors are supported, with categorical/dense-memory safeguards.
3. **Confirm business meaning:** optionally map a feature to pricing, lifecycle, support, engagement, contract, satisfaction or loyalty. Enter its meaning and explicitly confirm it. Unknown features may predict but cannot invent retention actions. Protected/immutable attributes cannot drive recommendations or what-if edits.
4. **Explore:** analytics, category counts and confidence intervals, numeric distributions, box summaries, correlations, relationships and statistical associations adapt to the configured schema. Filters affect descriptive/customer outputs, never held-out metrics.
5. **Train:** Model Lab creates a persisted background job. Actual progress/error messages appear while requests remain responsive. Separate models are trained for this dataset version. Inspect all four models, curves, calibration and real held-out results.
6. **Investigate:** choose a customer, inspect actual historical labels separately from predicted risk, compare global and individual explanations, or upload a compatible unlabelled scoring batch. Individual effects are model sensitivities, not causes. CSV exports preserve the selected view.
7. **Review strategy:** inspect evidence-qualified versus insufficient-evidence suggestions. Configure an editable hypothetical campaign budget, cost, contribution period and unconditional incremental retention assumption. Sensitivity and budget ranking use these declared assumptions.
8. **Export:** download customer CSV, metrics, recommendations or a JSON report with the exact dataset/model version, methodology, current observations and saved scenario assumptions.

| Mode | Requirements | Available behavior |
|---|---|---|
| A: historical labelled | A meaningful binary outcome and explicit mapping | Descriptive analysis, training, held-out evaluation, predictions and observational recommendations |
| B: unlabelled scoring | A selected fitted analysis with compatible saved feature schema | Actual probabilities and available explanations; no invented performance or labels |
| C: exploratory only | No labels and no matching model | Quality and descriptive analysis; prediction/evaluation endpoints explicitly report unavailable |

Missing labels, more than two target classes, corrupt files, unknown model/version IDs, incompatible scoring schemas, impossible ordinal values and unsupported future dates are rejected. Batch scoring preserves row count and reports unseen categories; saved encoders handle supported nominal categories. Dataset fingerprints and immutable configurations prevent unrelated analyses from sharing models or stale results.

## Modeling and scientific limits

Logistic Regression, Random Forest and XGBoost are fitted with training-only pipelines; the fourth predictor is a genuine equal-weight mean of their selected raw/calibrated probabilities. Reproducible independent-customer stratified **60/20/20** partitions precede imputation, scaling, encoding and tuning. Small training-only cross-validation grids tune trees. Sigmoid calibration uses training out-of-fold predictions and is selected using validation Brier score. Model selection uses the chosen validation metric; test metrics never choose weights, preprocessing or the preferred model. The test classification threshold is 0.5; threshold trade-offs use validation predictions.

OOF calibration uses hyperparameters selected within the same training cohort rather than nested tuning; validation and final test cohorts remain independent. Exact duplicate historical rows are removed and reported. Conflicting repeated IDs, longitudinal/time-stream training, temporal/group splits and multiclass outcomes are explicitly unsupported in this release. Aggregate longitudinal histories to one legitimate pre-outcome customer snapshot; do not hide repeated customers by dropping IDs.

Numeric imputation/scaling, nominal encoding with infrequent-category grouping, explicit ordinal orders and fixed-reference elapsed-day date features are saved with each model. Date reference choices must be meaningful and known at prediction time. General raw text, event streams, images and causal intervention datasets require different workflows.

Global explanations show raw-estimator coefficients/odds ratios or tree importances. Optional supported SHAP displays its actual estimator and output scale before probability calibration. Individual perturbations explain the **selected calibrated model or ensemble**, comparing one feature against a training reference; effects are non-additive and may describe unusual combinations. These are neither verified causes nor intervention-effect estimates. Statistical significance alone does not establish business value.

Recommendation rules require confirmed meanings, business context, counts, Wilson uncertainty and adjusted association evidence. Weak signals become requests to investigate, not assertions. Technical-support subscription does not establish a service failure. Reported complaint flags only establish a recorded complaint, not why someone left. No discounts or contacts are sent.

The economic calculator uses **N × assumed unconditional incremental impact**, not churn probability multiplied by uplift. Contribution is margin after service costs over the stated period. Costs and impact are hypothetical; budget limits target count after fixed costs. Equal-value ranking uses predicted risk only as a tie-breaker and does not claim personalized treatment effects. No actual campaign profit is observed.

## Actual datasets and benchmark reproduction

The three requested non-IBM datasets were downloaded from their named sources and executed, not replaced with fixtures. [Benchmark results](benchmarks/results.md) contain actual rows, selected predictors, selected model, test ROC-AUC/PR-AUC/F1 and recommendation status. [Explicit configurations](benchmarks/configurations.py) document exclusions and semantics; [source metadata](benchmarks/sources.json) records hashes and license information.

```powershell
.\platform\.venv\Scripts\python.exe platform\scripts\download_benchmarks.py
# With the API running:
.\platform\.venv\Scripts\python.exe platform\scripts\run_benchmarks.py
```

The benchmark runner performs real multipart upload, schema confirmation, asynchronous training/progress, analytics, explanation, customer selection, unlabelled scoring, schema/model errors, restricted what-if, economics and exports. It creates new local dataset/model versions and writes results from actual responses. `--dataset iranian|banking|ecommerce|ibm` selects a single source; `--fast` explicitly uses a smaller training search. Raw external data, model binaries and detailed local per-row reports are excluded from Git.

The e-commerce workbook has `Data Dict` and `E Comm` sheets; the benchmark explicitly selects `E Comm`. UCI charge codes contain an undocumented `10` in addition to documented `0–9`; the benchmark treats this as an unordered feature and does not confirm its pricing recommendation mapping. Iranian `Status` is excluded after timing review. Bank/e-commerce prediction horizons are unknown. See [dataset guide](docs/datasets.md) for source attribution, license restrictions and manual download fallback.

## Tests

```powershell
.\platform\.venv\Scripts\python.exe -m pytest platform\backend\tests -q -p no:asyncio
Set-Location platform\frontend
pnpm test
pnpm build
Set-Location ..\..
# Independently verify the preserved IBM system:
.\.venv\Scripts\python.exe -m pytest tests -q -p no:asyncio
.\.venv\Scripts\python.exe -m scripts.smoke_dashboard
```

Synthetic fixtures in unit/frontend tests are explicitly test data; they are not the external benchmarks. End-to-end benchmark results, test totals and browser verification are documented separately. Codex's Windows sandbox may block local sockets or temporary artifact renames; run local servers/tests with normal local access. Do not disable system security protections.

## Structure

```text
platform/
  backend/app/
    api/                   Versioned routes and export endpoints
    core/                  Runtime, model caches and serialization
    schemas/               Typed request/response schemas
    persistence/           SQLite metadata and restart recovery
    workers/               Local background-job namespace
    services/
      data_engine.py       Uploads, editable schema, types and leakage checks
      generic_pipeline.py  Dataset-specific preprocessing and ensemble wrappers
      analytics.py         Dynamic EDA/statistical analysis
      modeling.py          Training, calibration, evaluation and persistence
      explainability.py    Global/individual explanations and optional SHAP
      recommendations*     Configurable evidence rules and industry catalogue
      economics.py         Incremental campaign/budget scenarios
  backend/tests/           Unit, integration and actual-model API tests
  frontend/src/            React pages, UI/chart components, schema wizard, API client
  benchmarks/              Executed summaries, source metadata, explicit mappings
  scripts/                 Download, HTTP benchmark and live-verification utilities
  runtime/                 Ignored local SQLite database, uploads and fitted artifacts
  docs/                    API, architecture, schema guide, sources and verification
  launch.ps1               Convenient local launcher
```

The backend defaults to `platform/runtime/`; `RETENTION_PLATFORM_HOME` can set another dedicated local storage directory. This is a single-user local prototype with one training worker, not a multi-tenant production service. No user-supplied model binaries are loaded. The original repository root remains independently runnable.
