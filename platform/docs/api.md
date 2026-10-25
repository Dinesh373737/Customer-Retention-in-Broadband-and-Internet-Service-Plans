# Versioned local API

Base: `http://127.0.0.1:8000/api/v1`. Interactive typed documentation: `/docs`; machine-readable schemas: `/openapi.json`. Validation errors use a JSON `detail` message or Pydantic field list. IDs are opaque local identifiers, not file paths.

| Method | Path | Purpose |
|---|---|---|
| GET | /health | Actual service health and scope |
| POST | /datasets/upload | Multipart file plus optional sheet_name; inspect real CSV/XLSX |
| GET | /datasets | Saved datasets |
| GET | /datasets/{id}/profile | Quality, schema suggestions, preview, saved configuration |
| POST | /datasets/{id}/configuration | Explicit target/positive class/features/semantics |
| POST | /analyses | Immutable descriptive/historical analysis snapshot |
| POST | /analyses/train | Queue new dataset-specific training; returns analysis_id/job_id with HTTP 202 |
| GET | /jobs/{id} | Real queued/running/completed/failed status, progress, diagnostic |
| GET | /analyses | Saved versions with dataset identity |
| GET | /analyses/{id}/overview | Computed KPIs/findings; optional filters JSON |
| GET | /analyses/{id}/eda | Dynamic feature summaries, histograms, relationships and tests |
| GET | /analyses/{id}/models | Actual four-model results/curves/calibration; HTTP 409 if not fitted |
| GET | /analyses/{id}/explanations | Selected model importance; optional include_shap=true |
| GET | /analyses/{id}/customers | Search/risk/filter/sort/pagination; actual labels separate from predictions |
| GET | /analyses/{id}/customers/{row} | Correct-version details, individual explanation and supported actionable fields |
| GET | /analyses/{id}/recommendations | Confirmed semantics, segment/customer evidence and catalogue |
| POST | /analyses/{id}/batch-score | Multipart compatible unlabelled file; same saved pipelines; limited individual explanations |
| POST | /analyses/{id}/what-if | Row index, allowed changes and optional model; constrained sensitivity |
| POST | /analyses/{id}/economics | Hypothetical budget/cost/contribution/unconditional incremental impact |
| GET | /analyses/{id}/export | kind=report/customers/metrics/recommendations; JSON or valid CSV |

`model` selects only a member of the fitted analysis. `filters` is a JSON object: category lists (e.g. `{ "Plan": ["Monthly"] }`) or numeric min/max/include_missing objects. Customer exports preserve the selected model, risk, search and business filters. Reports describe their exact snapshot; model metrics are not recalculated on a filtered test subset.

Historical training is asynchronous; poll the returned job endpoint. A failed job retains its error and never gets fabricated metrics. Exploratory mode returns descriptive results and explicitly unavailable model responses. No API accepts uploaded joblib/Python models. Files and artifacts stay in dedicated local storage and are never executed as uploaded code.
