# Shared implementation contract

The legacy application, dataset, docs, tests and artifacts remain unchanged. New code stays under platform/. Python app package lives under platform/backend/app and is launched from platform/backend. Use relative imports. No platform/__init__.py (avoid shadowing Python stdlib).

## Schema

Dataset configuration JSON: `target_column: str|null`, `positive_label: str|null`, `customer_id: str|null`, `excluded_columns: list[str]`, `features: list[{column: str, kind: numeric|categorical|ordinal|datetime, ordinal_order?: list[str], date_reference?: str}]`, `business_domain: telecommunications|banking|ecommerce|subscription|generic`, `prediction_horizon: str|null`, `semantics: list[{column, concept: pricing|lifecycle|support|engagement|contract|satisfaction|loyalty, meaning: str, confirmed: bool, actionable: bool, protected: bool, estimated_cost?: float}]`, `leakage_approved: dict[str,str]`.

Column names always refer to the uploaded schema, never IBM-specific constants. Confirmed semantics require a meaning; sensitive/protected/immutable features cannot drive interventions or what-if changes. Inference suggestions never silently become confirmed mappings. Labels canonicalize to strings; positive_label must select one of exactly two nonmissing canonical labels for historical training.

## Data/ML service APIs (agent data_ml owns)

- `app.services.data_engine.read_table(contents: bytes, filename: str, sheet_name: str|None=None) -> pd.DataFrame` validates headers, file limits, formats, .xlsx sheet selection; no execution.
- `profile_dataframe(df) -> dict` keys rows, columns_count, missing_cells, duplicate_rows, columns (each name, inferred_type, missing, unique, sample_values, identifier, sensitive, leakage_warning, suggested_role), target_candidates, warnings.
- `validate_configuration(df, config:dict) -> dict` returns normalized config with warnings/decisions. Raises ValueError on invalid target/features/proxies or unsupported temporal/group usage. Initial analysis may have no target.
- `frame_fingerprint(df) -> str`; `configuration_fingerprint(config) -> str`.
- `app.services.analytics.overview(df, config, filters=None) -> dict`; `eda(df, config, numeric=None, categorical=None, second=None, filters=None) -> dict`. APIs below may adapt payload shape; communicate final columns to frontend.
- `app.services.modeling.train_dataset(df, config, settings=None, progress=None) -> bundle:dict`; settings fast, random_state, selection_metric. callback progress(percent:int,message:str). Strict independent stratified training/validation/test; training-only tuning, calibration and preprocessing; equal hybrid. Generic exact inference schema in bundle.
- `score_frame(bundle, df, model_name=None) -> pd.DataFrame` returns input columns plus `row_index`, `customer_reference`, `churn_probability`, `risk_category`; actual labels clearly identified, schema validated, batch row cardinality preserved.
- `save_bundle(bundle,path)` / `load_bundle(path)` / `model_report(bundle)->dict` JSON safe metrics, ROC/PR/calibration/confusion/threshold curves (validation threshold); metadata, selected model, version/fingerprints.
- Bundle required fields: `configuration`, `dataset_fingerprint`, `configuration_fingerprint`, `feature_columns`, `models`, `preferred_model`, `metrics` (list of row dict), `validation_metrics`, `metadata`, `train_indices`, `validation_indices`, `test_indices`, `y_test`, `probabilities`, `training_baselines`, `background`. Fitted pipelines/model wrappers serializable. Generated model_version assigned by API as analysis_id.
- `app.services.explainability.global_explanation(bundle, model_name=None, include_shap=False)->dict` and `customer_explanation(bundle, row:pd.DataFrame, model_name=None)->dict` (probability, risk_category, drivers [{feature,value,reference,contribution}], method, caveats).

## Versioned API (agent backend owns)

Base `/api/v1`. Errors JSON `{detail: string|object}`. Dates ISO. JSON non-finite numbers become null. Every analysis response identifies analysis_id, dataset_id and model_version. Immutable analysis snapshots isolate changes to dataset configuration. Frontend uses TanStack Query keys including analysis ID.

- GET /health
- POST /datasets/upload multipart `file`, optional `sheet_name`; response dataset {id, filename, file_size, profile, configuration:null, fingerprint}.
- GET /datasets -> {datasets:[...]}; GET /datasets/{id}/profile -> dataset.
- POST /datasets/{id}/configuration config JSON -> {dataset_id,configuration,...}
- POST /analyses {dataset_id} -> {analysis_id,dataset_id,mode,status}; permits descriptive analysis without models.
- POST /analyses/train {dataset_id,fast:false,random_state:42,selection_metric:'pr_auc'} -> {analysis_id,job_id,status}
- GET /jobs/{id} -> {id,analysis_id,status:queued|running|completed|failed,progress,message,error}
- GET /analyses -> {analyses:[...]}
- GET /analyses/{id}/overview -> {analysis_id,dataset_id,model_version,...overview}
- GET /analyses/{id}/eda?numeric=&categorical=&second= -> {...EDA}
- GET /analyses/{id}/models -> {...model_report}; honest 409 when no fitted artifact.
- GET /analyses/{id}/explanations?model=&include_shap=false -> global result
- GET /analyses/{id}/customers?model=&risk=&search=&offset=0&limit=50&sort=desc -> {rows,total,...}; includes actual historical outcome separately.
- GET /analyses/{id}/customers/{row_index}?model= -> {customer,explanation,...}
- GET /analyses/{id}/recommendations?row_index=&model= -> {customers:[],segments:[],catalog:[],limitations:[],...}
- POST /analyses/{id}/batch-score multipart file, optional model -> {rows,total,warnings,...}; no target required.
- POST /analyses/{id}/what-if {row_index,changes,model?} -> {original_probability,scenario_probability,changes,caveats,...}
- POST /analyses/{id}/economics {budget,targeted,intervention_cost,incremental_impact,contribution,fixed_cost,period,currency} -> {scenario,sensitivity,ranking,assumptions,...}; increment is unconditional causal uplift assumption, never multiplied by predicted churn.
- GET /analyses/{id}/export?kind=report|customers|metrics|recommendations -> download JSON/CSV; no fabricated outputs in exploratory mode. `kind=customers&risk=&search=` honors current customer filters.

Frontend and backend agents should coordinate exact response details directly. API can add query filters JSON string and endpoints where helpful. It must validate IDs and selected model names, return useful not-ready states, persist metadata in SQLite, store new files/models only in platform/runtime/, and expose genuine background jobs without blocking training requests.
