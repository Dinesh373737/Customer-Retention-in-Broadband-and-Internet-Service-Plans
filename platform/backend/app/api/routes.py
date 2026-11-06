from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from ..core.serialization import json_safe, safe_csv
from ..persistence.store import new_id, utcnow
from ..schemas.requests import DatasetConfiguration, CreateAnalysis, TrainRequest, WhatIfRequest, EconomicsRequest, JobResponse, QuickstartRequest
from ..schemas.responses import (DatasetPayload, DatasetListing, ConfigurationPayload, AnalysisPayload,
    TrainingPayload, ModelPayload, CustomersPayload, CustomerPayload, BatchPayload,
    RecommendationsPayload, EconomicsPayload, WhatIfPayload)
from ..services.data_engine import read_table, profile_dataframe, validate_configuration, frame_fingerprint, target_values, is_sensitive, canonical_label
from ..services.analytics import overview, eda, apply_filters
from ..services.modeling import score_frame, model_report
from ..services.explainability import global_explanation, customer_explanation
from ..services.recommendations import generate_recommendations, protected_feature
from ..services.economics import economic_analysis
from ..services.autoconfig import suggest_configuration, suggest_economics

router = APIRouter(prefix="/api/v1")
SAMPLE_DATASET = Path(__file__).resolve().parents[4] / "data" / "Telco-Customer-Churn.csv"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
RECOMMENDATION_EXPORT_COLUMNS = [
    "customer_reference", "churn_probability", "feature", "business_concept", "business_meaning",
    "segment", "action", "rationale", "evidence", "sample_size", "comparison_size",
    "observed_churn_rate", "overall_churn_rate", "ci_lower", "ci_upper", "p_value", "p_adjusted",
    "objective", "caveats", "estimated_cost", "evidence_status", "risk_drivers",
]


def runtime(request: Request):
    return request.app.state.runtime


def response(data, status=200, schema=None):
    body = json_safe(data)
    if schema is not None:
        body = schema.model_validate(body).model_dump(mode="json")
    return JSONResponse(body, status_code=status)


def get_entity(rt, kind, identifier):
    data = rt.store.get(kind, identifier)
    if data is None:
        raise HTTPException(404, f"Unknown {kind} ID. Choose a saved {kind} from this workspace.")
    return data


def envelope(analysis):
    return {"analysis_id": analysis["id"], "dataset_id": analysis["dataset_id"],
            "model_version": analysis.get("model_version"), "mode": analysis["mode"],
            "operating_mode": analysis.get("operating_mode"), "status": analysis["status"]}


def snapshot(rt, analysis_id):
    analysis = get_entity(rt, "analysis", analysis_id)
    dataset = get_entity(rt, "dataset", analysis["dataset_id"])
    try:
        frame = rt.frame(dataset)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return analysis, dataset, frame


def fitted(rt, analysis, model=None):
    try:
        bundle = rt.bundle(analysis)
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    if model is not None and model not in bundle["models"]:
        raise HTTPException(422, "The selected model does not belong to this analysis version.")
    return bundle


def global_driver_evidence(rt, analysis, model=None):
    if not analysis.get("artifact_path") or analysis.get("status") != "completed":
        return {}
    explanation = global_explanation(fitted(rt, analysis, model), model, include_shap=False)
    result = {}
    for record in explanation["importance"]:
        source = record["source_feature"]
        entry = result.setdefault(source, {"importance": 0., "model": explanation["model"],
                                          "method": record["method"],
                                          "caveat": "Predictive importance is not a verified churn cause or intervention effect."})
        entry["importance"] += float(record["importance"])
    return result


def parsed_filters(value):
    if not value:
        return None
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise HTTPException(422, "Filters must be a JSON object keyed by dataset column.") from error
    if not isinstance(result, dict):
        raise HTTPException(422, "Filters must be a JSON object.")
    return result


def display_frame(frame, configuration):
    columns = [item["column"] for item in configuration.get("features", [])]
    metadata = [name for name in ["row_index", "customer_reference", "churn_probability", "risk_category", "actual_outcome", "actual_churn"] if name in frame]
    result = frame[[column for column in columns if column in frame] + [name for name in metadata if name not in columns]].copy()
    identifier = configuration.get("customer_id")
    if identifier and is_sensitive(identifier) and "customer_reference" in result:
        result["customer_reference"] = result["row_index"].map(lambda index: f"Record-{int(index)+1:06d}")
    return result


def customer_table(rt, analysis, frame, model=None, filters=None, risk=None, search=None):
    configuration = analysis["configuration"]
    if analysis.get("artifact_path") and analysis["status"] == "completed":
        bundle = fitted(rt, analysis, model)
        scored = score_frame(bundle, frame, model)
    else:
        if model:
            raise HTTPException(409, "No fitted model is available in this analysis.")
        scored = frame.copy()
        scored["row_index"] = np.arange(len(scored))
        identifier = configuration.get("customer_id")
        scored["customer_reference"] = frame[identifier].fillna("").astype(str) if identifier else pd.Series([f"Record-{index+1:06d}" for index in range(len(frame))])
        missing = scored["customer_reference"].eq("")
        scored.loc[missing, "customer_reference"] = scored.loc[missing, "row_index"].map(lambda value: f"Record-{int(value)+1:06d}")
        scored["churn_probability"] = np.nan
        scored["risk_category"] = "Unscored"
        if configuration.get("target_column"):
            scored["actual_outcome"] = frame[configuration["target_column"]].astype(str)
            scored["actual_churn"] = target_values(frame, configuration)
    if filters:
        selected = apply_filters(frame, filters)
        scored = scored.loc[selected.index]
    if risk:
        bands = [item.strip().title() for item in risk.split(",") if item.strip()]
        if set(bands) - {"Low", "Medium", "High", "Unscored"}:
            raise HTTPException(422, "Risk filters must be Low, Medium, High or Unscored.")
        scored = scored.loc[scored["risk_category"].isin(bands)]
    if search:
        scored = scored.loc[scored["customer_reference"].astype(str).str.contains(search, case=False, regex=False)]
    return scored


async def read_upload(file, limit=MAX_UPLOAD_BYTES):
    data = await file.read(limit + 1)
    await file.close()
    if len(data) > limit:
        raise HTTPException(413, "Uploads are limited to 25 MiB.")
    if not data:
        raise HTTPException(422, "The uploaded file is empty.")
    return data


@router.get("/health")
def health():
    return {"status": "ok", "service": "Retention Lab 2.0", "api_version": "v1", "scope": "Local historical analysis; no live provider connection"}


@router.get("/datasets", response_model=DatasetListing)
def datasets(rt=Depends(runtime)):
    return response({"datasets": [{key: value for key, value in item.items() if key != "storage_path"} for item in rt.store.list("dataset")]}, schema=DatasetListing)


def store_dataset(rt, contents, filename, sheet_name=None):
    frame = read_table(contents, filename, sheet_name)
    identifier = new_id()
    extension = Path(filename).suffix.lower()
    path = Path("datasets") / f"{identifier}{extension}"
    (rt.home / path).write_bytes(contents)
    profile = profile_dataframe(frame)
    dataset = rt.store.put("dataset", identifier, {"id": identifier, "filename": filename, "file_size": len(contents),
                "fingerprint": frame_fingerprint(frame), "storage_path": str(path), "sheet_name": sheet_name,
                "configuration": None, "profile": profile, "preview": profile.get("preview", []),
                "worksheets": profile.get("worksheets", []), "worksheet": profile.get("worksheet")})
    return {key: value for key, value in dataset.items() if key != "storage_path"}


@router.post("/datasets/upload", status_code=201, response_model=DatasetPayload)
async def upload_dataset(file: Annotated[UploadFile, File()], sheet_name: Annotated[str | None, Form()] = None, rt=Depends(runtime)):
    contents = await read_upload(file)
    filename = (file.filename or "upload.csv").replace("\\", "/").split("/")[-1]
    if len(filename) > 200:
        raise HTTPException(422, "Use a filename shorter than 200 characters.")
    return response(store_dataset(rt, contents, filename, sheet_name), 201, DatasetPayload)


@router.post("/datasets/sample", status_code=201, response_model=DatasetPayload)
def load_sample_dataset(rt=Depends(runtime)):
    """Load the bundled IBM Telco sample so the platform can be tried without a file."""
    if not SAMPLE_DATASET.is_file():
        raise HTTPException(404, "The bundled sample dataset is not available in this installation.")
    existing = [item for item in rt.store.list("dataset") if item.get("filename") == SAMPLE_DATASET.name]
    if existing:
        return response({key: value for key, value in existing[0].items() if key != "storage_path"}, 201, DatasetPayload)
    return response(store_dataset(rt, SAMPLE_DATASET.read_bytes(), SAMPLE_DATASET.name), 201, DatasetPayload)


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetPayload)
def dataset_profile(dataset_id: str, rt=Depends(runtime)):
    dataset = get_entity(rt, "dataset", dataset_id)
    return response({key: value for key, value in dataset.items() if key != "storage_path"}, schema=DatasetPayload)


@router.post("/datasets/{dataset_id}/configuration", response_model=ConfigurationPayload)
def configure_dataset(dataset_id: str, request: DatasetConfiguration, rt=Depends(runtime)):
    dataset = get_entity(rt, "dataset", dataset_id)
    configuration = validate_configuration(rt.frame(dataset), request.model_dump(exclude_none=True))
    rt.store.update("dataset", dataset_id, configuration=configuration)
    return response({"dataset_id": dataset_id, "configuration": configuration, "fingerprint": dataset["fingerprint"]}, schema=ConfigurationPayload)


@router.get("/datasets/{dataset_id}/suggested-configuration")
def suggested_configuration(dataset_id: str, rt=Depends(runtime)):
    """Automatically inferred schema: outcome, churn label, identifier, predictors and business concepts."""
    dataset = get_entity(rt, "dataset", dataset_id)
    return response({"dataset_id": dataset_id, **suggest_configuration(rt.frame(dataset), dataset["profile"])})


@router.post("/datasets/{dataset_id}/quickstart", status_code=202)
def quickstart(dataset_id: str, request: QuickstartRequest | None = None, rt=Depends(runtime)):
    """One call from uploaded file to running analysis: auto-configure, save, then train when labelled."""
    request = request or QuickstartRequest()
    dataset = get_entity(rt, "dataset", dataset_id)
    frame = rt.frame(dataset)
    if request.use_existing and dataset.get("configuration"):
        suggestion = {"configuration": dataset["configuration"], "notes": ["Using the saved schema for this dataset."], "excluded": []}
    else:
        suggestion = suggest_configuration(frame, dataset["profile"])
        rt.store.update("dataset", dataset_id, configuration=suggestion["configuration"])
        dataset = get_entity(rt, "dataset", dataset_id)
    configuration = dataset["configuration"]
    job_id = None
    trainable = bool(configuration.get("target_column") and configuration.get("features")) and request.train
    if trainable:
        y = target_values(frame, configuration)
        trainable = len(frame) >= 60 and int(y.value_counts().min()) >= 12
        if not trainable:
            suggestion["notes"].append("Too few labelled records for reliable model training; opened descriptive analysis instead.")
    if trainable:
        analysis = rt.create_analysis(dataset, trained=True)
        try:
            job = rt.submit(analysis, {"fast": request.fast, "selection_metric": "pr_auc", "random_state": 42})
            job_id = job["id"]
        except ValueError as error:
            rt.store.update("analysis", analysis["id"], status="failed", error=str(error))
            raise HTTPException(429, str(error)) from error
    else:
        analysis = rt.create_analysis(dataset)
    return response({**envelope(analysis), "job_id": job_id, "configuration": configuration,
                     "notes": suggestion["notes"], "excluded": suggestion["excluded"]}, 202)


@router.get("/analyses")
def analyses(rt=Depends(runtime)):
    return response({"analyses": [{key: value for key, value in analysis.items() if key != "artifact_path"} for analysis in rt.store.list("analysis")]})


@router.post("/analyses", status_code=201, response_model=AnalysisPayload)
def create_analysis(request: CreateAnalysis, rt=Depends(runtime)):
    dataset = get_entity(rt, "dataset", request.dataset_id)
    if dataset.get("configuration") is None:
        raise HTTPException(409, "Confirm the dataset schema before creating an analysis.")
    analysis = rt.create_analysis(dataset)
    return response({**envelope(analysis), "id": analysis["id"], "configuration": analysis["configuration"], "has_model": False}, 201, AnalysisPayload)


@router.post("/analyses/train", status_code=202, response_model=TrainingPayload)
def train_analysis(request: TrainRequest, rt=Depends(runtime)):
    dataset = get_entity(rt, "dataset", request.dataset_id)
    configuration = dataset.get("configuration")
    if not configuration or not configuration.get("target_column"):
        raise HTTPException(409, "Training requires a user-confirmed binary historical churn target. Exploratory-only datasets cannot invent labels.")
    if not configuration.get("features"):
        raise HTTPException(422, "Select at least one valid predictor before training.")
    analysis = rt.create_analysis(dataset, trained=True)
    try:
        job = rt.submit(analysis, request.model_dump(exclude={"dataset_id"}))
    except ValueError as error:
        rt.store.update("analysis", analysis["id"], status="failed", error=str(error))
        raise HTTPException(429, str(error)) from error
    return response({**envelope(analysis), "job_id": job["id"], "status": job["status"]}, 202, TrainingPayload)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str, rt=Depends(runtime)):
    return get_entity(rt, "job", job_id)


@router.get("/analyses/{analysis_id}/overview", response_model=AnalysisPayload)
def analysis_overview(analysis_id: str, filters: str | None = None, rt=Depends(runtime)):
    analysis, dataset, frame = snapshot(rt, analysis_id)
    return response({**envelope(analysis), **overview(frame, analysis["configuration"], parsed_filters(filters)),
                     "configuration": analysis["configuration"], "filename": dataset["filename"],
                     "profile": dataset["profile"], "has_model": bool(analysis.get("artifact_path")), "created_at": analysis["created_at"]})


@router.get("/analyses/{analysis_id}/eda", response_model=AnalysisPayload)
def analysis_eda(analysis_id: str, numeric: str | None = None, categorical: str | None = None, second: str | None = None,
                 filters: str | None = None, rt=Depends(runtime)):
    analysis, _, frame = snapshot(rt, analysis_id)
    return response({**envelope(analysis), **eda(frame, analysis["configuration"], numeric, categorical, second, parsed_filters(filters))})


@router.get("/analyses/{analysis_id}/models", response_model=ModelPayload)
def analysis_models(analysis_id: str, rt=Depends(runtime)):
    analysis = get_entity(rt, "analysis", analysis_id)
    bundle = fitted(rt, analysis)
    return response({**envelope(analysis), **model_report(bundle)}, schema=ModelPayload)


@router.get("/analyses/{analysis_id}/explanations", response_model=AnalysisPayload)
def analysis_explanations(analysis_id: str, model: str | None = None, include_shap: bool = False, rt=Depends(runtime)):
    analysis = get_entity(rt, "analysis", analysis_id)
    return response({**envelope(analysis), **global_explanation(fitted(rt, analysis, model), model, include_shap)})


@router.get("/analyses/{analysis_id}/customers", response_model=CustomersPayload)
def customers(analysis_id: str, model: str | None = None, risk: str | None = None, search: str | None = None,
              filters: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500),
              sort: str = Query("desc", pattern="^(asc|desc)$"), rt=Depends(runtime)):
    analysis, _, frame = snapshot(rt, analysis_id)
    table = customer_table(rt, analysis, frame, model, parsed_filters(filters), risk, search)
    table = table.sort_values("churn_probability", ascending=sort == "asc", na_position="last", kind="stable")
    shown = display_frame(table.iloc[offset:offset+limit], analysis["configuration"])
    return response({**envelope(analysis), "rows": shown.to_dict("records"), "total": len(table), "offset": offset,
                     "limit": limit, "has_model": bool(analysis.get("artifact_path")), "selected_model": model or analysis.get("preferred_model")}, schema=CustomersPayload)


@router.get("/analyses/{analysis_id}/customers/{row_index}", response_model=CustomerPayload)
def customer_detail(analysis_id: str, row_index: int, model: str | None = None, rt=Depends(runtime)):
    analysis, _, frame = snapshot(rt, analysis_id)
    if not 0 <= row_index < len(frame):
        raise HTTPException(404, "Customer row is outside this dataset version.")
    table = customer_table(rt, analysis, frame, model)
    selected = table.loc[table["row_index"] == row_index]
    explanation = customer_explanation(fitted(rt, analysis, model), frame.iloc[[row_index]], model) if analysis.get("artifact_path") else None
    return response({**envelope(analysis), "customer": display_frame(selected, analysis["configuration"]).iloc[0].to_dict(),
                     "explanation": explanation, "actionable_features": actionable_features(frame, analysis["configuration"])}, schema=CustomerPayload)


def actionable_features(frame, config):
    kinds = {item["column"]: item["kind"] for item in config.get("features", [])}
    output = []
    for mapping in config.get("semantics", []):
        column = mapping.get("column")
        if column not in kinds or column not in frame or not mapping.get("confirmed") or not mapping.get("actionable") or mapping.get("protected") or protected_feature(column) or mapping.get("concept") == "lifecycle":
            continue
        feature = {**mapping, "kind": kinds[column]}
        if kinds[column] == "numeric":
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            feature.update(minimum=values.min() if len(values) else None, maximum=values.max() if len(values) else None)
        else:
            feature["categories"] = frame[column].dropna().map(canonical_label).drop_duplicates().head(100).tolist()
        output.append(feature)
    return output


@router.get("/analyses/{analysis_id}/recommendations", response_model=RecommendationsPayload)
def recommendations(analysis_id: str, row_index: int | None = Query(None, ge=0), model: str | None = None,
                    filters: str | None = None, rt=Depends(runtime)):
    analysis, _, frame = snapshot(rt, analysis_id)
    if model:
        fitted(rt, analysis, model)
    if row_index is not None and row_index >= len(frame):
        raise HTTPException(404, "Customer row is outside this dataset version.")
    scored = customer_table(rt, analysis, frame, model) if analysis.get("artifact_path") else None
    drivers = None
    if row_index is not None and scored is not None:
        drivers = customer_explanation(fitted(rt, analysis, model), frame.iloc[[row_index]], model).get("drivers")
    conditions = parsed_filters(filters)
    if conditions:
        frame = apply_filters(frame, conditions)
        if scored is not None:
            scored = scored.loc[frame.index]
    global_drivers = global_driver_evidence(rt, analysis, model)
    result = generate_recommendations(frame, analysis["configuration"], scored, row_index, drivers, global_drivers) if len(frame) else {
        "customers": [], "segments": [], "catalog": {}, "limitations": ["No customer records match these filters."], "evidence_status": "insufficient"}
    rt.store.put("recommendations", new_id(), {"analysis_id": analysis_id, "dataset_id": analysis["dataset_id"], "filters": conditions, "row_index": row_index, "result": result})
    return response({**envelope(analysis), **result}, schema=RecommendationsPayload)


@router.post("/analyses/{analysis_id}/batch-score", response_model=BatchPayload)
async def batch_score(analysis_id: str, file: Annotated[UploadFile, File()], model: Annotated[str | None, Form()] = None,
                       sheet_name: Annotated[str | None, Form()] = None, rt=Depends(runtime)):
    analysis = get_entity(rt, "analysis", analysis_id)
    bundle = fitted(rt, analysis, model)
    contents = await read_upload(file)
    incoming = read_table(contents, file.filename or "scoring.csv", sheet_name)
    scored = score_frame(bundle, incoming, model)
    result = display_frame(scored, analysis["configuration"])
    # Explain a bounded actual uploaded sample without persisting a second raw
    # customer table or pretending the saved historical rows are the new rows.
    explanation_limit = min(10, len(incoming))
    explanations = {str(index): customer_explanation(bundle, incoming.iloc[[index]], model)
                    for index in range(explanation_limit)}
    return response({**envelope(analysis), "operating_mode": "B", "rows": result.to_dict("records"), "total": len(result),
                     "warnings": scored.attrs.get("warnings", []), "selected_model": model or bundle["preferred_model"],
                     "explanations": explanations, "explanation_limit": explanation_limit,
                     "explanation_note": "Actual selected-model sensitivities are provided for the first 10 uploaded records at most; no intervention effect is estimated.",
                     "columns": result.columns.tolist(), "limitation": "Scores are model estimates; no historical performance is computed for unlabelled records."}, schema=BatchPayload)


@router.post("/analyses/{analysis_id}/what-if", response_model=WhatIfPayload)
def what_if(analysis_id: str, request: WhatIfRequest, rt=Depends(runtime)):
    analysis, _, frame = snapshot(rt, analysis_id)
    bundle = fitted(rt, analysis, request.model)
    if request.row_index >= len(frame):
        raise HTTPException(404, "Customer row is outside this dataset version.")
    allowed = {feature["column"]: feature for feature in actionable_features(frame, analysis["configuration"])}
    if set(request.changes) - set(allowed):
        raise HTTPException(422, "What-if changes are limited to confirmed, actionable, non-protected business features. Identifiers, outcomes and customer age/lifecycle cannot be changed.")
    original = frame.iloc[[request.row_index]].copy()
    changed = original.copy()
    for column, value in request.changes.items():
        feature = allowed[column]
        if value is None or isinstance(value, (dict, list)):
            raise HTTPException(422, "Scenario feature values must be non-missing scalar inputs.")
        if feature["kind"] == "numeric":
            try:
                number = float(value)
            except (ValueError, TypeError) as error:
                raise HTTPException(422, f"'{column}' requires a numeric scenario input.") from error
            if not np.isfinite(number) or number < feature["minimum"] or number > feature["maximum"]:
                raise HTTPException(422, f"'{column}' must stay within the observed supported range {feature['minimum']} to {feature['maximum']}.")
            changed[column] = number
        elif canonical_label(value) not in feature.get("categories", []):
            raise HTTPException(422, f"'{column}' requires an observed valid category.")
        else:
            changed[column] = canonical_label(value)
    original_score = float(score_frame(bundle, original, request.model)["churn_probability"].iloc[0])
    scenario_score = float(score_frame(bundle, changed, request.model)["churn_probability"].iloc[0])
    return response({**envelope(analysis), "original_probability": original_score, "scenario_probability": scenario_score,
                     "changes": request.changes, "difference": scenario_score - original_score,
                     "caveats": ["Model sensitivity scenario only; changing an input does not prove a real intervention will reduce churn.",
                                 "Other account inputs remain fixed; real-world feasibility needs analyst review."]}, schema=WhatIfPayload)


@router.get("/analyses/{analysis_id}/economics/defaults")
def economics_defaults(analysis_id: str, model: str | None = None, rt=Depends(runtime)):
    """Campaign assumptions derived from the dataset (risk band size, pricing columns) for a zero-input start."""
    analysis, _, frame = snapshot(rt, analysis_id)
    if model:
        fitted(rt, analysis, model)
    scored = customer_table(rt, analysis, frame, model) if analysis.get("artifact_path") and analysis["status"] == "completed" else None
    return response({**envelope(analysis), **suggest_economics(frame, analysis["configuration"], scored)})


@router.post("/analyses/{analysis_id}/economics", response_model=EconomicsPayload)
def economics(analysis_id: str, request: EconomicsRequest, filters: str | None = None,
              model: str | None = None, rt=Depends(runtime)):
    analysis, _, frame = snapshot(rt, analysis_id)
    if request.model and model and request.model != model:
        raise HTTPException(422, "Choose the same economic-ranking model in the request body and query, or supply it only once.")
    model = request.model or model
    if model:
        fitted(rt, analysis, model)
    conditions = parsed_filters(filters)
    scored = customer_table(rt, analysis, frame, model, conditions) if analysis.get("artifact_path") else None
    selected_model = model or analysis.get("preferred_model")
    result = economic_analysis(request.model_dump(exclude={"model"}), scored)
    result["selected_model"] = selected_model
    result["scenario"]["selected_model"] = selected_model
    result["population"] = {"filters": conditions, "available_customers": len(apply_filters(frame, conditions)),
                            "selected_model": selected_model}
    rt.store.put("economics", new_id(), {"analysis_id": analysis_id, "dataset_id": analysis["dataset_id"], "result": result})
    return response({**envelope(analysis), **result}, schema=EconomicsPayload)


@router.get("/analyses/{analysis_id}/export")
def export_analysis(analysis_id: str, kind: str = "report", model: str | None = None, risk: str | None = None,
                     search: str | None = None, filters: str | None = None, rt=Depends(runtime)):
    analysis, dataset, frame = snapshot(rt, analysis_id)
    if model:
        fitted(rt, analysis, model)
    headers = {"Content-Disposition": f'attachment; filename="retention-{analysis_id[:8]}-{kind}.csv"'}
    if kind == "customers":
        table = display_frame(customer_table(rt, analysis, frame, model, parsed_filters(filters), risk, search), analysis["configuration"])
        return Response(safe_csv(table), media_type="text/csv", headers=headers)
    if kind == "metrics":
        table = pd.DataFrame(model_report(fitted(rt, analysis, model))["metrics"])
        return Response(safe_csv(table), media_type="text/csv", headers=headers)
    if kind == "recommendations":
        scored = customer_table(rt, analysis, frame, model) if analysis.get("artifact_path") else None
        selected = apply_filters(frame, parsed_filters(filters))
        if scored is not None:
            scored = scored.loc[selected.index]
        result = generate_recommendations(selected, analysis["configuration"], scored,
                                          global_drivers=global_driver_evidence(rt, analysis, model))
        table = pd.DataFrame(result["segments"], columns=RECOMMENDATION_EXPORT_COLUMNS)
        return Response(safe_csv(table), media_type="text/csv", headers=headers)
    if kind != "report":
        raise HTTPException(422, "Export kind must be report, customers, metrics or recommendations.")
    scenarios = [record for record in rt.store.list("economics") if record["analysis_id"] == analysis_id]
    selected = apply_filters(frame, parsed_filters(filters))
    scored = customer_table(rt, analysis, frame, model) if analysis.get("artifact_path") else None
    if scored is not None:
        scored = scored.loc[selected.index]
    report = {**envelope(analysis), "title": "Retention Lab 2.0 historical analysis report", "generated_at": utcnow(),
              "dataset": {"id": dataset["id"], "filename": dataset["filename"], "fingerprint": dataset["fingerprint"], "profile": dataset["profile"]},
              "configuration": analysis["configuration"], "overview": overview(frame, analysis["configuration"], parsed_filters(filters)),
              "models": model_report(fitted(rt, analysis, model)) if analysis.get("artifact_path") else None,
              "recommendations": generate_recommendations(selected, analysis["configuration"], scored,
                    global_drivers=global_driver_evidence(rt, analysis, model)),
              "population": {"filters": parsed_filters(filters), "selected_rows": len(selected), "dataset_rows": len(frame)},
              "hypothetical_economics": scenarios[0]["result"] if scenarios else None,
              "limitations": ["Historical associations and model explanations are not verified causes.", "Economic outcomes are hypothetical assumptions.", "No live customer contact or outcome monitoring is connected."]}
    return Response(json.dumps(json_safe(report), indent=2, allow_nan=False), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="retention-{analysis_id[:8]}-report.json"'})
