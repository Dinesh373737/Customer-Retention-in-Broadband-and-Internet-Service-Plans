"""Validated response contracts for versioned dataset and analysis APIs.

Data-dependent feature names remain dictionaries. The stable navigation,
versioning, metric and job fields have explicit types in generated OpenAPI.
"""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Payload(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class ColumnProfile(Payload):
    name: str
    inferred_type: str
    missing: int = Field(ge=0)
    unique: int = Field(ge=0)
    sample_values: list[Any]
    identifier: bool
    sensitive: bool
    suggested_role: str


class DatasetProfile(Payload):
    rows: int = Field(ge=0)
    columns_count: int = Field(ge=0)
    missing_cells: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    columns: list[ColumnProfile]
    target_candidates: list[str]
    warnings: list[str]
    preview: list[dict[str, Any]]


class DatasetPayload(Payload):
    id: str
    filename: str
    file_size: int = Field(ge=0)
    fingerprint: str
    configuration: dict[str, Any] | None
    profile: DatasetProfile


class DatasetListing(Payload):
    datasets: list[DatasetPayload]


class ConfigurationPayload(Payload):
    dataset_id: str
    configuration: dict[str, Any]
    fingerprint: str


class AnalysisPayload(Payload):
    analysis_id: str
    dataset_id: str
    model_version: str | None
    mode: Literal["historical", "exploratory"]
    operating_mode: Literal["A", "B", "C"]
    status: Literal["ready", "queued", "running", "completed", "failed"]


class TrainingPayload(AnalysisPayload):
    job_id: str


class MetricPayload(Payload):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    brier: float
    confusion_matrix: list[list[int]]


class ModelPayload(AnalysisPayload):
    preferred_model: str
    metrics: list[MetricPayload]
    validation_metrics: list[MetricPayload]
    models: list[str]
    curves: dict[str, dict[str, Any]]
    metadata: dict[str, Any]
    dataset_fingerprint: str
    configuration_fingerprint: str


class CustomersPayload(AnalysisPayload):
    rows: list[dict[str, Any]]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    has_model: bool


class CustomerPayload(AnalysisPayload):
    customer: dict[str, Any]
    explanation: dict[str, Any] | None
    actionable_features: list[dict[str, Any]]


class BatchPayload(AnalysisPayload):
    rows: list[dict[str, Any]]
    total: int = Field(ge=0)
    warnings: list[str]
    selected_model: str
    columns: list[str]
    explanations: dict[str, dict[str, Any]]
    explanation_limit: int = Field(ge=0, le=10)


class RecommendationsPayload(AnalysisPayload):
    customers: list[dict[str, Any]]
    segments: list[dict[str, Any]]
    catalog: dict[str, Any]
    limitations: list[str]
    evidence_status: str


class EconomicsPayload(AnalysisPayload):
    scenario: dict[str, Any]
    sensitivity: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    assumptions: list[str]


class WhatIfPayload(AnalysisPayload):
    original_probability: float = Field(ge=0, le=1)
    scenario_probability: float = Field(ge=0, le=1)
    changes: dict[str, Any]
    caveats: list[str]
