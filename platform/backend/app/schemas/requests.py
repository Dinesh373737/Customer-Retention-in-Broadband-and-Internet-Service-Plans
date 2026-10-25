from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FeatureMapping(StrictModel):
    column: str = Field(min_length=1, max_length=200)
    kind: Literal["numeric", "categorical", "ordinal", "datetime"]
    ordinal_order: list[str] | None = None
    date_reference: str | None = None


class SemanticMapping(StrictModel):
    column: str
    concept: Literal["pricing", "lifecycle", "support", "engagement", "contract", "satisfaction", "loyalty"]
    meaning: str = Field(default="", max_length=1500)
    confirmed: bool = False
    actionable: bool = False
    protected: bool = False
    estimated_cost: float | None = Field(default=None, ge=0, le=1e12)


class DatasetConfiguration(StrictModel):
    target_column: str | None = None
    positive_label: str | int | float | bool | None = None
    customer_id: str | None = None
    excluded_columns: list[str] = Field(default_factory=list)
    features: list[FeatureMapping] = Field(default_factory=list)
    business_domain: Literal["telecommunications", "banking", "ecommerce", "subscription", "generic"] = "generic"
    prediction_horizon: str | None = Field(default=None, max_length=1000)
    semantics: list[SemanticMapping] = Field(default_factory=list)
    leakage_approved: dict[str, str] = Field(default_factory=dict)
    split_strategy: Literal["stratified", "temporal", "group"] = "stratified"


class CreateAnalysis(StrictModel):
    dataset_id: str


class TrainRequest(CreateAnalysis):
    fast: bool = False
    random_state: int = Field(default=42, ge=0, le=2**31 - 1)
    selection_metric: Literal["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy", "brier"] = "pr_auc"


class WhatIfRequest(StrictModel):
    row_index: int = Field(ge=0)
    changes: dict[str, Any] = Field(min_length=1, max_length=20)
    model: str | None = None


class EconomicsRequest(StrictModel):
    model: str | None = None
    budget: float | None = Field(default=None, ge=0, le=1e15)
    targeted: int = Field(default=100, ge=0, le=10_000_000)
    intervention_cost: float = Field(default=15, ge=0, le=1e12)
    incremental_impact: float = Field(default=.10, ge=0, le=1)
    contribution: float = Field(default=240, ge=0, le=1e12)
    fixed_cost: float = Field(default=0, ge=0, le=1e15)
    period: str = Field(default="12 months", min_length=1, max_length=100)
    currency: str = Field(default="USD", min_length=1, max_length=20)


class JobResponse(BaseModel):
    id: str
    analysis_id: str
    dataset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: int
    message: str
    error: str | None = None


class AnalysisEnvelope(BaseModel):
    analysis_id: str
    dataset_id: str
    model_version: str | None
    mode: str
