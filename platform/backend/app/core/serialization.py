"""Explicit JSON normalization at the HTTP/storage boundaries."""
from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict("records"))
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def safe_csv(frame: pd.DataFrame) -> bytes:
    """Quote CSV fields and neutralize spreadsheet formulas in textual cells."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].map(
            lambda value: "'" + value if isinstance(value, str)
            and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else value
        )
    return result.to_csv(index=False).encode("utf-8-sig")
