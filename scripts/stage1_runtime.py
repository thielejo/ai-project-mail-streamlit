"""Shared Stage-1 runtime helpers for the app and evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_MODEL_DIR = PROJECT_ROOT / "archive" / "models" / "artifacts"
V1_MODEL_PATH = ARCHIVE_MODEL_DIR / "stage1_legacy_histgb_model.joblib"
V2_MODEL_PATH = PROJECT_ROOT / "models" / "stage1_production_model.joblib"
V1_METRICS_PATH = ARCHIVE_MODEL_DIR / "stage1_legacy_histgb_metrics.json"
V2_METRICS_PATH = PROJECT_ROOT / "models" / "stage1_production_metrics.json"

V1_FEATURES = ["vehicle_age", "sale_month", "odometer", "condition", "year_month", "make", "model", "body"]
V2_NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition"]
V2_CATEGORICAL_FEATURES = ["make", "model", "trim", "body", "transmission", "state", "color", "interior", "make_model"]
V2_FEATURES = V2_NUMERIC_FEATURES + V2_CATEGORICAL_FEATURES


def load_production_model() -> tuple[object, str]:
    """Prefer V2 and retain V1 as an automatic fallback."""
    if V2_MODEL_PATH.exists():
        return joblib.load(V2_MODEL_PATH), "v2"
    if V1_MODEL_PATH.exists():
        return joblib.load(V1_MODEL_PATH), "v1"
    raise FileNotFoundError("Neither the V2 nor V1 Stage-1 model is available.")


def build_v2_input(*, model_year: int, vehicle_age: int, odometer: int, condition: float,
                   make: str, model: str, trim: str, body: str, transmission: str,
                   state: str, color: str, interior: str) -> pd.DataFrame:
    values = {
        "model_year": int(model_year), "vehicle_age": int(vehicle_age),
        "odometer": int(odometer), "condition": float(condition),
        "make": str(make).strip().lower(), "model": str(model).strip().lower(),
        "trim": str(trim).strip().lower(), "body": str(body).strip().lower(),
        "transmission": str(transmission).strip().lower(), "state": str(state).strip().lower(),
        "color": str(color).strip().lower(), "interior": str(interior).strip().lower(),
    }
    values["make_model"] = values["make"] + "|" + values["model"]
    return pd.DataFrame([values], columns=V2_FEATURES)


def build_v1_input(*, vehicle_age: int, odometer: int, condition: float, make: str,
                   model: str, body: str, reference_month: int = 2,
                   reference_year_month: str = "2015-02") -> pd.DataFrame:
    return pd.DataFrame([{
        "vehicle_age": int(vehicle_age), "sale_month": int(reference_month),
        "odometer": int(odometer), "condition": float(condition),
        "year_month": reference_year_month, "make": str(make).strip().lower(),
        "model": str(model).strip().lower(), "body": str(body).strip().lower(),
    }], columns=V1_FEATURES)


def get_v2_category_options(model: object) -> dict[str, list[str]]:
    """Read fitted encoder categories so the UI offers values known to V2."""
    raw_pipeline = model.named_estimators_["raw_price"]
    encoder = raw_pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
    return {name: [str(value) for value in values]
            for name, values in zip(V2_CATEGORICAL_FEATURES, encoder.categories_)}
