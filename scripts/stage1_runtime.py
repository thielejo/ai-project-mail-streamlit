"""Shared Stage-1 runtime helpers for the app and evaluation scripts."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_MODEL_DIR = PROJECT_ROOT / "archive" / "models" / "artifacts"
V1_MODEL_PATH = ARCHIVE_MODEL_DIR / "stage1_legacy_histgb_model.joblib"
V2_MODEL_PATH = PROJECT_ROOT / "models" / "stage1_production_model.joblib"
V1_METRICS_PATH = ARCHIVE_MODEL_DIR / "stage1_legacy_histgb_metrics.json"
V2_METRICS_PATH = PROJECT_ROOT / "models" / "stage1_production_metrics.json"

# --- CatBoost (getunt + FIN/Hubraum) — neues bevorzugtes Modell ---
CATBOOST_MODEL_PATH = PROJECT_ROOT / "models" / "price_model_catboost.cbm"
CATBOOST_METRICS_PATH = PROJECT_ROOT / "models" / "price_model_catboost_metrics.json"
DISPLACEMENT_LOOKUP_PATH = PROJECT_ROOT / "models" / "displacement_lookup.csv"
CATBOOST_OPTIONS_PATH = PROJECT_ROOT / "models" / "catboost_category_options.json"

V1_FEATURES = ["vehicle_age", "sale_month", "odometer", "condition", "year_month", "make", "model", "body"]
V2_NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition"]
V2_CATEGORICAL_FEATURES = ["make", "model", "trim", "body", "transmission", "state", "color", "interior", "make_model"]
V2_FEATURES = V2_NUMERIC_FEATURES + V2_CATEGORICAL_FEATURES

# CatBoost nutzt dieselben V2-Merkmale + den Hubraum (displacement). Reihenfolge
# muss exakt dem Training entsprechen (scripts/train_stage1_catboost.py).
CATBOOST_NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition", "displacement"]
CATBOOST_CATEGORICAL_FEATURES = V2_CATEGORICAL_FEATURES
CATBOOST_FEATURES = CATBOOST_NUMERIC_FEATURES + CATBOOST_CATEGORICAL_FEATURES
DEFAULT_DISPLACEMENT = 3.0  # globaler Fallback (Median), falls Modell unbekannt


def load_production_model() -> tuple[object, str]:
    """Bevorzugt CatBoost (getunt + Hubraum), dann V2, dann V1 als Fallback."""
    if CATBOOST_MODEL_PATH.exists():
        try:
            from catboost import CatBoostRegressor

            model = CatBoostRegressor()
            model.load_model(str(CATBOOST_MODEL_PATH))
            return model, "catboost"
        except Exception as error:
            print(f"Could not load Stage 1 CatBoost model, falling back to V2/V1: {error}")
    if V2_MODEL_PATH.exists():
        try:
            return joblib.load(V2_MODEL_PATH), "v2"
        except Exception as error:
            if not V1_MODEL_PATH.exists():
                raise
            print(f"Could not load Stage 1 V2 model, falling back to V1: {error}")
    if V1_MODEL_PATH.exists():
        return joblib.load(V1_MODEL_PATH), "v1"
    raise FileNotFoundError("Kein Stage-1-Modell verfuegbar (CatBoost/V2/V1).")


def predict_stage1(model: object, version: str, prediction_input: pd.DataFrame) -> float:
    """Einheitliche Vorhersage. CatBoost wurde auf log1p(Preis) trainiert →
    Rueck-Transformation mit expm1; V1/V2 geben den Preis direkt aus."""
    raw = float(model.predict(prediction_input)[0])
    if version == "catboost":
        return float(np.expm1(raw))
    return raw


# --- Hubraum-Lookup (Hybrid: Auto-Vorschlag, in der App ueberschreibbar) ---
def load_displacement_lookup() -> dict[str, float]:
    if not DISPLACEMENT_LOOKUP_PATH.exists():
        return {}
    df = pd.read_csv(DISPLACEMENT_LOOKUP_PATH)
    return dict(zip(df["make_model"].astype(str), df["displacement"].astype(float)))


def lookup_displacement(make: str, model: str, lookup: dict[str, float] | None = None) -> float:
    if lookup is None:
        lookup = load_displacement_lookup()
    key = f"{str(make).strip().lower()}|{str(model).strip().lower()}"
    return float(lookup.get(key, DEFAULT_DISPLACEMENT))


def get_catboost_category_options() -> dict[str, list[str]]:
    """Kategorie-Optionen fuers UI (CatBoost hat keinen sklearn-Encoder)."""
    if CATBOOST_OPTIONS_PATH.exists():
        return json.loads(CATBOOST_OPTIONS_PATH.read_text(encoding="utf-8"))
    return {}


def build_catboost_input(*, model_year: int, vehicle_age: int, odometer: int, condition: float,
                         displacement: float, make: str, model: str, trim: str, body: str,
                         transmission: str, state: str, color: str, interior: str) -> pd.DataFrame:
    values = {
        "model_year": int(model_year), "vehicle_age": int(vehicle_age),
        "odometer": int(odometer), "condition": float(condition),
        "displacement": float(displacement),
        "make": str(make).strip().lower(), "model": str(model).strip().lower(),
        "trim": str(trim).strip().lower(), "body": str(body).strip().lower(),
        "transmission": str(transmission).strip().lower(), "state": str(state).strip().lower(),
        "color": str(color).strip().lower(), "interior": str(interior).strip().lower(),
    }
    values["make_model"] = values["make"] + "|" + values["model"]
    return pd.DataFrame([values], columns=CATBOOST_FEATURES)


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
