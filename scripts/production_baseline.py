"""Gemeinsame Stage-1-Produktionsbasis (CatBoost + FIN/Hubraum) für Stage 2 und Stage 3.

Vorher rechneten `evaluate_stage2.py` und `stage3_seasonality.py` gegen das ältere
V2-XGBoost-Ensemble (`stage1_production_model.joblib`), während die App bereits das
getunte CatBoost-Modell mit Hubraum ausliefert. Die berichteten Stage-2-/Stage-3-
Kennzahlen gehörten damit zu einem Modell, das gar nicht mehr im Einsatz ist.

Dieses Modul liefert beiden Stufen dieselbe Basis wie die App:
  - Daten: car_prices_clean.csv + Hubraum aus dem versionierten VIN-Decode-Cache
  - Filter: identisch zu scripts/train_stage1_catboost.py (gleiche Zeilen)
  - Modell: models/price_model_catboost.cbm (Log-Ziel -> expm1)

Weil Filter und Reihenfolge dem Training entsprechen, reproduziert
train_test_split(..., random_state=42) exakt den Stage-1-Testsplit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "car_prices_clean.csv"
CACHE_PATH = PROJECT_ROOT / "vin_fin_enrichment" / "vin_decoded_cache_full.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "price_model_catboost.cbm"

RANDOM_STATE = 42
TARGET = "sellingprice"

NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition", "displacement"]
CATEGORICAL_FEATURES = ["make", "model", "trim", "body", "transmission",
                        "state", "color", "interior", "make_model"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_LABEL = "Stage 1 CatBoost (getunt, Log-Ziel) + FIN/Hubraum"


def load_production_frame() -> pd.DataFrame:
    """Bereinigte Fahrzeugdaten inkl. Hubraum, exakt wie im Produktionstraining gefiltert."""
    required = ["vin", "year", "saledate", "make", "model", "trim", "body", "transmission",
                "state", "condition", "odometer", "color", "interior", TARGET]

    if not CACHE_PATH.exists():
        raise SystemExit(
            f"VIN-Decode-Cache fehlt: {CACHE_PATH.relative_to(PROJECT_ROOT)}\n"
            "Er wird mit dem Repo ausgeliefert und laesst sich sonst regenerieren:\n"
            "    uv run python vin_fin_enrichment/build_full_vin_cache.py"
        )

    df = pd.read_csv(CLEAN_PATH, usecols=required)

    cache = pd.read_csv(CACHE_PATH, usecols=["VIN", "DisplacementL"], dtype={"VIN": str})
    cache = cache.drop_duplicates(subset="VIN").rename(
        columns={"VIN": "vin", "DisplacementL": "displacement"})
    df["vin"] = df["vin"].astype(str)
    df = df.merge(cache, on="vin", how="left")
    df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")

    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=[c for c in required if c != "vin"]).copy()
    df["model_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vehicle_age"] = (df["saledate"].dt.year - df["model_year"]).clip(lower=0)
    df = df[df[TARGET].between(500, 150_000)
            & df["odometer"].between(1, 500_000)
            & df["vehicle_age"].between(0, 30)
            & df["condition"].between(1, 5)].copy()

    for c in CATEGORICAL_FEATURES:
        if c == "make_model":
            continue
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["make_model"] = df["make"] + "|" + df["model"]
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())

    df["sale_month"] = df["saledate"].dt.month.astype(int)
    df["year_month"] = df["saledate"].dt.strftime("%Y-%m")
    return df.reset_index(drop=True)


def predict_baseline(df: pd.DataFrame) -> np.ndarray:
    """Zeitneutraler Stage-1-Basispreis des Produktionsmodells (CatBoost, Log-Ziel)."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    return np.maximum(np.expm1(model.predict(X)), 500.0)
