"""Gemeinsame Daten-/Hilfsfunktionen für die Tuninge (CatBoost)."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "car_prices_fin.csv"
RANDOM_STATE = 42
TARGET = "sellingprice"

# Basis-Features (= CatBoost-Bestmodell, Test 9)
NUMERIC = ["model_year", "vehicle_age", "odometer", "condition", "displacement"]
CATEGORICAL = ["make", "model", "trim", "body", "transmission",
               "state", "color", "interior", "make_model"]

PRICE_SEGMENTS = [("Budget", 500, 5_000), ("Economy", 5_000, 10_000),
                  ("Mid-Range", 10_000, 20_000), ("Premium", 20_000, 40_000),
                  ("Luxury", 40_000, 150_000)]


def load(sample: int = 0, with_derived: bool = False) -> pd.DataFrame:
    cols = ["year", "saledate", "make", "model", "trim", "body", "transmission",
            "state", "condition", "odometer", "color", "interior", TARGET, "displacement"]
    df = pd.read_csv(SRC, usecols=cols)
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=[c for c in cols if c != "displacement"]).copy()
    df["model_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vehicle_age"] = (df["saledate"].dt.year - df["model_year"]).clip(lower=0)
    df = df[df[TARGET].between(500, 150_000)
            & df["odometer"].between(1, 500_000)
            & df["vehicle_age"].between(0, 30)
            & df["condition"].between(1, 5)].copy()
    for c in CATEGORICAL:
        if c == "make_model":
            continue
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["make_model"] = df["make"] + "|" + df["model"]
    df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())

    if with_derived:
        df["miles_per_year"] = df["odometer"] / df["vehicle_age"].clip(lower=1)
        df["age_squared"] = df["vehicle_age"] ** 2
        df["disp_per_cond"] = df["displacement"] / df["condition"].clip(lower=0.1)

    df = df.reset_index(drop=True)
    if sample > 0 and len(df) > sample:
        df = df.sample(n=sample, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def dollar_metrics(y_true, pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {"mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "r2": float(r2_score(y_true, pred)),
            "mape": float(np.mean(np.abs((y_true - pred) / y_true)) * 100)}
