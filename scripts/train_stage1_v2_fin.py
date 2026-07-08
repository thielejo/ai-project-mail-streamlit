"""
Stage 1 V2 + FIN — integrierte Trainings-Pipeline.

Vereint die zwei Entwicklungsstraenge des Teams:
  1. Pascals V2-Ensemble (reiche Manheim-Merkmale: trim, transmission, state,
     color, interior, make_model) — XGBoost VotingRegressor 50/50.
  2. Moritz' FIN-Anreicherung: der aus der VIN dekodierte Hubraum (displacement),
     der laut Ablation den gesamten FIN-Effekt traegt.

Architektur und Hyperparameter sind identisch zu scripts/train_stage1_v2.py;
einzige Aenderung: `displacement` wird als numerisches Feature ergaenzt.

Datenquelle: car_prices_clean.csv + Hubraum aus dem VIN-Decode-Cache
(vin_fin_enrichment/vin_decoded_cache_full.csv), per VIN gejoint.

Aufruf:
    uv run python scripts/train_stage1_v2_fin.py            # 200k Schnelllauf
    uv run python scripts/train_stage1_v2_fin.py --max-rows 0   # voller Datensatz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

REPO = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO / "car_prices_clean.csv"
CACHE_PATH = REPO / "vin_fin_enrichment/vin_decoded_cache_full.csv"
MODEL_PATH = REPO / "models/price_model_v2_fin.joblib"
METRICS_PATH = REPO / "models/price_model_v2_fin_metrics.json"

RANDOM_STATE = 42
TARGET = "sellingprice"

# V2-Features + neues FIN-Feature `displacement`
NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition", "displacement"]
CATEGORICAL_FEATURES = ["make", "model", "trim", "body", "transmission",
                        "state", "color", "interior", "make_model"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PRICE_SEGMENTS = [("Budget", 500, 5_000), ("Economy", 5_000, 10_000),
                  ("Mid-Range", 10_000, 20_000), ("Premium", 20_000, 40_000),
                  ("Luxury", 40_000, 150_000)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 V2 + FIN (displacement).")
    p.add_argument("--max-rows", type=int, default=200_000,
                   help="Max Zeilen nach Cleaning; 0 = voller Datensatz.")
    return p.parse_args()


def load_data(max_rows: int) -> pd.DataFrame:
    required = ["vin", "year", "saledate", "make", "model", "trim", "body",
                "transmission", "state", "condition", "odometer", "color", "interior", TARGET]
    df = pd.read_csv(INPUT_PATH, usecols=required)

    # Hubraum aus dem VIN-Decode-Cache joinen
    cache = pd.read_csv(CACHE_PATH, usecols=["VIN", "DisplacementL"], dtype={"VIN": str})
    cache = cache.drop_duplicates(subset="VIN").rename(columns={"VIN": "vin", "DisplacementL": "displacement"})
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
    # Hubraum: fehlende Werte mit Median fuellen (Abdeckung ~99 %)
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def build_xgb(objective: str) -> XGBRegressor:
    return XGBRegressor(objective=objective, eval_metric="mae", n_estimators=700,
                        learning_rate=0.045, max_depth=9, min_child_weight=12,
                        subsample=0.90, colsample_bytree=0.90, reg_alpha=0.02,
                        reg_lambda=1.0, tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1)


def build_model() -> VotingRegressor:
    def pre():
        return ColumnTransformer([
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True), CATEGORICAL_FEATURES),
        ])
    raw = Pipeline([("preprocessor", pre()), ("model", build_xgb("reg:squarederror"))])
    logp = TransformedTargetRegressor(
        regressor=Pipeline([("preprocessor", pre()), ("model", build_xgb("reg:absoluteerror"))]),
        func=np.log1p, inverse_func=np.expm1)
    return VotingRegressor(estimators=[("raw_price", raw), ("log_price", logp)], weights=[0.5, 0.5])


def metrics(y, p) -> dict:
    return {"mae": round(float(mean_absolute_error(y, p)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "r2": round(float(r2_score(y, p)), 4),
            "mape_percent": round(float(np.mean(np.abs((y.to_numpy() - p) / y.to_numpy())) * 100), 2)}


def main() -> None:
    args = parse_args()
    df = load_data(args.max_rows)
    print(f"Zeilen nach Filter: {len(df):,}", flush=True)

    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = build_model()
    print("Trainiere V2+FIN-Ensemble ...", flush=True)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    m = metrics(yte, pred)

    seg = []
    ya = yte.to_numpy()
    for name, lo, hi in PRICE_SEGMENTS:
        mask = (ya >= lo) & (ya <= hi)
        if mask.sum() >= 10:
            seg.append({"segment": name, "n": int(mask.sum()),
                        "mae": round(float(mean_absolute_error(ya[mask], pred[mask])), 2),
                        "mape_percent": round(float(np.mean(np.abs((ya[mask]-pred[mask])/ya[mask]))*100), 2)})

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    meta = {"created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": "Stage1 V2 + FIN (XGBoost VotingRegressor + displacement)",
            "rows_used": int(len(df)), "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
            "features": FEATURES, "target": TARGET, "metrics": m, "segment_metrics": seg}
    METRICS_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"MAE ${m['mae']:,.0f} | RMSE ${m['rmse']:,.0f} | R2 {m['r2']:.4f} | MAPE {m['mape_percent']:.1f}%")
    print(f"Modell: {MODEL_PATH.name} | Metriken: {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
