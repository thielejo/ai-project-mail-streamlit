"""
Stage 1 — Produktions-Trainingsskript: CatBoost + FIN (Hubraum), getunt.

Vereint die Erkenntnisse des Tuning-Experiments (tuning/) in einem
produktiven Skript:
  - Modell: CatBoost (native kategoriale Merkmale)
  - Zielgröße: log1p(Preis) mit MAE-Loss  (Schritt 1)
  - Hyperparameter: aus der Optuna-Suche  (Schritt 2)
  - FIN-Feature: Hubraum (displacement) aus dem VIN-Decode-Cache, per VIN gejoint
  - KEINE Monotonie-Constraints (Schritt 3+5: verschlechtern das Modell)

Ergebnis des committeten Artefakts (voller Datensatz, --iterations 2000):
MAE $1.056,54 / RMSE $1.892,82 / R² 0,9606 / MAPE 11,91 %
(siehe models/price_model_catboost_metrics.json).

Datenquelle: data/car_prices_clean.csv + Hubraum aus
vin_fin_enrichment/vin_decoded_cache_full.csv (per VIN).

Der vollständige Decode-Cache wird im Repo versioniert, damit das Training ohne
erneuten Abruf aller VINs reproduzierbar bleibt. Zum Aktualisieren oder
vollständigen Neuerzeugen über die freie NHTSA-API:
    uv run python vin_fin_enrichment/build_full_vin_cache.py   # resume-sicher

Aufruf:
    uv run python scripts/train_stage1_catboost.py            # 200k Schnelllauf
    uv run python scripts/train_stage1_catboost.py --max-rows 0   # voller Datensatz
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent
INPUT_PATH = REPO / "data" / "car_prices_clean.csv"
CACHE_PATH = REPO / "vin_fin_enrichment/vin_decoded_cache_full.csv"
TUNED_PARAMS_PATH = REPO / "tuning/02_best_params.json"
MODEL_PATH = REPO / "models/price_model_catboost.cbm"
METRICS_PATH = REPO / "models/price_model_catboost_metrics.json"

RANDOM_STATE = 42
TARGET = "sellingprice"

NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition", "displacement"]
CATEGORICAL_FEATURES = ["make", "model", "trim", "body", "transmission",
                        "state", "color", "interior", "make_model"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Getunte Parameter (Optuna, Schritt 2) — Fallback, falls JSON fehlt.
DEFAULT_PARAMS = {"learning_rate": 0.05988715015642537, "depth": 10,
                  "l2_leaf_reg": 3.369446753344074, "random_strength": 0.7592849564817107,
                  "bagging_temperature": 1.7569362887347286, "min_data_in_leaf": 15}

PRICE_SEGMENTS = [("Budget", 500, 5_000), ("Economy", 5_000, 10_000),
                  ("Mid-Range", 10_000, 20_000), ("Premium", 20_000, 40_000),
                  ("Luxury", 40_000, 150_000)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the tuned CatBoost Stage 1 model (with FIN displacement).")
    p.add_argument("--max-rows", type=int, default=200_000, help="Max Zeilen nach Cleaning; 0 = voller Datensatz.")
    p.add_argument("--depth", type=int, default=None, help="Baumtiefe ueberschreiben (Deploy: kleineres Modell z. B. 8).")
    p.add_argument("--iterations", type=int, default=2000,
                   help="Anzahl Baeume. Standard 2000 haelt die Modell-Datei unter GitHubs "
                        "100-MB-Grenze (committbar) bei nahezu voller Genauigkeit (Tiefe 10 bleibt).")
    return p.parse_args()


def load_tuned_params() -> dict:
    if TUNED_PARAMS_PATH.exists():
        return json.loads(TUNED_PARAMS_PATH.read_text()).get("best_params", DEFAULT_PARAMS)
    return DEFAULT_PARAMS


def load_data(max_rows: int) -> pd.DataFrame:
    required = ["vin", "year", "saledate", "make", "model", "trim", "body",
                "transmission", "state", "condition", "odometer", "color", "interior", TARGET]
    if not CACHE_PATH.exists():
        raise SystemExit(
            f"VIN-Decode-Cache fehlt: {CACHE_PATH.relative_to(REPO)}\n"
            "Er wird normalerweise mit dem Repo ausgeliefert und laesst sich bei Bedarf\n"
            "aus der freien NHTSA-API regenerieren (resume-sicher):\n"
            "    uv run python vin_fin_enrichment/build_full_vin_cache.py"
        )

    df = pd.read_csv(INPUT_PATH, usecols=required)

    # FIN: Hubraum aus dem VIN-Decode-Cache per VIN joinen
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
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    params = load_tuned_params()
    if args.depth is not None:  # Deploy-Override fuer ein kleineres Modell
        params = dict(params); params["depth"] = args.depth
    df = load_data(args.max_rows)
    print(f"Zeilen nach Filter: {len(df):,} | getunte Parameter: {params}", flush=True)

    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=float)
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL_FEATURES]

    Xtr, Xte, ytr_raw, yte_raw = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    ytr = np.log1p(ytr_raw)  # Log-Zielgröße (Schritt 1)

    p = dict(params); p.update({"iterations": args.iterations, "loss_function": "MAE",
                                "random_seed": RANDOM_STATE, "od_type": "Iter",
                                "od_wait": 80, "verbose": 300})
    model = CatBoostRegressor(**p)
    model.fit(Pool(Xtr, ytr, cat_features=cat_idx),
              eval_set=Pool(Xte, np.log1p(yte_raw), cat_features=cat_idx))

    pred = np.expm1(model.predict(Xte))
    mae = mean_absolute_error(yte_raw, pred)
    rmse = float(np.sqrt(mean_squared_error(yte_raw, pred)))
    r2 = r2_score(yte_raw, pred)
    mape = float(np.mean(np.abs((yte_raw - pred) / yte_raw)) * 100)

    seg = []
    for name, lo, hi in PRICE_SEGMENTS:
        mask = (yte_raw >= lo) & (yte_raw <= hi)
        if mask.sum() >= 10:
            seg.append({"segment": name, "n": int(mask.sum()),
                        "mae": round(float(mean_absolute_error(yte_raw[mask], np.asarray(pred)[mask])), 2)})

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    meta = {"created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": "Stage 1 CatBoost (tuned, log-target) + FIN displacement",
            "rows_used": int(len(df)), "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
            "features": FEATURES, "target": TARGET, "params": params,
            "metrics": {"mae": round(float(mae), 2), "rmse": round(rmse, 2),
                        "r2": round(float(r2), 4), "mape_percent": round(mape, 2)},
            "segment_metrics": seg}
    METRICS_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nMAE ${mae:,.0f} | RMSE ${rmse:,.0f} | R2 {r2:.4f} | MAPE {mape:.1f}%", flush=True)
    print(f"Modell: {MODEL_PATH.name} | Metriken: {METRICS_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
