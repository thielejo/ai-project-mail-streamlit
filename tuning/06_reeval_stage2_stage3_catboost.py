"""
Schritt 6 — Stage 2 & Stage 3 gegen das neue CatBoost-Stage-1 neu auswerten.

Hintergrund: Stage 2 (CPI) und Stage 3 (Saison) waren gegen das alte V2-Modell
kalibriert. Da die App jetzt das getunte CatBoost-Modell (+ Hubraum) nutzt,
werden die Faktoren und Kennzahlen hier gegen DIESE Basis neu berechnet — die
Logik bleibt identisch, nur die Stage-1-Basispreise wechseln.

Methodik 1:1 wie in scripts/stage3_seasonality.py (dieselbe Faktor-Funktion wird
importiert und wiederverwendet); einzige Änderung: reference_prediction kommt aus
CatBoost (zeitneutral trainiert, Log-Ziel → expm1) statt aus V2.

Voraussetzung: car_prices_fin.csv (clean + Hubraum; gitignored, regenerierbar via
scripts + experiments) und models/price_model_catboost.cbm.

Aufruf: uv run python tuning/06_reeval_stage2_stage3_catboost.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stage3_seasonality import calculate_seasonality_factors  # noqa: E402
from stage1_runtime import CATBOOST_FEATURES, CATBOOST_CATEGORICAL_FEATURES  # noqa: E402

RANDOM_STATE = 42  # identisch zu evaluate_stage3.py

FIN_PATH = REPO / "car_prices_fin.csv"
MACRO_PATH = REPO / "data" / "macro_index.csv"
MODEL_PATH = REPO / "models" / "price_model_catboost.cbm"
OUT_FACTORS = REPO / "tuning" / "stage3_seasonality_factors_catboost.csv"
OUT_JSON = REPO / "tuning" / "06_reeval_results.json"


def load_catboost_base() -> pd.DataFrame:
    """Baut adjusted_rows (wie stage3_seasonality), aber mit CatBoost-Basispreis."""
    from catboost import CatBoostRegressor
    cols = ["year", "saledate", "make", "model", "trim", "body", "transmission",
            "state", "condition", "odometer", "color", "interior", "sellingprice", "displacement"]
    df = pd.read_csv(FIN_PATH, usecols=cols)
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=[c for c in cols if c != "displacement"]).copy()
    df["model_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["vehicle_age"] = (df["saledate"].dt.year - df["model_year"]).clip(lower=0)
    df = df[df["sellingprice"].between(500, 150_000)
            & df["odometer"].between(1, 500_000)
            & df["vehicle_age"].between(0, 30)
            & df["condition"].between(1, 5)].copy()
    for c in CATBOOST_CATEGORICAL_FEATURES:
        if c == "make_model":
            continue
        df[c] = df[c].astype(str).str.strip().str.lower()
    df["make_model"] = df["make"] + "|" + df["model"]
    df["displacement"] = pd.to_numeric(df["displacement"], errors="coerce")
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())
    df["sale_month"] = df["saledate"].dt.month.astype(int)
    df["year_month"] = df["saledate"].dt.strftime("%Y-%m")

    model = CatBoostRegressor(); model.load_model(str(MODEL_PATH))
    X = df[CATBOOST_FEATURES].copy()
    for c in CATBOOST_CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    # CatBoost: Log-Ziel → expm1; zeitneutral (kein Monat) = Referenz-Basispreis
    df["reference_prediction"] = np.maximum(np.expm1(model.predict(X)), 500.0)

    macro = pd.read_csv(MACRO_PATH, usecols=["year_month", "cpi_multiplier"])
    cpi = macro.set_index("year_month")["cpi_multiplier"]
    df["cpi_multiplier"] = df["year_month"].map(cpi)
    df = df.dropna(subset=["cpi_multiplier"]).copy()
    df["normalized_price"] = df["sellingprice"] / df["cpi_multiplier"]
    df["price_ratio"] = df["normalized_price"] / df["reference_prediction"]
    return df[["body", "sale_month", "year_month", "normalized_price",
               "reference_prediction", "price_ratio"]]


def holdout_eval(adjusted_rows: pd.DataFrame) -> dict:
    """Identisch zu evaluate_stage3._holdout_evaluation."""
    rng = np.random.default_rng(RANDOM_STATE)
    mask = rng.random(len(adjusted_rows)) < 0.8
    train, test = adjusted_rows.loc[mask], adjusted_rows.loc[~mask].copy()
    factors = calculate_seasonality_factors(train)
    lookup = factors.set_index(["body", "sale_month"])["seasonal_factor"]
    test["seasonal_factor"] = [float(lookup.get((b, m), 1.0))
                               for b, m in zip(test["body"], test["sale_month"])]
    base = (test["normalized_price"] - test["reference_prediction"]).abs().mean()
    seas = (test["normalized_price"] - test["reference_prediction"] * test["seasonal_factor"]).abs().mean()
    return {"train_rows": int(mask.sum()), "test_rows": int((~mask).sum()),
            "baseline_mae": round(float(base), 2), "stage3_mae": round(float(seas), 2),
            "mae_change": round(float(seas - base), 2),
            "mae_change_percent": round(float((seas / base - 1.0) * 100), 2)}


def main() -> None:
    print("Baue CatBoost-Basispreise (reference_prediction) ...", flush=True)
    adjusted = load_catboost_base()
    print(f"Auswertbare Zeilen: {len(adjusted):,}", flush=True)

    # Stage 3: volle Faktoren + Holdout-Bewertung
    full_factors = calculate_seasonality_factors(adjusted)
    full_factors.to_csv(OUT_FACTORS, index=False)
    ev = holdout_eval(adjusted)
    print("\n=== Stage 3 (gegen CatBoost) ===", flush=True)
    print(f"  Baseline-MAE (CPI-norm.): ${ev['baseline_mae']:,.0f}", flush=True)
    print(f"  mit Stage 3:              ${ev['stage3_mae']:,.0f}", flush=True)
    print(f"  Änderung: {ev['mae_change']:+.2f} $ ({ev['mae_change_percent']:+.2f} %)", flush=True)

    # Beste/schlechteste Monate für die großen Karosserieformen
    top = (full_factors.groupby("body").agg(n=("body_observations", "first")).reset_index()
           .sort_values("n", ascending=False).head(6)["body"].tolist())
    print("  Beispiele Bestmonat:", flush=True)
    for b in top:
        sub = full_factors[(full_factors.body == b) & (full_factors.confidence != "no_data")]
        if len(sub):
            bm = sub.loc[sub.seasonal_factor.idxmax()]
            print(f"    {b}: Monat {int(bm.sale_month)} (+{(bm.seasonal_factor-1)*100:.1f}%)", flush=True)

    json.dump({"stage3_holdout": ev, "n_rows": int(len(adjusted)),
               "model": "price_model_catboost.cbm"}, open(OUT_JSON, "w"), indent=2)
    print(f"\nFaktoren: {OUT_FACTORS.name} | Ergebnis: {OUT_JSON.name}", flush=True)


if __name__ == "__main__":
    main()
