"""
Test 3 — Ablation: VIN-Features vs. einfaches Baseline-Feature-Set.

Frage:
    Wie viel besser wird ein Stage-1-aehnliches Modell, wenn wir Kraftstoff,
    Hubraum und Zylinderzahl aus der VIN ergaenzen?

Vergleich (identischer Split, identisches Modell HistGradientBoosting):
    A) Baseline = make, model, body, condition, odometer, vehicle_age,
                  year_month, sale_month   (entspricht dem Produktions-Stage-1)
    B) Enriched = A + fuel_type + displacement + cylinders

Hinweis:
    Nutzt HistGradientBoosting fuer beide Arme. Der Vergleich ist dadurch
    intern valide; die absoluten Zahlen sind nicht 1:1 mit dem produktiven
    XGBoost-Ensemble vergleichbar.

Aufruf:
    uv run python experiments/vin_fin_enrichment/02_ablation_vs_baseline.py
"""

from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "car_prices_clean.csv"

SAMPLE_SIZE = 12000
BATCH_SIZE = 50
SLEEP = 0.4
BATCH_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
SEED = 42


def decode_batch(vins: list[str]) -> list[dict]:
    r = requests.post(BATCH_URL, data={"format": "json", "data": ";".join(vins)}, timeout=90)
    r.raise_for_status()
    return r.json()["Results"]


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=["year", "make", "model", "body", "condition",
                           "odometer", "saledate", "sellingprice"])
    df["sale_year"] = df["saledate"].dt.year
    df["sale_month"] = df["saledate"].dt.month
    df["year_month"] = df["saledate"].dt.strftime("%Y-%m")
    df["vehicle_age"] = (df["sale_year"] - df["year"]).clip(lower=0)
    return df


def main() -> None:
    print(f"Lese {SAMPLE_SIZE} Zeilen (read-only) ...")
    raw = pd.read_csv(
        SRC,
        usecols=["vin", "year", "make", "model", "body", "condition",
                 "odometer", "saledate", "sellingprice"],
    ).sample(SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

    vins = raw["vin"].astype(str).tolist()
    print(f"Dekodiere {len(vins)} VINs ...")
    decoded: list[dict] = []
    for i in range(0, len(vins), BATCH_SIZE):
        decoded.extend(decode_batch(vins[i:i + BATCH_SIZE]))
        time.sleep(SLEEP)
    dec = pd.DataFrame(decoded)

    raw["fuel_type"] = dec["FuelTypePrimary"].replace("", np.nan).values
    raw["displacement"] = pd.to_numeric(dec["DisplacementL"], errors="coerce").round(1).values
    raw["cylinders"] = pd.to_numeric(dec["EngineCylinders"], errors="coerce").values

    df = build_base_features(raw)
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())
    df["cylinders"] = df["cylinders"].fillna(df["cylinders"].median())

    base_cat = ["make", "model", "body", "year_month"]
    base_num = ["condition", "odometer", "vehicle_age", "sale_month"]
    vin_cat = ["fuel_type"]
    vin_num = ["displacement", "cylinders"]

    y = np.log1p(df["sellingprice"].astype(float).values)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
    price_test = df["sellingprice"].astype(float).values[te]

    def run(cat_cols, num_cols, label):
        X = df[cat_cols + num_cols].copy()
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols].astype(str))
        Xte[cat_cols] = enc.transform(Xte[cat_cols].astype(str))
        m = HistGradientBoostingRegressor(random_state=SEED, max_iter=400,
                                          learning_rate=0.06, max_depth=8)
        m.fit(Xtr[cat_cols + num_cols], y[tr])
        pred = np.expm1(m.predict(Xte[cat_cols + num_cols]))
        mae = mean_absolute_error(price_test, pred)
        print(f"  {label:34} MAE ${mae:,.0f}  R2 {r2_score(price_test, pred):.4f}")
        return mae

    print("\n=== Ablation (identischer Split, identisches Modell) ===")
    mae_base = run(base_cat, base_num, "A) Baseline (Produktions-Features)")
    mae_enr = run(base_cat + vin_cat, base_num + vin_num, "B) + VIN (Fuel/Disp/Cyl)")
    pct = (mae_base - mae_enr) / mae_base * 100
    print(f"\n  MAE-Differenz: ${mae_base - mae_enr:,.0f}  ({pct:+.2f} %)")


if __name__ == "__main__":
    main()
