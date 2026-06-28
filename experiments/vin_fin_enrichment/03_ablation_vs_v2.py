"""
Test 4 — Ablation: VIN-Features zusaetzlich zum V2-Feature-Set.

Frage:
    Bringt die VIN (Kraftstoff/Hubraum/Zylinder) noch ZUSAETZLICHEN Wert,
    wenn das Modell bereits das V2-Feature-Set hat? V2 nutzt u.a. `trim`,
    das die Motorisierung schon stark codiert.

Vergleich (identischer Split, identisches Modell HistGradientBoosting):
    A) V2-Features = model_year, vehicle_age, odometer, condition,
                     trim, transmission, state, color, interior, make_model
    B) V2 + VIN    = A + fuel_type + displacement + cylinders

WICHTIG zur Interpretation:
    Die V2-Features sind hochkardinal (trim, state, make_model) und brauchen
    viele Zeilen. Auf einer kleinen Stichprobe overfittet V2 und der scheinbare
    VIN-Zusatznutzen ist nach oben verzerrt. Siehe README, Abschnitt
    "Interpretation".

Aufruf:
    uv run python experiments/vin_fin_enrichment/03_ablation_vs_v2.py
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

SAMPLE_SIZE = 14000
BATCH_SIZE = 50
SLEEP = 0.4
BATCH_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
SEED = 42


def decode_batch(vins: list[str]) -> list[dict]:
    r = requests.post(BATCH_URL, data={"format": "json", "data": ";".join(vins)}, timeout=90)
    r.raise_for_status()
    return r.json()["Results"]


def main() -> None:
    print(f"Lese {SAMPLE_SIZE} Zeilen (read-only) ...")
    raw = pd.read_csv(
        SRC,
        usecols=["vin", "year", "make", "model", "trim", "transmission", "state",
                 "condition", "odometer", "color", "interior", "saledate", "sellingprice"],
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

    df = raw.copy()
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    req = ["year", "make", "model", "trim", "transmission", "state",
           "condition", "odometer", "color", "interior", "saledate", "sellingprice"]
    df = df.dropna(subset=req).copy()
    df = df[df["odometer"].between(1, 500_000) & df["condition"].between(1, 5)]

    df["model_year"] = df["year"]
    df["vehicle_age"] = (df["saledate"].dt.year - df["year"]).clip(lower=0)
    df["make_model"] = df["make"].astype(str) + "|" + df["model"].astype(str)
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    df["displacement"] = df["displacement"].fillna(df["displacement"].median())
    df["cylinders"] = df["cylinders"].fillna(df["cylinders"].median())

    print(f"  Verwertbare Zeilen nach V2-Filter: {len(df):,}")

    v2_num = ["model_year", "vehicle_age", "odometer", "condition"]
    v2_cat = ["trim", "transmission", "state", "color", "interior", "make_model"]
    vin_num = ["displacement", "cylinders"]
    vin_cat = ["fuel_type"]

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

    print("\n=== Ablation vs. V2 (identischer Split, identisches Modell) ===")
    mae_v2 = run(v2_cat, v2_num, "A) V2-Features (inkl. trim)")
    mae_v2_vin = run(v2_cat + vin_cat, v2_num + vin_num, "B) V2 + VIN (Fuel/Disp/Cyl)")
    pct = (mae_v2 - mae_v2_vin) / mae_v2 * 100
    print(f"\n  MAE-Differenz: ${mae_v2 - mae_v2_vin:,.0f}  ({pct:+.2f} %)")


if __name__ == "__main__":
    main()
