"""
Test 6 — Per-Feature-Ablation: welches FIN-Merkmal traegt den Effekt?

Schaltet jedes dekodierte Merkmal EINZELN zum V2-Set dazu, um seinen
isolierten Beitrag zu messen. Nutzt den mitgelieferten Decode-Cache
(vin_decoded_cache.csv) -> KEINE API-Aufrufe noetig.

Arme (identischer Split, identisches Modell HistGB):
   A) V2-Features (Basis)
   B) V2 + Kraftstoff
   C) V2 + Hubraum
   D) V2 + Zylinder
   E) V2 + alle drei

Aufruf:
    uv run python vin_fin_enrichment/05_per_feature_ablation.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "car_prices_clean.csv"
CACHE = Path(__file__).resolve().parent / "vin_decoded_cache.csv"
SAMPLE_SIZE = 100000
SEED = 42


def main() -> None:
    print("Lese 100k Sample + Decode-Cache (kein API-Call) ...", flush=True)
    raw = pd.read_csv(
        SRC,
        usecols=["vin", "year", "make", "model", "trim", "transmission", "state",
                 "condition", "odometer", "color", "interior", "saledate", "sellingprice"],
    ).sample(SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

    cache = pd.read_csv(CACHE, dtype=str)[["VIN", "FuelTypePrimary", "DisplacementL", "EngineCylinders"]]
    cache = cache.drop_duplicates(subset="VIN").rename(columns={"VIN": "vin"})
    raw["vin"] = raw["vin"].astype(str)
    df = raw.merge(cache, on="vin", how="left")

    df["fuel_type"] = df["FuelTypePrimary"].replace("", np.nan)
    df["displacement"] = pd.to_numeric(df["DisplacementL"], errors="coerce").round(1)
    df["cylinders"] = pd.to_numeric(df["EngineCylinders"], errors="coerce")

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

    print(f"  Verwertbare Zeilen: {len(df):,}", flush=True)

    v2_num = ["model_year", "vehicle_age", "odometer", "condition"]
    v2_cat = ["trim", "transmission", "state", "color", "interior", "make_model"]

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
        print(f"  {label:28} MAE ${mae:,.0f}  R2 {r2_score(price_test, pred):.4f}", flush=True)
        return mae

    print("\n=== Per-Feature-Ablation @ 100k ===", flush=True)
    base = run(v2_cat, v2_num, "A) V2 (Basis)")
    f = run(v2_cat + ["fuel_type"], v2_num, "B) V2 + Kraftstoff")
    d = run(v2_cat, v2_num + ["displacement"], "C) V2 + Hubraum")
    c = run(v2_cat, v2_num + ["cylinders"], "D) V2 + Zylinder")
    allf = run(v2_cat + ["fuel_type"], v2_num + ["displacement", "cylinders"], "E) V2 + alle drei")

    print("\n=== Beitrag je Merkmal (vs. Basis A) ===")
    for label, mae in [("Kraftstoff", f), ("Hubraum", d), ("Zylinder", c), ("alle drei", allf)]:
        print(f"  {label:12}: -${base - mae:>5,.0f}  ({(base - mae) / base * 100:+.2f} %)")


if __name__ == "__main__":
    main()
