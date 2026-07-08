"""
GROSSER Ablations-Test (100k) vs. V2 — lokal, ändert NICHTS im Repo.

Die belastbare Zahl: Bringt VIN (Kraftstoff/Hubraum/Zylinder) noch Zusatzwert,
wenn das V2-Feature-Set (inkl. trim) GENUG Daten hat, um nicht zu overfitten?

A) V2-Features        B) V2 + VIN   — identischer Split, identisches Modell.
Fortschritt wird mitgeloggt, damit man den Lauf verfolgen kann.
"""

from __future__ import annotations
import time
import sys
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
CACHE = Path(__file__).resolve().parent / "vin_decoded_cache_full.csv"

SAMPLE_SIZE = 100000
BATCH_SIZE = 50
SLEEP = 0.3
BATCH_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
SEED = 42


def decode_batch(vins: list[str], retries: int = 3) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(BATCH_URL, data={"format": "json", "data": ";".join(vins)}, timeout=120)
            r.raise_for_status()
            return r.json()["Results"]
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return []


FLUSH_EVERY = 5000  # alle N VINs den Zwischenstand auf Platte sichern


def get_decoded(raw: pd.DataFrame) -> pd.DataFrame:
    """VIN-Decode mit inkrementellem Cache: übersteht Abbrüche und setzt fort."""
    vins = raw["vin"].astype(str).tolist()

    rows: list[dict] = []
    if CACHE.exists():
        cached = pd.read_csv(CACHE, dtype=str)
        rows = cached.to_dict("records")
        if len(rows) >= len(vins):
            print(f"  Cache vollständig ({len(rows)} Zeilen) — überspringe Decode.")
            return cached
        print(f"  Cache gefunden: setze bei {len(rows)}/{len(vins)} fort.", flush=True)

    start = len(rows)
    t0 = time.time()
    for i in range(start, len(vins), BATCH_SIZE):
        rows.extend(decode_batch(vins[i:i + BATCH_SIZE]))
        done = min(i + BATCH_SIZE, len(vins))
        if done % FLUSH_EVERY < BATCH_SIZE:
            pd.DataFrame(rows).to_csv(CACHE, index=False)  # Zwischenstand sichern
            rate = (done - start) / max(time.time() - t0, 1)
            eta = (len(vins) - done) / max(rate, 1)
            print(f"  {done:>6}/{len(vins)}  (~{eta/60:.1f} min verbleibend, Cache gesichert)", flush=True)
        time.sleep(SLEEP)
    dec = pd.DataFrame(rows)
    dec.to_csv(CACHE, index=False)
    print(f"  Decode fertig, Cache geschrieben: {CACHE.name}")
    return dec


def main() -> None:
    print(f"Lese {SAMPLE_SIZE} Zeilen (read-only) ...", flush=True)
    raw = pd.read_csv(
        SRC,
        usecols=["vin", "year", "make", "model", "trim", "transmission", "state",
                 "condition", "odometer", "color", "interior", "saledate", "sellingprice"],
    ).sample(SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

    print("Dekodiere VINs (mit Cache) ...", flush=True)
    dec = get_decoded(raw).reset_index(drop=True)

    raw["fuel_type"] = dec["FuelTypePrimary"].replace("", np.nan).values[:len(raw)]
    raw["displacement"] = pd.to_numeric(dec["DisplacementL"], errors="coerce").round(1).values[:len(raw)]
    raw["cylinders"] = pd.to_numeric(dec["EngineCylinders"], errors="coerce").values[:len(raw)]

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

    cov_fuel = (raw["fuel_type"].notna()).mean()
    cov_disp = (raw["displacement"].notna()).mean()
    cov_cyl = (raw["cylinders"].notna()).mean()
    print(f"\n  Verwertbare Zeilen nach V2-Filter: {len(df):,}")
    print(f"  VIN-Abdeckung: fuel {cov_fuel:.3f} | disp {cov_disp:.3f} | cyl {cov_cyl:.3f}", flush=True)

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
        print(f"  {label:34} MAE ${mae:,.0f}  R2 {r2_score(price_test, pred):.4f}", flush=True)
        return mae

    print("\n=== Ablation vs. V2 @ 100k (identischer Split, identisches Modell) ===", flush=True)
    mae_v2 = run(v2_cat, v2_num, "A) V2-Features (inkl. trim)")
    mae_v2_vin = run(v2_cat + vin_cat, v2_num + vin_num, "B) V2 + VIN (Fuel/Disp/Cyl)")
    pct = (mae_v2 - mae_v2_vin) / mae_v2 * 100
    print("\n=== Ergebnis ===")
    print(f"  MAE-Differenz: ${mae_v2 - mae_v2_vin:,.0f}  ({pct:+.2f} %)")
    if abs(pct) < 1:
        print("  → Praktisch kein Zusatznutzen — trim deckt die Motorinfo bereits ab.")
    elif pct > 0:
        print("  → VIN bringt auch bei viel Daten echten Zusatzwert.")
    else:
        print("  → VIN verschlechtert leicht (Redundanz/Rauschen).")


if __name__ == "__main__":
    main()
