"""
Merge: haengt die dekodierten Motordaten (per VIN) an car_prices_clean.csv an
und erzeugt eine Stage-1-faehige Datei mit FIN-Features.

Voraussetzung: vin_decoded_cache_full.csv ist (weitgehend) vollstaendig
(siehe build_full_vin_cache.py).

Ergebnis: car_prices_fin.csv im Projekt-Root — car_prices_clean.csv plus die
Spalten fuel_type, displacement, cylinders (Hubraum ist laut Ablation das
entscheidende Merkmal).

Aufruf:
    uv run python experiments/vin_fin_enrichment/build_micro_fin_dataset.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "car_prices_clean.csv"
CACHE = HERE / "vin_decoded_cache_full.csv"
OUT = REPO / "car_prices_fin.csv"


def main() -> None:
    print("Lese Micro-Daten + Decode-Cache ...", flush=True)
    df = pd.read_csv(SRC, dtype={"vin": str})
    cache = pd.read_csv(CACHE, dtype=str)[
        ["VIN", "FuelTypePrimary", "DisplacementL", "EngineCylinders"]
    ].drop_duplicates(subset="VIN").rename(columns={"VIN": "vin"})

    merged = df.merge(cache, on="vin", how="left")
    merged["fuel_type"] = merged["FuelTypePrimary"].replace("", np.nan)
    merged["displacement"] = pd.to_numeric(merged["DisplacementL"], errors="coerce").round(1)
    merged["cylinders"] = pd.to_numeric(merged["EngineCylinders"], errors="coerce")
    merged = merged.drop(columns=["FuelTypePrimary", "DisplacementL", "EngineCylinders"])

    n = len(merged)
    print("Abdeckung im finalen Datensatz:")
    for col in ["displacement", "fuel_type", "cylinders"]:
        cov = merged[col].notna().mean() * 100
        print(f"  {col:14}: {cov:.1f} %")

    merged.to_csv(OUT, index=False)
    print(f"\nGeschrieben: {OUT.name}  ({n:,} Zeilen, {merged.shape[1]} Spalten)", flush=True)


if __name__ == "__main__":
    main()
