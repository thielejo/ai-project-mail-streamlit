"""
Test 1+2 — VIN/FIN-Datenabdeckung über die NHTSA vPIC API.

Zweck:
    Prüft, wie viele Fahrzeugmerkmale sich aus der VIN (US-Pendant zur FIN)
    zuverlaessig dekodieren lassen. Reine Mess-Routine, trainiert kein Modell.

Vorgehen:
    - Liest car_prices_clean.csv READ-ONLY aus dem Projekt-Root.
    - Zieht eine Zufallsstichprobe (SAMPLE_SIZE) und dekodiert die VINs ueber
      den NHTSA-Batch-Endpoint (bis 50 VINs pro Request, kein API-Key noetig).
    - Gibt die Befuellungsquote je Zielfeld aus und speichert das Rohergebnis
      lokal in diesem Ordner (vin_coverage_result.csv).

Aufruf:
    uv run python experiments/vin_fin_enrichment/01_coverage_test.py
    (SAMPLE_SIZE unten anpassen; getestet mit 200 und 1000)
"""

from __future__ import annotations
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "car_prices_clean.csv"
OUT = Path(__file__).resolve().parent / "vin_coverage_result.csv"

SAMPLE_SIZE = 1000
BATCH_SIZE = 50
SLEEP = 0.4
BATCH_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"
SEED = 42

WANTED = [
    "FuelTypePrimary",
    "EngineHP",
    "EngineCylinders",
    "DisplacementL",
    "DriveType",
    "TransmissionStyle",
    "BodyClass",
]


def decode_batch(vins: list[str]) -> list[dict]:
    payload = {"format": "json", "data": ";".join(vins)}
    resp = requests.post(BATCH_URL, data=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["Results"]


def main() -> None:
    print("Lese Stichprobe (read-only):", SRC.name)
    df = pd.read_csv(SRC, usecols=["vin", "make", "model", "year"]).sample(
        SAMPLE_SIZE, random_state=SEED
    )
    vins = df["vin"].astype(str).tolist()

    print(f"Dekodiere {len(vins)} VINs in Batches zu je {BATCH_SIZE} ...")
    rows: list[dict] = []
    for i in range(0, len(vins), BATCH_SIZE):
        rows.extend(decode_batch(vins[i:i + BATCH_SIZE]))
        time.sleep(SLEEP)
    res = pd.DataFrame(rows)

    print(f"\n=== Trefferquote je Feld (von {len(res)} VINs) ===")
    for col in WANTED:
        if col in res.columns:
            filled = res[col].replace("", pd.NA).notna().sum()
            print(f"  {col:20}: {filled:>4} / {len(res)}  ({filled / len(res) * 100:.1f} %)")
        else:
            print(f"  {col:20}: nicht in API-Antwort")

    keep = ["VIN", "Make", "Model", "ModelYear"] + [c for c in WANTED if c in res.columns]
    res[keep].to_csv(OUT, index=False)
    print(f"\nErgebnis gespeichert: {OUT.name}")


if __name__ == "__main__":
    main()
