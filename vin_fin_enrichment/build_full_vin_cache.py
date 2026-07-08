"""
Vollstaendiger VIN-Decode: dekodiert ALLE eindeutigen VINs aus
car_prices_clean.csv ueber die NHTSA vPIC Batch-API.

Eigenschaften:
- RESUME-SICHER: schreibt den Cache alle FLUSH_EVERY VINs per Append auf Platte.
  Bei einem Abbruch einfach erneut starten -> es wird dort fortgesetzt, wo es
  aufgehoert hat (bereits dekodierte VINs werden uebersprungen).
- SEED: uebernimmt beim ersten Lauf optional bereits vorhandene Decodes aus
  dem archivierten Sample-Cache, sodass weniger VINs neu geladen werden muessen.
- SLIM: speichert nur die relevanten Felder (klein genug fuer die Weiterverarbeitung).

Aufruf (am besten im Hintergrund, Laufzeit ~3 Std.):
    uv run python vin_fin_enrichment/build_full_vin_cache.py
"""

from __future__ import annotations
import csv
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "car_prices_clean.csv"
SEED = REPO / "archive/fin_enrichment_legacy_2026-07-08/vin_decoded_cache_sample.csv"
FULL = HERE / "vin_decoded_cache_full.csv"     # wachsender Voll-Cache (resume-Ziel)

BATCH_SIZE = 50
SLEEP = 0.3
FLUSH_EVERY = 5000
BATCH_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"

KEEP = ["VIN", "Make", "Model", "ModelYear", "FuelTypePrimary",
        "DisplacementL", "EngineCylinders", "EngineHP", "DriveType", "BodyClass"]


def decode_batch(vins: list[str], retries: int = 4) -> list[dict]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(BATCH_URL, data={"format": "json", "data": ";".join(vins)}, timeout=120)
            r.raise_for_status()
            return r.json()["Results"]
        except Exception as e:
            if attempt == retries:
                print(f"   ! Batch endgueltig fehlgeschlagen ({e}) — uebersprungen", flush=True)
                return []
            time.sleep(3 * attempt)
    return []


def load_done() -> set[str]:
    """VINs, die bereits dekodiert sind (aus Voll-Cache, sonst aus Seed)."""
    if FULL.exists():
        done = set(pd.read_csv(FULL, usecols=["VIN"], dtype=str)["VIN"])
        print(f"Voll-Cache gefunden: {len(done):,} VINs bereits dekodiert — setze fort.", flush=True)
        return done
    # Erstlauf: Seed uebernehmen
    if SEED.exists():
        seed = pd.read_csv(SEED, dtype=str)
        for c in KEEP:
            if c not in seed.columns:
                seed[c] = ""
        seed[KEEP].to_csv(FULL, index=False)
        done = set(seed["VIN"].astype(str))
        print(f"Seed uebernommen: {len(done):,} VINs in Voll-Cache geschrieben.", flush=True)
        return done
    # gar nichts vorhanden
    with FULL.open("w", newline="") as f:
        csv.writer(f).writerow(KEEP)
    print("Kein Seed — starte mit leerem Voll-Cache.", flush=True)
    return set()


def main() -> None:
    all_vins = pd.read_csv(SRC, usecols=["vin"], dtype=str)["vin"].dropna().unique().tolist()
    print(f"Eindeutige VINs gesamt: {len(all_vins):,}", flush=True)

    done = load_done()
    todo = [v for v in all_vins if v not in done]
    print(f"Noch zu dekodieren: {len(todo):,}", flush=True)
    if not todo:
        print("Nichts zu tun — Cache ist vollstaendig.", flush=True)
        return

    buffer: list[list] = []
    processed = 0
    t0 = time.time()

    def flush():
        nonlocal buffer
        if not buffer:
            return
        with FULL.open("a", newline="") as f:
            csv.writer(f).writerows(buffer)
        buffer = []

    for i in range(0, len(todo), BATCH_SIZE):
        results = decode_batch(todo[i:i + BATCH_SIZE])
        for r in results:
            buffer.append([r.get(c, "") for c in KEEP])
        processed += BATCH_SIZE
        if processed % FLUSH_EVERY < BATCH_SIZE:
            flush()
            rate = processed / max(time.time() - t0, 1)
            eta = (len(todo) - processed) / max(rate, 1) / 60
            print(f"  {min(processed, len(todo)):>7}/{len(todo)}  (~{eta:.0f} min verbleibend, Cache gesichert)", flush=True)
        time.sleep(SLEEP)

    flush()
    total = len(pd.read_csv(FULL, usecols=["VIN"]))
    print(f"\nFertig. Voll-Cache enthaelt jetzt {total:,} VINs: {FULL.name}", flush=True)


if __name__ == "__main__":
    main()
