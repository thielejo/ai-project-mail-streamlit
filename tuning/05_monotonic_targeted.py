"""
Schritt 5 — Gezielte Monotonie-Constraints (condition ↑, odometer ↓).

Motivation: Domänenwissen als harte Nebenbedingung — ein Auto in besserem
Zustand darf NIE billiger, mit mehr Kilometern NIE teurer vorhergesagt werden.
BEWUSST NICHT eingeschränkt: vehicle_age / model_year — denn Oldtimer verletzen
die Alters-Monotonie (an den Daten geprüft: Preis steigt bei 26+ Jahren wieder;
64 Fahrzeuge / 0,01 %, U-förmige Alterskurve).

Vergleich gegen das getunte Baseline-Finale (Schritt 4, MAE $1.042, identischer
Split/Config). Ziel ist NICHT primär MAE-Verbesserung, sondern garantierte
Plausibilität bei gleichbleibender Genauigkeit.

Aufruf: uv run python tuning/05_monotonic_targeted.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, dollar_metrics, NUMERIC, CATEGORICAL, TARGET, RANDOM_STATE, PRICE_SEGMENTS  # noqa

HERE = Path(__file__).resolve().parent
REF_MAE = 1042  # getuntes Baseline-Finale (Schritt 4, ohne Constraints)

# Nur die ausnahmslos monotonen Merkmale einschränken:
MONO = {"condition": 1, "odometer": -1}

# Orientierung + Parameter aus Schritt 1/2
TRANSFORM, LOSS = "log", "MAE"
params = {"learning_rate": 0.06, "depth": 10, "l2_leaf_reg": 3.0}
if (HERE / "02_best_params.json").exists():
    b = json.loads((HERE / "02_best_params.json").read_text())
    TRANSFORM, LOSS = b["transform"], b["loss"]
    params = b["best_params"]


def main() -> None:
    print(f"Setup: {TRANSFORM}/{LOSS} | Monotonie: {MONO} (Alter bewusst frei)", flush=True)
    df = load(sample=0)
    feats = NUMERIC + CATEGORICAL
    X = df[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=float)
    cat_idx = [feats.index(c) for c in CATEGORICAL]
    Xtr, Xte, ytr_raw, yte_raw = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    ytr = np.log1p(ytr_raw) if TRANSFORM == "log" else ytr_raw

    mono_list = [MONO.get(f, 0) for f in feats]  # 0 für alle Kategorien + freie Numerische
    p = dict(params); p.update({"iterations": 3000, "loss_function": LOSS,
                                "random_seed": RANDOM_STATE, "od_type": "Iter", "od_wait": 80,
                                "verbose": 300, "monotone_constraints": mono_list})

    print(f"Training (voller Datensatz): {len(Xtr):,} train / {len(Xte):,} test", flush=True)
    m = CatBoostRegressor(**p)
    m.fit(Pool(Xtr, ytr, cat_features=cat_idx),
          eval_set=Pool(Xte, np.log1p(yte_raw) if TRANSFORM == "log" else yte_raw, cat_features=cat_idx))
    pred = m.predict(Xte)
    if TRANSFORM == "log":
        pred = np.expm1(pred)
    mt = dollar_metrics(yte_raw, pred)

    print(f"\n=== Mit Monotonie (condition↑, odometer↓) ===", flush=True)
    print(f"MAE ${mt['mae']:,.0f} | RMSE ${mt['rmse']:,.0f} | R2 {mt['r2']:.4f} | MAPE {mt['mape']:.1f}%", flush=True)
    print(f"Referenz ohne Constraints (Schritt 4): MAE ${REF_MAE:,}", flush=True)
    d = mt['mae'] - REF_MAE
    print(f"=> Differenz: {d:+.0f} $ MAE ({d/REF_MAE*100:+.1f}%) — erwartet ~0", flush=True)

    seg = []
    for name, lo, hi in PRICE_SEGMENTS:
        mask = (yte_raw >= lo) & (yte_raw <= hi)
        if mask.sum() >= 10:
            from sklearn.metrics import mean_absolute_error
            seg.append({"segment": name, "n": int(mask.sum()),
                        "mae": float(mean_absolute_error(yte_raw[mask], np.asarray(pred)[mask]))})

    m.save_model(str(HERE / "price_model_catboost_monotonic.cbm"))
    out = {"monotone_constraints": MONO, "note_age": "vehicle_age/model_year bewusst frei (Oldtimer, U-Kurve)",
           "transform": TRANSFORM, "loss": LOSS, "metrics": mt, "reference_mae": REF_MAE, "segments": seg}
    (HERE / "05_results.json").write_text(json.dumps(out, indent=2))
    print("Modell: price_model_catboost_monotonic.cbm | Ergebnis: 05_results.json", flush=True)


if __name__ == "__main__":
    main()
