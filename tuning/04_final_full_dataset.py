"""
Schritt 4 — Finale: bestes getuntes Setup auf dem VOLLEN Datensatz.

Übernimmt:
  - beste Zielgröße/Loss-Orientierung (Schritt 1),
  - beste Hyperparameter (Schritt 2),
  - beste Feature-/Monotonie-Variante (Schritt 3, automatisch die mit dem
    geringsten CV-MAE).
Trainiert auf dem vollen Datensatz (80/20-Split rs=42) und vergleicht gegen
das bisherige CatBoost-Bestmodell (MAE $1.120).

Aufruf: uv run python tuning/04_final_full_dataset.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (load, dollar_metrics, NUMERIC, CATEGORICAL, TARGET,
                     RANDOM_STATE, PRICE_SEGMENTS)  # noqa

HERE = Path(__file__).resolve().parent
REF_BASELINE_MAE = 1120  # CatBoost + Hubraum, ungetunt, voller Datensatz (Test 9)
DERIVED = ["miles_per_year", "age_squared", "disp_per_cond"]
MONO = {"model_year": 1, "vehicle_age": -1, "odometer": -1, "condition": 1,
        "displacement": 1, "miles_per_year": -1, "age_squared": -1, "disp_per_cond": 0}


def load_config():
    transform, loss = "log", "RMSE"
    params = {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0}
    if (HERE / "02_best_params.json").exists():
        b = json.loads((HERE / "02_best_params.json").read_text())
        transform, loss = b["transform"], b["loss"]
        params = b["best_params"]
    # beste Feature-Variante aus Schritt 3
    use_derived, use_mono = False, False
    if (HERE / "03_results.json").exists():
        r = json.loads((HERE / "03_results.json").read_text())
        best_key = min(["a_baseline", "b_derived", "c_derived_monotonic"], key=lambda k: r[k]["mae"])
        use_derived = best_key in ("b_derived", "c_derived_monotonic")
        use_mono = best_key == "c_derived_monotonic"
    return transform, loss, params, use_derived, use_mono


def main() -> None:
    transform, loss, params, use_derived, use_mono = load_config()
    print(f"Setup: {transform}/{loss} | derived={use_derived} | monotonic={use_mono}", flush=True)
    print(f"Parameter: {params}", flush=True)

    df = load(sample=0, with_derived=use_derived)
    num = NUMERIC + (DERIVED if use_derived else [])
    feats = num + CATEGORICAL
    X = df[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=float)
    cat_idx = [feats.index(c) for c in CATEGORICAL]

    Xtr, Xte, ytr_raw, yte_raw = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    ytr = np.log1p(ytr_raw) if transform == "log" else ytr_raw

    p = dict(params); p.update({"iterations": 3000, "loss_function": loss,
                                "random_seed": RANDOM_STATE, "od_type": "Iter",
                                "od_wait": 80, "verbose": 200})
    if use_mono:
        p["monotone_constraints"] = [MONO.get(f, 0) if f in num else 0 for f in feats]

    print(f"Training auf vollem Datensatz: {len(Xtr):,} train / {len(Xte):,} test", flush=True)
    valid = Pool(Xte, np.log1p(yte_raw) if transform == "log" else yte_raw, cat_features=cat_idx)
    m = CatBoostRegressor(**p)
    m.fit(Pool(Xtr, ytr, cat_features=cat_idx), eval_set=valid)

    pred = m.predict(Xte)
    if transform == "log":
        pred = np.expm1(pred)
    mt = dollar_metrics(yte_raw, pred)
    print(f"\n=== Getuntes Finale (voller Datensatz) ===", flush=True)
    print(f"MAE ${mt['mae']:,.0f} | RMSE ${mt['rmse']:,.0f} | R2 {mt['r2']:.4f} | MAPE {mt['mape']:.1f}%", flush=True)
    print(f"Referenz (ungetunt): MAE ${REF_BASELINE_MAE:,}", flush=True)
    d = REF_BASELINE_MAE - mt["mae"]
    print(f"=> Verbesserung durch Tuning: {'-' if d>0 else '+'}${abs(d):,.0f} ({-d/REF_BASELINE_MAE*100:+.1f}%)", flush=True)

    seg = []
    for name, lo, hi in PRICE_SEGMENTS:
        mask = (yte_raw >= lo) & (yte_raw <= hi)
        if mask.sum() >= 10:
            from sklearn.metrics import mean_absolute_error
            seg.append((name, int(mask.sum()),
                        float(mean_absolute_error(yte_raw[mask], np.asarray(pred)[mask])),
                        float(np.mean(np.abs((yte_raw[mask]-np.asarray(pred)[mask])/yte_raw[mask]))*100)))

    m.save_model(str(HERE / "price_model_catboost_tuned.cbm"))
    out = {"setup": {"transform": transform, "loss": loss, "use_derived": use_derived,
                     "use_monotonic": use_mono, "params": params},
           "metrics": mt, "reference_mae": REF_BASELINE_MAE,
           "segments": [{"segment": s[0], "n": s[1], "mae": s[2], "mape": s[3]} for s in seg]}
    (HERE / "04_final_results.json").write_text(json.dumps(out, indent=2))
    print("Modell: price_model_catboost_tuned.cbm | Ergebnis: 04_final_results.json", flush=True)


if __name__ == "__main__":
    main()
