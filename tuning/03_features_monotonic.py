"""
Schritt 3 — Abgeleitete Features + Monotonie-Constraints.

Baut auf den besten HPO-Parametern (Schritt 2) auf und prüft per 4-facher CV
zwei Erweiterungen:
  (b) + abgeleitete Features: miles_per_year, age_squared, disp_per_cond
  (c) + Monotonie-Constraints: Preis fällt monoton mit odometer/Alter,
       steigt mit condition/Baujahr/Hubraum (Domänenwissen, Plausibilität).

Aufruf: uv run python tuning/03_features_monotonic.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, dollar_metrics, NUMERIC, CATEGORICAL, TARGET, RANDOM_STATE  # noqa

SAMPLE = 150_000
N_SPLITS = 4
HERE = Path(__file__).resolve().parent

DERIVED = ["miles_per_year", "age_squared", "disp_per_cond"]
# Monotonie-Richtungen (nur numerische Features)
MONO = {"model_year": 1, "vehicle_age": -1, "odometer": -1, "condition": 1,
        "displacement": 1, "miles_per_year": -1, "age_squared": -1, "disp_per_cond": 0}

# Beste Orientierung + Parameter aus Schritt 1/2 laden (mit Fallback)
TRANSFORM, LOSS = "log", "RMSE"
params = {"iterations": 2000, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0}
if (HERE / "02_best_params.json").exists():
    b = json.loads((HERE / "02_best_params.json").read_text())
    TRANSFORM, LOSS = b["transform"], b["loss"]
    params.update(b["best_params"]); params["iterations"] = 2000


def cv_mae(df, num_feats, mono_list=None) -> dict:
    feats = num_feats + CATEGORICAL
    X = df[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=float)
    cat_idx = [feats.index(c) for c in CATEGORICAL]
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    maes, mapes, r2s = [], [], []
    for tr, va in kf.split(X):
        ytr = np.log1p(y[tr]) if TRANSFORM == "log" else y[tr]
        yva = np.log1p(y[va]) if TRANSFORM == "log" else y[va]
        p = dict(params); p.update({"loss_function": LOSS, "random_seed": RANDOM_STATE,
                                    "od_type": "Iter", "od_wait": 60, "verbose": 0})
        if mono_list is not None:
            p["monotone_constraints"] = mono_list
        m = CatBoostRegressor(**p)
        m.fit(Pool(X.iloc[tr], ytr, cat_features=cat_idx),
              eval_set=Pool(X.iloc[va], yva, cat_features=cat_idx))
        pred = m.predict(X.iloc[va])
        if TRANSFORM == "log":
            pred = np.expm1(pred)
        mt = dollar_metrics(y[va], pred)
        maes.append(mt["mae"]); mapes.append(mt["mape"]); r2s.append(mt["r2"])
    return {"mae": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "mape": float(np.mean(mapes)), "r2": float(np.mean(r2s))}


def main() -> None:
    print(f"Orientierung {TRANSFORM}/{LOSS} | Stichprobe {SAMPLE:,} | {N_SPLITS}-fache CV", flush=True)
    df = load(sample=SAMPLE, with_derived=True)

    # Resume-sicher: bereits berechnete Varianten aus 03_results.json übernehmen
    out_path = HERE / "03_results.json"
    out = {"transform": TRANSFORM, "loss": LOSS}
    if out_path.exists():
        out.update(json.loads(out_path.read_text()))

    def save():
        out_path.write_text(json.dumps(out, indent=2))

    num_b = NUMERIC + DERIVED
    feats_c = num_b + CATEGORICAL
    mono_list = [MONO.get(f, 0) if f in num_b else 0 for f in feats_c]

    if "a_baseline" not in out:
        print("(a) Tuned-Baseline (nur Basis-Features) ...", flush=True)
        out["a_baseline"] = cv_mae(df, NUMERIC); save()
        a = out["a_baseline"]
        print(f"    MAE ${a['mae']:,.0f} ± {a['mae_std']:,.0f} | MAPE {a['mape']:.1f}% | R2 {a['r2']:.4f} [gespeichert]", flush=True)
    else:
        print("(a) bereits vorhanden — übersprungen", flush=True)

    if "b_derived" not in out:
        print("(b) + abgeleitete Features ...", flush=True)
        out["b_derived"] = cv_mae(df, num_b); save()
        b = out["b_derived"]
        print(f"    MAE ${b['mae']:,.0f} ± {b['mae_std']:,.0f} | MAPE {b['mape']:.1f}% | R2 {b['r2']:.4f} [gespeichert]", flush=True)
    else:
        print("(b) bereits vorhanden — übersprungen", flush=True)

    if "c_derived_monotonic" not in out:
        print("(c) + Monotonie-Constraints ...", flush=True)
        out["c_derived_monotonic"] = cv_mae(df, num_b, mono_list=mono_list); save()
        c = out["c_derived_monotonic"]
        print(f"    MAE ${c['mae']:,.0f} ± {c['mae_std']:,.0f} | MAPE {c['mape']:.1f}% | R2 {c['r2']:.4f} [gespeichert]", flush=True)
    else:
        print("(c) bereits vorhanden — übersprungen", flush=True)

    print("Ergebnis vollständig: 03_results.json", flush=True)


if __name__ == "__main__":
    main()
