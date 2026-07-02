"""
Schritt 1 — CV-Gerüst + Zielgröße/Loss-Vergleich.

Frage: Sollte CatBoost den Rohpreis (MAE-Loss) oder den log-transformierten
Preis (RMSE bzw. MAE) vorhersagen? Autopreise sind rechtsschief/multiplikativ —
Log-Ziel hilft oft bei MAPE und im teuren Segment.

Methode: 4-fache Kreuzvalidierung auf einer Teilstichprobe (Geschwindigkeit),
Early Stopping je Fold. Alle Metriken werden in DOLLAR berechnet (Log-Vorhersagen
werden zurücktransformiert), damit die Orientierungen fair vergleichbar sind.

Aufruf: uv run python tuning_experiment/01_cv_target_loss.py
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

ORIENTATIONS = {
    "raw_mae":  {"transform": "raw", "loss": "MAE"},
    "log_rmse": {"transform": "log", "loss": "RMSE"},
    "log_mae":  {"transform": "log", "loss": "MAE"},
}


def main() -> None:
    df = load(sample=SAMPLE)
    print(f"Stichprobe: {len(df):,} Zeilen | {N_SPLITS}-fache CV", flush=True)
    feats = NUMERIC + CATEGORICAL
    X = df[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=float)
    cat_idx = [feats.index(c) for c in CATEGORICAL]

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, cfg in ORIENTATIONS.items():
        fold_maes, fold_mapes, fold_r2 = [], [], []
        for k, (tr, va) in enumerate(kf.split(X)):
            ytr = np.log1p(y[tr]) if cfg["transform"] == "log" else y[tr]
            yva = np.log1p(y[va]) if cfg["transform"] == "log" else y[va]
            train_pool = Pool(X.iloc[tr], ytr, cat_features=cat_idx)
            valid_pool = Pool(X.iloc[va], yva, cat_features=cat_idx)
            m = CatBoostRegressor(iterations=1500, depth=8, learning_rate=0.05,
                                  loss_function=cfg["loss"], l2_leaf_reg=3.0,
                                  random_seed=RANDOM_STATE, od_type="Iter", od_wait=50,
                                  verbose=0)
            m.fit(train_pool, eval_set=valid_pool)
            pred = m.predict(X.iloc[va])
            if cfg["transform"] == "log":
                pred = np.expm1(pred)
            mt = dollar_metrics(y[va], pred)
            fold_maes.append(mt["mae"]); fold_mapes.append(mt["mape"]); fold_r2.append(mt["r2"])
            print(f"  {name} Fold {k+1}: MAE ${mt['mae']:,.0f}  MAPE {mt['mape']:.1f}%", flush=True)
        results[name] = {"mae_mean": float(np.mean(fold_maes)), "mae_std": float(np.std(fold_maes)),
                         "mape_mean": float(np.mean(fold_mapes)), "r2_mean": float(np.mean(fold_r2))}
        print(f"==> {name}: MAE ${results[name]['mae_mean']:,.0f} ± {results[name]['mae_std']:,.0f} | "
              f"MAPE {results[name]['mape_mean']:.1f}% | R2 {results[name]['r2_mean']:.4f}\n", flush=True)

    best = min(results, key=lambda k: results[k]["mae_mean"])
    print(f"BESTE ORIENTIERUNG (nach MAE): {best}", flush=True)
    (HERE / "01_results.json").write_text(json.dumps({"results": results, "best": best,
                                                       "sample": SAMPLE, "n_splits": N_SPLITS}, indent=2))
    print("Ergebnis: 01_results.json", flush=True)


if __name__ == "__main__":
    main()
