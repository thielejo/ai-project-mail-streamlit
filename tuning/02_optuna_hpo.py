"""
Schritt 2 — Hyperparameter-Optimierung mit Optuna (Bayesian/TPE).

Optimiert die CatBoost-Parameter auf der besten Zielgröße/Loss-Orientierung aus
Schritt 1. Nutzt einen 80/20-Validierungssplit der Teilstichprobe je Trial
(schneller als CV-pro-Trial) mit Early Stopping.

RESUME-SICHER: Optuna-Study in SQLite. Bei Abbruch erneut starten — die Study
wird fortgesetzt, bis N_TRIALS erreicht ist.

Aufruf: uv run python tuning/02_optuna_hpo.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load, dollar_metrics, NUMERIC, CATEGORICAL, TARGET, RANDOM_STATE  # noqa

SAMPLE = 150_000
N_TRIALS = 40
HERE = Path(__file__).resolve().parent
STORAGE = f"sqlite:///{HERE / 'optuna_study.db'}"

# Orientierung aus Schritt 1 (Fallback log_rmse; wird aus 01_results.json gelesen)
TRANSFORM, LOSS = "log", "RMSE"
res1 = HERE / "01_results.json"
if res1.exists():
    best = json.loads(res1.read_text())["best"]
    TRANSFORM = "log" if "log" in best else "raw"
    LOSS = "MAE" if best.endswith("mae") else "RMSE"


def main() -> None:
    df = load(sample=SAMPLE)
    feats = NUMERIC + CATEGORICAL
    X = df[feats].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(dtype=float)
    cat_idx = [feats.index(c) for c in CATEGORICAL]
    Xtr, Xva, ytr_raw, yva_raw = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    ytr = np.log1p(ytr_raw) if TRANSFORM == "log" else ytr_raw
    train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
    valid_pool = Pool(Xva, np.log1p(yva_raw) if TRANSFORM == "log" else yva_raw, cat_features=cat_idx)
    print(f"Orientierung: {TRANSFORM}/{LOSS} | Stichprobe {len(df):,} | {N_TRIALS} Trials", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "depth": trial.suggest_int("depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 3.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "loss_function": LOSS,
            "random_seed": RANDOM_STATE, "od_type": "Iter", "od_wait": 60, "verbose": 0,
        }
        m = CatBoostRegressor(**params)
        m.fit(train_pool, eval_set=valid_pool)
        pred = m.predict(Xva)
        if TRANSFORM == "log":
            pred = np.expm1(pred)
        return dollar_metrics(yva_raw, pred)["mae"]

    study = optuna.create_study(study_name="catboost_hpo", storage=STORAGE,
                                direction="minimize", load_if_exists=True)
    done = len(study.trials)
    remaining = max(0, N_TRIALS - done)
    print(f"Bereits {done} Trials, fahre fort mit {remaining}.", flush=True)
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    print(f"\nBester MAE (Valid): ${study.best_value:,.0f}", flush=True)
    print("Beste Parameter:", json.dumps(study.best_params, indent=2), flush=True)
    (HERE / "02_best_params.json").write_text(json.dumps(
        {"transform": TRANSFORM, "loss": LOSS, "best_value_mae": study.best_value,
         "best_params": study.best_params, "n_trials": len(study.trials)}, indent=2))
    print("Ergebnis: 02_best_params.json", flush=True)


if __name__ == "__main__":
    main()
