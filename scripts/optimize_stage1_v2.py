from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from train_stage1_v2 import (
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    load_data,
)


OUTPUT = Path("archive/model_artifacts_2026-07-08/stage1_xgboost_optimization.json")
TUNING_ROWS = 200_000

CATEGORICAL = [
    "make",
    "model",
    "trim",
    "body",
    "transmission",
    "state",
    "color",
    "interior",
    "make_model",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL

TRIALS = [
    {"name": "log_squared", "objective": "reg:squarederror", "target_mode": "log", "max_depth": 9, "min_child_weight": 12, "learning_rate": 0.045},
    {"name": "log_absolute", "objective": "reg:absoluteerror", "target_mode": "log", "max_depth": 9, "min_child_weight": 12, "learning_rate": 0.045},
    {"name": "raw_squared", "objective": "reg:squarederror", "target_mode": "raw", "max_depth": 9, "min_child_weight": 12, "learning_rate": 0.045},
    {"name": "raw_absolute", "objective": "reg:absoluteerror", "target_mode": "raw", "max_depth": 9, "min_child_weight": 12, "learning_rate": 0.045},
]


def add_interactions(data):
    data = data.copy()
    data["make_model"] = data["make"] + "|" + data["model"]
    return data


def main() -> None:
    data = add_interactions(load_data(Path("car_prices_clean.csv"), TUNING_ROWS))
    development, untouched_test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE
    )
    train, validation = train_test_split(
        development, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                CATEGORICAL,
            ),
        ]
    )
    x_train = preprocessor.fit_transform(train[FEATURES])
    x_validation = preprocessor.transform(validation[FEATURES])
    y_train_raw = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()

    results = []
    predictions = {}
    for trial in TRIALS:
        print(f"Training {trial['name']}...", flush=True)
        model = XGBRegressor(
            objective=trial["objective"],
            eval_metric="mae",
            n_estimators=700,
            max_depth=trial["max_depth"],
            min_child_weight=trial["min_child_weight"],
            learning_rate=trial["learning_rate"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.02,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        started = time.perf_counter()
        training_target = (
            np.log1p(y_train_raw) if trial["target_mode"] == "log" else y_train_raw
        )
        model.fit(
            x_train,
            training_target,
            verbose=False,
        )
        prediction = model.predict(x_validation)
        if trial["target_mode"] == "log":
            prediction = np.expm1(prediction)
        mae = float(mean_absolute_error(y_validation, prediction))
        predictions[trial["name"]] = prediction
        result = {
            **trial,
            "iterations": 700,
            "validation_mae": round(mae, 2),
            "seconds": round(time.perf_counter() - started, 2),
        }
        results.append(result)
        print(result)

    # Check simple two-model ensembles without touching the held-out test set.
    ranked = sorted(results, key=lambda item: item["validation_mae"])
    first, second = ranked[0]["name"], ranked[1]["name"]
    ensembles = []
    for weight in (0.25, 0.5, 0.75):
        prediction = weight * predictions[first] + (1 - weight) * predictions[second]
        ensembles.append(
            {
                "models": [first, second],
                "first_model_weight": weight,
                "validation_mae": round(float(mean_absolute_error(y_validation, prediction)), 2),
            }
        )

    payload = {
        "method": "200k development sample; 64% train, 16% validation, 20% untouched test",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "untouched_test_rows": len(untouched_test),
        "features": FEATURES,
        "trials": sorted(results, key=lambda item: item["validation_mae"]),
        "ensembles": sorted(ensembles, key=lambda item: item["validation_mae"]),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
