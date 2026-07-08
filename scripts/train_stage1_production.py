from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


INPUT_PATH = Path("car_prices_clean.csv")
MODEL_PATH = Path("models/stage1_production_model.joblib")
METRICS_PATH = Path("models/stage1_production_metrics.json")
RESULTS_PATH = Path("docs/stage1/model_results_stage1_v2.md")
CURRENT_MODEL_PATH = Path("archive/model_artifacts_2026-07-08/stage1_legacy_histgb_model.joblib")

RANDOM_STATE = 42
TARGET = "sellingprice"

# Stage 1 V2 deliberately contains no sale month or macroeconomic feature.
# Its only job is to estimate the vehicle-specific base value. Stage 2 and
# Stage 3 remain responsible for market-level and seasonal adjustments.
NUMERIC_FEATURES = ["model_year", "vehicle_age", "odometer", "condition"]
CATEGORICAL_FEATURES = [
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
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CURRENT_MODEL_FEATURES = [
    "vehicle_age",
    "sale_month",
    "odometer",
    "condition",
    "year_month",
    "make",
    "model",
    "body",
]

PRICE_SEGMENTS = [
    ("Budget", 500, 5_000),
    ("Economy", 5_000, 10_000),
    ("Mid-Range", 10_000, 20_000),
    ("Premium", 20_000, 40_000),
    ("Luxury", 40_000, 150_000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a separate, time-neutral Stage 1 V2 vehicle model."
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_PATH)
    parser.add_argument("--results-output", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Maximum rows after cleaning; use 0 for the complete dataset.",
    )
    return parser.parse_args()


def load_data(path: Path, max_rows: int) -> pd.DataFrame:
    required = [
        "year",
        "saledate",
        "make",
        "model",
        "trim",
        "body",
        "transmission",
        "state",
        "condition",
        "odometer",
        "color",
        "interior",
        TARGET,
    ]
    df = pd.read_csv(path, usecols=required)
    df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce", utc=True)
    df = df.dropna(subset=required).copy()

    df["model_year"] = pd.to_numeric(df["year"], errors="coerce")
    df["sale_year"] = df["saledate"].dt.year
    df["sale_month"] = df["saledate"].dt.month
    df["year_month"] = df["saledate"].dt.strftime("%Y-%m")
    df["vehicle_age"] = (df["sale_year"] - df["model_year"]).clip(lower=0)

    df = df[
        df[TARGET].between(500, 150_000)
        & df["odometer"].between(1, 500_000)
        & df["vehicle_age"].between(0, 30)
        & df["condition"].between(1, 5)
    ].copy()

    for column in CATEGORICAL_FEATURES:
        if column == "make_model":
            continue
        df[column] = df[column].astype(str).str.strip().str.lower()
    df["make_model"] = df["make"] + "|" + df["model"]

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)

    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=20, sparse_output=True
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_xgboost(objective: str) -> XGBRegressor:
    return XGBRegressor(
        objective=objective,
        eval_metric="mae",
        n_estimators=700,
        learning_rate=0.045,
        max_depth=9,
        min_child_weight=12,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.02,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def build_model() -> VotingRegressor:
    raw_price_model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", build_xgboost("reg:squarederror")),
        ]
    )

    log_price_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", build_xgboost("reg:absoluteerror")),
        ]
    )
    log_price_model = TransformedTargetRegressor(
        regressor=log_price_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return VotingRegressor(
        estimators=[("raw_price", raw_price_model), ("log_price", log_price_model)],
        weights=[0.5, 0.5],
    )


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, prediction)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, prediction))), 2),
        "r2": round(float(r2_score(y_true, prediction)), 4),
        "mape_percent": round(
            float(np.mean(np.abs((y_true.to_numpy() - prediction) / y_true.to_numpy())) * 100),
            2,
        ),
    }


def segment_metrics(y_true: pd.Series, prediction: np.ndarray) -> list[dict]:
    result = []
    y_array = y_true.to_numpy()
    for name, low, high in PRICE_SEGMENTS:
        mask = (y_array >= low) & (y_array <= high)
        result.append(
            {
                "segment": name,
                "n": int(mask.sum()),
                "mae": round(float(mean_absolute_error(y_array[mask], prediction[mask])), 2),
                "mape_percent": round(
                    float(np.mean(np.abs((y_array[mask] - prediction[mask]) / y_array[mask])) * 100),
                    2,
                ),
            }
        )
    return result


def evaluate_current_model(test: pd.DataFrame) -> tuple[dict | None, str | None]:
    if not CURRENT_MODEL_PATH.exists():
        return None, "The existing Stage 1 model file was not found."
    try:
        current = joblib.load(CURRENT_MODEL_PATH)
        prediction = current.predict(test[CURRENT_MODEL_FEATURES])
        return metrics(test[TARGET], prediction), None
    except Exception as error:
        return None, f"Existing model comparison failed: {error}"


def write_results(payload: dict, path: Path) -> None:
    new = payload["v2_metrics"]
    old = payload.get("current_model_metrics_same_test")
    lines = [
        "# Stage 1 V2 — Vehicle Base Price Model",
        "",
        "## Purpose",
        "",
        "Stage 1 V2 is a separate vehicle-value model that runs before Stage 2 and Stage 3. "
        "The existing production model remains unchanged as a backup.",
        "",
        "Unlike the existing model, V2 deliberately excludes sale month and `year_month`. "
        "Stage 2 remains responsible for market movement and Stage 3 for seasonality.",
        "",
        "## Architecture",
        "",
        "- 50/50 ensemble of two XGBoost gradient-boosted tree models",
        "- one component predicts the raw dollar price; one predicts the log-transformed price",
        "- one-hot encoding for categorical values",
        "- additional vehicle information: trim, transmission, state, exterior and interior color",
        "- explicit make-model interaction selected on a separate validation split",
        "- no VIN, seller or MMR feature; MMR was excluded to avoid target-like leakage",
        "- architecture and ensemble weight selected without using the final test split",
        "",
        "## Evaluation",
        "",
        f"- Rows: {payload['rows_used']:,}",
        f"- Train/test split: {payload['train_rows']:,} / {payload['test_rows']:,}",
        f"- V2 MAE: **${new['mae']:,.2f}**",
        f"- V2 RMSE: **${new['rmse']:,.2f}**",
        f"- V2 R²: **{new['r2']:.4f}**",
        f"- V2 MAPE: **{new['mape_percent']:.2f}%**",
    ]
    if old:
        change = old["mae"] - new["mae"]
        change_percent = change / old["mae"] * 100
        lines += [
            f"- Existing model MAE on exactly the same V2 test rows: **${old['mae']:,.2f}**",
            f"- MAE improvement: **${change:,.2f} ({change_percent:.2f}%)**",
        ]
    elif payload.get("current_model_comparison_error"):
        lines.append(
            f"- Existing-model comparison unavailable: {payload['current_model_comparison_error']}"
        )

    lines += [
        "",
        "The ensemble is optimized for MAE in dollars. Compared with the single log model, "
        "it lowers MAE but can trade off some RMSE or percentage-error performance.",
        "",
        "## Error by price segment",
        "",
        "| Segment | Test rows | MAE | MAPE |",
        "|---|---:|---:|---:|",
    ]
    for segment in payload["segment_metrics"]:
        lines.append(
            f"| {segment['segment']} | {segment['n']:,} | "
            f"${segment['mae']:,.2f} | {segment['mape_percent']:.2f}% |"
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python scripts/train_stage1_production.py --max-rows 0",
        "```",
        "",
        "Omit `--max-rows 0` for a faster 200,000-row development run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_data(args.input, args.max_rows)
    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Rows: {len(data):,}; train: {len(train):,}; test: {len(test):,}")
    model = build_model()
    model.fit(train[FEATURES], train[TARGET])
    prediction = np.maximum(model.predict(test[FEATURES]), 0)

    v2_metrics = metrics(test[TARGET], prediction)
    current_metrics, comparison_error = evaluate_current_model(test)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "Stage 1 V2 XGBoost raw/log ensemble",
        "model_path": str(args.model_output),
        "rows_used": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "max_rows_argument": args.max_rows,
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "excluded_to_prevent_leakage_or_overlap": [
            "vin",
            "seller",
            "mmr",
            "sale_month",
            "year_month",
        ],
        "v2_metrics": v2_metrics,
        "current_model_metrics_same_test": current_metrics,
        "current_model_comparison_error": comparison_error,
        "segment_metrics": segment_metrics(test[TARGET], prediction),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)
    args.metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_results(payload, args.results_output)

    print(json.dumps(payload["v2_metrics"], indent=2))
    if current_metrics:
        improvement = current_metrics["mae"] - v2_metrics["mae"]
        print(f"Current model MAE on same test rows: ${current_metrics['mae']:,.2f}")
        print(f"V2 MAE improvement: ${improvement:,.2f}")
    print(f"Saved model: {args.model_output}")


if __name__ == "__main__":
    main()
