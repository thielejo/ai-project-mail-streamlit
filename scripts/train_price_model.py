from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


INPUT_PATH = Path("car_prices_features.csv")
MODEL_PATH = Path("models/price_model.joblib")
METRICS_PATH = Path("models/price_model_metrics.json")
RESULTS_PATH = Path("model_results.md")
RANDOM_STATE = 42

TARGET_COLUMN = "sellingprice"
NUMERIC_FEATURES = ["vehicle_age", "sale_month", "odometer", "condition"]
CATEGORICAL_FEATURES = ["year_month", "make", "model", "body"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Stage 1 vehicle price model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Feature CSV created by scripts/build_features.py.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_PATH,
        help="Where the trained model should be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=METRICS_PATH,
        help="Where machine-readable metrics should be saved.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=RESULTS_PATH,
        help="Where the human-readable results summary should be saved.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Maximum rows used for quick baseline training. Use 0 for all rows.",
    )
    return parser.parse_args()


def load_modeling_data(path: Path, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    df = df.dropna(subset=required_columns).copy()

    # Remove extreme values that would dominate the first baseline model.
    df = df[
        (df[TARGET_COLUMN].between(500, 150_000))
        & (df["odometer"].between(1, 500_000))
        & (df["vehicle_age"].between(0, 30))
    ].copy()

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE).copy()

    return df


def build_xgboost_pipeline() -> TransformedTargetRegressor:
    from xgboost import XGBRegressor

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=True,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_sklearn_pipeline() -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_pipeline() -> tuple[TransformedTargetRegressor, str, str | None]:
    try:
        return build_xgboost_pipeline(), "XGBoost Regressor", None
    except Exception as error:
        error_text = str(error)
        if "libomp" in error_text:
            fallback_reason = (
                "XGBoost could not be loaded because the macOS OpenMP runtime "
                "`libomp.dylib` is missing. Install it with `brew install libomp` "
                "to enable XGBoost locally."
            )
        else:
            fallback_reason = f"XGBoost was unavailable in this environment: {error_text}"

        return (
            build_sklearn_pipeline(),
            "HistGradientBoostingRegressor",
            fallback_reason,
        )


PRICE_SEGMENTS = [
    ("Budget", 500, 5_000),
    ("Economy", 5_000, 10_000),
    ("Mid-Range", 10_000, 20_000),
    ("Premium", 20_000, 40_000),
    ("Luxury", 40_000, 150_000),
]


def evaluate_model(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
        "mape_percent": round(float(mape), 2),
    }


def evaluate_by_segment(
    y_true: pd.Series, predictions: np.ndarray
) -> list[dict]:
    pred_series = pd.Series(predictions, index=y_true.index)
    results = []
    for label, low, high in PRICE_SEGMENTS:
        mask = y_true.between(low, high)
        n = int(mask.sum())
        if n < 10:
            continue
        seg_true = y_true[mask]
        seg_pred = pred_series[mask]
        mae = mean_absolute_error(seg_true, seg_pred)
        rmse = np.sqrt(mean_squared_error(seg_true, seg_pred))
        mape = np.mean(np.abs((seg_true - seg_pred) / seg_true)) * 100
        results.append({
            "segment": label,
            "price_range": f"${low:,}–${high:,}",
            "n": n,
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "mape_percent": round(float(mape), 2),
        })
    return results


def get_top_feature_importance(
    model: TransformedTargetRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict[str, float | str]]:
    pipeline = model.regressor_
    preprocessor = pipeline.named_steps["preprocessor"]
    fitted_model = pipeline.named_steps["model"]

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = np.array(FEATURE_COLUMNS)

    importances = getattr(fitted_model, "feature_importances_", None)
    if importances is None:
        sample_size = min(2_000, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index]
        permutation = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        feature_names = np.array(FEATURE_COLUMNS)
        importances = permutation.importances_mean

    ranked_indices = np.argsort(importances)[::-1][:15]

    return [
        {
            "feature": str(feature_names[index]),
            "importance": round(float(importances[index]), 4),
        }
        for index in ranked_indices
    ]


def write_results_markdown(
    path: Path,
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    top_features: list[dict[str, float | str]],
    segment_metrics: list[dict],
    rows_used: int,
    test_rows: int,
    model_path: Path,
    model_name: str,
    fallback_reason: str | None,
) -> None:
    if top_features:
        feature_table = "\n".join(
            f"| {item['feature']} | {item['importance']} |" for item in top_features
        )
    else:
        feature_table = "| Not available for this model type | n/a |"

    if segment_metrics:
        segment_table = "\n".join(
            f"| {s['segment']} | {s['price_range']} | {s['n']:,} | ${s['mae']:,.0f} | ${s['rmse']:,.0f} | {s['mape_percent']:.1f}% |"
            for s in segment_metrics
        )
    else:
        segment_table = "| n/a | n/a | n/a | n/a | n/a | n/a |"

    fallback_note = ""
    if fallback_reason:
        fallback_note = f"""
## Environment Note

{fallback_reason}

The script automatically used a sklearn gradient boosting model so the baseline can still be trained locally. Once the missing XGBoost system dependency is installed, the same script will use XGBoost.
"""

    content = f"""# Stage 1 Model Results

This document summarizes the first vehicle price prediction model.

## Goal

The Stage 1 model predicts the baseline selling price from vehicle-level features only.

## Input Features

- `vehicle_age`
- `sale_month`
- `odometer`
- `condition`
- `year_month`
- `make`
- `model`
- `body`

Target variable:

- `sellingprice`

## Training Setup

- Input file: `car_prices_features.csv`
- Rows used for training/evaluation: {rows_used:,}
- Test split: 20%
- Test rows: {test_rows:,}
- Model: {model_name}
- Text features: encoded for model training
- Target transformation: `log1p(sellingprice)`
- Saved model: `{model_path}`
{fallback_note}

## Results

| Metric | {model_name} | Median-price baseline |
|---|---:|---:|
| MAE | ${metrics["mae"]:,.2f} | ${baseline_metrics["mae"]:,.2f} |
| RMSE | ${metrics["rmse"]:,.2f} | ${baseline_metrics["rmse"]:,.2f} |
| R2 | {metrics["r2"]:.4f} | {baseline_metrics["r2"]:.4f} |
| MAPE | {metrics["mape_percent"]:.2f}% | {baseline_metrics["mape_percent"]:.2f}% |

## Error by Price Segment

| Segment | Price Range | Test Rows | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|---:|
{segment_table}

## Most Important Features

| Feature | Importance |
|---|---:|
{feature_table}

## How to Reproduce

Run this command from the project root:

```bash
uv run python scripts/train_price_model.py
```

For full-dataset training, run:

```bash
uv run python scripts/train_price_model.py --max-rows 0
```
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_modeling_data(args.input, args.max_rows)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model, model_name, fallback_reason = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = evaluate_model(y_test, predictions)
    segment_metrics = evaluate_by_segment(y_test, predictions)

    median_prediction = np.full(shape=len(y_test), fill_value=y_train.median())
    baseline_metrics = evaluate_model(y_test, median_prediction)

    top_features = get_top_feature_importance(model, X_test, y_test)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_output)

    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "model_path": str(args.model_output),
        "model_name": model_name,
        "fallback_reason": fallback_reason,
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "metrics": metrics,
        "median_baseline_metrics": baseline_metrics,
        "segment_metrics": segment_metrics,
        "top_features": top_features,
    }
    args.metrics_output.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_results_markdown(
        path=args.results_output,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        top_features=top_features,
        segment_metrics=segment_metrics,
        rows_used=len(df),
        test_rows=len(X_test),
        model_path=args.model_output,
        model_name=model_name,
        fallback_reason=fallback_reason,
    )

    print("Stage 1 price model trained successfully.")
    print(f"Model: {model_name}")
    if fallback_reason:
        print(f"Note: {fallback_reason}")
    print(f"Rows used: {len(df):,}")
    print(f"MAE: ${metrics['mae']:,.2f}")
    print(f"RMSE: ${metrics['rmse']:,.2f}")
    print(f"R2: {metrics['r2']:.4f}")
    print(f"Model saved to: {args.model_output}")
    print(f"Results written to: {args.results_output}")


if __name__ == "__main__":
    main()
