from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from train_price_model import (
    FEATURE_COLUMNS,
    PRICE_SEGMENTS,
    RANDOM_STATE,
    TARGET_COLUMN,
    evaluate_by_segment,
    evaluate_model,
    load_modeling_data,
)

FEATURES_PATH = Path("car_prices_features.csv")
MODEL_PATH = Path("models/price_model.joblib")
METRICS_PATH = Path("models/price_model_metrics.json")
RESULTS_PATH = Path("model_results.md")


def print_segment_table(segment_metrics: list[dict]) -> None:
    header = f"{'Segment':<12} {'Price Range':<22} {'N':>7} {'MAE':>9} {'RMSE':>9} {'MAPE':>8}"
    print(header)
    print("-" * len(header))
    for s in segment_metrics:
        print(
            f"{s['segment']:<12} {s['price_range']:<22} {s['n']:>7,} "
            f"${s['mae']:>8,.0f} ${s['rmse']:>8,.0f} {s['mape_percent']:>7.1f}%"
        )


def update_metrics_json(segment_metrics: list[dict]) -> None:
    if not METRICS_PATH.exists():
        print(f"Metrics file not found: {METRICS_PATH} — skipping JSON update.")
        return
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    data["segment_metrics"] = segment_metrics
    METRICS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated {METRICS_PATH}")


def update_results_markdown(segment_metrics: list[dict]) -> None:
    if not RESULTS_PATH.exists():
        print(f"Results file not found: {RESULTS_PATH} — skipping markdown update.")
        return

    segment_table = "\n".join(
        f"| {s['segment']} | {s['price_range']} | {s['n']:,} | ${s['mae']:,.0f} | ${s['rmse']:,.0f} | {s['mape_percent']:.1f}% |"
        for s in segment_metrics
    )
    new_section = (
        "## Error by Price Segment\n\n"
        "| Segment | Price Range | Test Rows | MAE | RMSE | MAPE |\n"
        "|---|---|---:|---:|---:|---:|\n"
        f"{segment_table}\n"
    )

    content = RESULTS_PATH.read_text(encoding="utf-8")

    if "## Error by Price Segment" in content:
        # Replace existing section up to the next ## heading
        import re
        content = re.sub(
            r"## Error by Price Segment.*?(?=\n## |\Z)",
            new_section,
            content,
            flags=re.DOTALL,
        )
    else:
        # Insert before "## Most Important Features"
        content = content.replace(
            "## Most Important Features",
            new_section + "\n## Most Important Features",
        )

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"Updated {RESULTS_PATH}")


def main() -> None:
    print("Loading data...")
    df = load_modeling_data(FEATURES_PATH, max_rows=0)

    _, X_test, _, y_test = train_test_split(
        df[FEATURE_COLUMNS],
        df[TARGET_COLUMN],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    overall = evaluate_model(y_test, predictions)
    print(f"\nOverall — MAE: ${overall['mae']:,.0f}  RMSE: ${overall['rmse']:,.0f}  R2: {overall['r2']:.4f}  MAPE: {overall['mape_percent']:.1f}%")

    segment_metrics = evaluate_by_segment(y_test, predictions)

    print("\nError by Price Segment:")
    print_segment_table(segment_metrics)

    update_metrics_json(segment_metrics)
    update_results_markdown(segment_metrics)


if __name__ == "__main__":
    main()
